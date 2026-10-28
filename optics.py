# -*- coding: utf-8 -*-
"""
optics.py —— Hopkins 部分相干成像前向模型。

物理模型
--------
投影光刻的成像过程由 Hopkins 方程描述（部分相干成像）:

    I(x) = ∫∫ TCC(f', f'') · M(f') · M*(f'') · exp(j·2π·(f'-f'')·x) df' df''

其中 M(f) 是掩模透射谱，TCC 为交叉传递系数（Transmission Cross Coefficient）:

    TCC(f', f'') = ∫ S(f) · P(f+f') · P*(f+f'') df

S(f) 为光源强度分布，P(f) 为投影光瞳函数。物理含义：光源上不同方向的
平面波照明彼此相干地产生强度贡献，部分相干度由光源的有限张角决定。

数值实现：光源点采样法（Abbe 法）
----------------------------------
把光源离散成若干点 {f_s, w_s}（环形光源按径向环 × 方位角采样），
每个光源点对应一束倾斜平面波，其像场为

    E_s(x) = FT⁻¹[ M(f) · P(f+f_s) ](x)

Hopkins 方程可严格改写为对这些点源的强度求和（数学上等价，非近似——
近似只来自光源采样点数）:

    I(x) = Σ_s w_s · |E_s(x)|²

对实值掩模和对称光瞳，±f_s 给出相同的像，因此方位角采样必须成对，
这也是"对称图形必须得到对称像"的数值保障。

约定（重要）
-----------
- 掩模像素 mask=1 表示不透明图形（铬/暗），透射场 T(x) = 1 - mask，
  即"亮场、暗图形"：无图形处归一化强度为 1（开域强度），图形内部趋近 0。
- 印刷判定：归一化强度 < RESIST_THRESHOLD 处形成图形（暗）。
- 频率域使用 np.fft.fftfreq 的自然序网格（DC 在索引 0，负频率在数组尾部），
  空间域 FFT 三件套配对：ifftshift(掩模) → fft2 → 与光瞳相乘 → ifft2 → fftshift。
- 带宽分析：掩模谱带宽 |f| ≤ (1+σ_out)·NA/λ（被光瞳截断），
  默认参数下 = 0.0133 nm⁻¹，远小于像素网格的奈奎斯特频率 0.25 nm⁻¹，
  因此像素化锯齿（亚分辨率高频）对空间像几乎无影响——这是像素化版图
  可以用于 OPC 的物理原因。
"""
import copy
import types

import numpy as np
from scipy import ndimage


# ------------------------------------------------------------------
# 光源与光瞳预计算
# ------------------------------------------------------------------

def _n_fft(cfg):
    """FFT 网格尺寸。周期边界模式 = GRID_N（视场即一个周期）；
    孤立边界模式 = 2×GRID_N（零填充后取中心，模拟线性卷积、无环绕）。"""
    if getattr(cfg, "BOUNDARY_MODE", "periodic") == "isolated":
        return 2 * cfg.GRID_N
    return cfg.GRID_N


def precompute_source_and_pupil(cfg):
    """采样光源、构建每个光源点对应的光瞳数组。只与光学参数有关，
    与掩模无关，整个 OPC 迭代过程只需调用一次。

    边界条件（重要）：
      周期边界 'periodic'（默认）——视场就是版图的一个周期，FFT 的
        循环卷积天然给出周期边界；周期版图的谱只含离散谐波，光瞳仅在
        这些谐波上取值，因此该实现对周期模型是**精确**的（无光瞳采样误差）。
      孤立边界 'isolated'——把掩模零填充到 2N 再做 FFT（线性卷积），
        结果取中心 N×N，消除相邻周期串扰；代价是 4 倍 FFT 时间。

    返回 dict:
      weights : 光源点权重 w_s（已归一化 Σw=1）
      pupils  : 每个光源点一张 (n_fft,n_fft) 复数光瞳数组 P(f+f_s)（自然序）
      i_clear : 开域强度解析值（空掩模的空间像 = 1，用于归一化）
      n_fft   : FFT 网格尺寸
    """
    N, dx = _n_fft(cfg), cfg.PIXEL_SIZE_NM
    lam, na = cfg.WAVELENGTH_NM, cfg.NA
    fc = na / lam                       # 光瞳截断频率 = NA/λ (nm⁻¹)

    # ---- 频率网格（自然序）：FX[i,j] = fx[j], FY[i,j] = fy[i]，
    #      与 fft2 输出 M[k_y, k_x] 的布局一致 ----
    fx = np.fft.fftfreq(N, d=dx)
    FY, FX = np.meshgrid(fx, fx, indexing="xy")

    # ---- 光源采样：径向环（取环中点，避免落在 σ 边界上）× 方位角 ----
    # 环形照明 σ ∈ [σ_in, σ_out]；圆形照明即 σ_in=0（天然避开轴上点 σ=0）
    sig_in = 0.0 if cfg.ILLUM_TYPE == "circle" else cfg.SIGMA_IN
    sig_out = cfg.SIGMA_OUT
    dsigma = (sig_out - sig_in) / cfg.N_SOURCE_RINGS
    sigmas = sig_in + (np.arange(cfg.N_SOURCE_RINGS) + 0.5) * dsigma
    thetas = 2.0 * np.pi * np.arange(cfg.N_SOURCE_PTS) / cfg.N_SOURCE_PTS
    # 方位角从 0 均匀起、每环点数相同 → 采样点 {f_s} 关于原点严格成对
    weights, fs_list = [], []
    for sr in sigmas:
        for th in thetas:
            fs_list.append((fc * sr * np.cos(th), fc * sr * np.sin(th)))
            # 等面积加权：环面积 ∝ 2π·σ_r·dσ → 权重 ∝ σ_r
            weights.append(sr)
    weights = np.array(weights) / np.sum(weights)

    # ---- 每个光源点一张光瞳数组（含离焦相位） ----
    pupils = []
    for fsx, fsy in fs_list:
        nu = np.sqrt((FX + fsx) ** 2 + (FY + fsy) ** 2) / fc   # 归一化频率
        P = np.zeros((N, N), dtype=np.complex128)
        inside = nu <= 1.0
        # 光瞳透过：|ν|≤1 处为 1；离焦相位（标量模型）：
        #   exp[j·(2π/λ)·Δz·(√(1−(NA·ν)²) − 1)]，±Δz 互为共轭 → I(+z)=I(−z)
        if cfg.DEFOCUS_NM != 0.0:
            ph = (2 * np.pi / lam) * cfg.DEFOCUS_NM * \
                 (np.sqrt(np.maximum(1.0 - (na * nu[inside]) ** 2, 0.0)) - 1.0)
            P[inside] = np.exp(1j * ph)
        else:
            P[inside] = 1.0
        pupils.append(P)

    # ---- 开域强度（解析值）：空掩模的像。|P|² 在光瞳内恒为 1（相位只有
    #      模 1），因此 i_clear = Σw = 1；仍显式计算以保持代码普适 ----
    i_clear = float(sum(w * abs(P[0, 0]) ** 2 for w, P in zip(weights, pupils)))

    return {"weights": weights, "pupils": pupils, "i_clear": i_clear,
            "n_fft": N}


# ------------------------------------------------------------------
# 空间像计算
# ------------------------------------------------------------------

def aerial_image(mask, cfg, precomp):
    """计算归一化空间像 I(x)（开域 = 1）。

    边界条件：
      周期（默认）：掩模直接做 FFT，视场即一个周期——循环卷积给出严格的
        周期边界，相邻周期之间没有间隙、没有零填充。
      孤立：掩模零填充到 2N 后计算，再裁中心 N×N（线性卷积，无环绕）。

    参数:
      mask     : 像素版图（1=不透明图形），shape (N,N)
      precomp  : precompute_source_and_pupil 的返回值
    返回:
      归一化空间像 (N,N) float64，物理含义为相对开域强度的剂量分布
    """
    N = cfg.GRID_N
    n_fft = precomp.get("n_fft", _n_fft(cfg))
    # 透射场：不透明处为 0、背景为 1（亮场暗图形约定）
    T = 1.0 - mask.astype(np.float64)
    if n_fft != N:
        # 孤立边界：把视场外填充为"清晰场"（透射 T=1，即该区域无图形），
        # 这样既没有对面周期的串扰，也不会在视场边缘引入虚假的暗区
        pad = n_fft - N
        T = np.pad(T, ((pad // 2, pad - pad // 2),
                       (pad // 2, pad - pad // 2)),
                   mode="constant", constant_values=1.0)
    # FFT 三件套之一：ifftshift 把图像中心（原点）移到数组 [0,0] 角
    M = np.fft.fft2(np.fft.ifftshift(T))

    I = np.zeros((n_fft, n_fft), dtype=np.float64)
    for w_s, P_s in zip(precomp["weights"], precomp["pupils"]):
        # ifft2 自带 1/N² 归一化；fftshift 把原点移回图像中心
        E_s = np.fft.fftshift(np.fft.ifft2(M * P_s))
        I += w_s * np.abs(E_s) ** 2
    if n_fft != N:
        # 裁掉填充区，回到 GRID_N 视场
        c0 = (n_fft - N) // 2
        I = I[c0:c0 + N, c0:c0 + N]
    return I / precomp["i_clear"]       # 开域归一化（阈值才有剂量含义）


def apply_resist_blur(aerial, cfg):
    """可选：光刻胶内扩散的高斯模糊（σ 单位 nm，自动换算成像素）。

    边界模式必须与前向模型一致：周期边界用 'wrap'（环绕），
    孤立边界用 'nearest'。用默认的 'reflect' 会破坏周期性
    （镜像边界会把视场边缘的像反射回来）。
    """
    if cfg.RESIST_BLUR_SIGMA_NM <= 0:
        return aerial
    sigma_px = cfg.RESIST_BLUR_SIGMA_NM / cfg.PIXEL_SIZE_NM
    mode = "wrap" if getattr(cfg, "BOUNDARY_MODE", "periodic") == "periodic" \
        else "nearest"
    return ndimage.gaussian_filter(aerial, sigma=sigma_px, mode=mode)


def forward(mask, cfg, precomp):
    """一次完整前向：空间像 → 胶模糊 → 阈值印刷轮廓（1=印刷成图形）。"""
    aerial = apply_resist_blur(aerial_image(mask, cfg, precomp), cfg)
    printed = (aerial < cfg.RESIST_THRESHOLD).astype(np.uint8)
    return aerial, printed


# ------------------------------------------------------------------
# 光学模型自检（跑 OPC 之前先全部 PASS）
# ------------------------------------------------------------------

def _temp_cfg(cfg, **overrides):
    """复制配置并覆盖部分参数，供自检构造临时光学条件。"""
    d = {k: v for k, v in vars(cfg).items() if k.isupper()}
    d.update(overrides)
    return types.SimpleNamespace(**d)


def self_check(cfg):
    """光学模型 5 项物理正确性检查，全部 PASS 才建议继续跑 OPC。

    1. 开域：无图形的掩模 → 归一化空间像处处 = 1（验证权重与归一化）
    2. 大图形：中心趋 0、远处趋 1（验证成像基本行为）
    3. 左右对称：不对称图形及其镜像的像严格互为镜像
                    （验证光源采样对称与 FFT 中心对齐，最能抓错）
    4. 周期线条：各线印刷宽度一致（验证平移不变性）
    5. 离焦：I(+z)=I(−z)，且离焦使边缘斜率下降（验证光瞳相位）
    """
    from mask_ops import polygons_to_pixels

    results = []

    def report(name, ok, detail):
        results.append((name, ok, detail))
        print("  [%s] %-6s %s" % ("PASS" if ok else "FAIL", name, detail))

    # ---- 检查 1：开域 ----
    N = cfg.GRID_N
    pre = precompute_source_and_pupil(cfg)
    I = aerial_image(np.zeros((N, N), np.uint8), cfg, pre)
    report("开域归一化", np.max(np.abs(I - 1.0)) < 1e-6,
           "max|I-1| = %.2e" % np.max(np.abs(I - 1.0)))

    # ---- 检查 2：大矩形 ----
    rect = [np.array([[-100.0, -100.0], [100.0, -100.0],
                      [100.0, 100.0], [-100.0, 100.0]])]
    mask = polygons_to_pixels(rect, N, cfg.PIXEL_SIZE_NM)
    I = aerial_image(mask, cfg, pre)
    c = N // 2
    # 门限宽松些：图形边缘振铃会使中心不完全为 0、近处过冲略大于 1
    report("大图形成像", I[c, c] < 0.1 and I[4, 4] > 0.95,
           "中心 I=%.3f (应≈0), 远处 I=%.3f (应≈1)" % (I[c, c], I[4, 4]))

    # ---- 检查 3：左右对称（不对称图形 + 镜像） ----
    asym = [np.array([[-80.0, -80.0], [20.0, -80.0], [20.0, 80.0],
                      [-80.0, 80.0]]),                          # 大矩形（不对称）
            np.array([[20.0, -20.0], [80.0, -20.0], [80.0, 20.0],
                      [20.0, 20.0]])]                           # 右侧小方块
    mask = polygons_to_pixels(asym, N, cfg.PIXEL_SIZE_NM)
    I1 = aerial_image(mask, cfg, pre)
    I2 = aerial_image(mask[:, ::-1], cfg, pre)
    err = np.max(np.abs(I1 - I2[:, ::-1]))
    report("左右对称性", err < 1e-9, "max|I - mirror(I)| = %.2e" % err)

    # ---- 检查 4：周期线条平移不变性（对称对印刷宽度一致） ----
    # 有限阵列存在邻近效应梯度：中间线最宽、最外线最窄是正确物理；
    # 平移不变性的正确判据是"左右对称的两条线宽度必须相等"。
    lines = [np.array([[-90 * (k - 2) - 22.5, -120.0],
                       [-90 * (k - 2) + 22.5, -120.0],
                       [-90 * (k - 2) + 22.5, 120.0],
                       [-90 * (k - 2) - 22.5, 120.0]]) for k in range(5)]
    mask = polygons_to_pixels(lines, N, cfg.PIXEL_SIZE_NM)
    I = aerial_image(mask, cfg, pre)
    row = (I < cfg.RESIST_THRESHOLD).astype(np.uint8)[c, :]
    runs = []                                   # 每段连续暗像素的长度（nm）
    k = 0
    while k < N:
        if row[k] == 1:
            j = k
            while j < N and row[j] == 1:
                j += 1
            runs.append((j - k) * cfg.PIXEL_SIZE_NM)
            k = j
        else:
            k += 1
    ok = (len(runs) == 5
          and all(abs(runs[i] - runs[4 - i]) < 0.5 for i in range(5)))
    report("周期线条一致性", ok,
           "各线印刷宽度 %s nm" % ["%.1f" % r for r in runs])

    # ---- 检查 5：离焦对称与斜率 ----
    cfg_z = _temp_cfg(cfg, DEFOCUS_NM=40.0)
    pre_z = precompute_source_and_pupil(cfg_z)
    cfg_zm = _temp_cfg(cfg, DEFOCUS_NM=-40.0)
    pre_zm = precompute_source_and_pupil(cfg_zm)
    rect2 = [np.array([[-120.0, -120.0], [120.0, -120.0],
                       [120.0, 120.0], [-120.0, 120.0]])]
    mask = polygons_to_pixels(rect2, N, cfg.PIXEL_SIZE_NM)
    I0 = aerial_image(mask, cfg, pre)
    Ip = aerial_image(mask, cfg_z, pre_z)
    Im = aerial_image(mask, cfg_zm, pre_zm)
    sym_ok = np.max(np.abs(Ip - Im)) < 1e-9
    # 边缘斜率：穿过中心行的水平梯度最大值（边缘处）
    grad0 = np.max(np.abs(np.gradient(I0[c, :])))
    gradz = np.max(np.abs(np.gradient(Ip[c, :])))
    slope_ok = gradz < grad0
    report("离焦物理", sym_ok and slope_ok,
           "I(+z)=I(−z) 误差 %.1e；斜率 %.4f → %.4f"
           % (np.max(np.abs(Ip - Im)), grad0, gradz))

    # ---- 检查 6：周期边界条件 ----
    # 判据：把掩模在视场内循环平移 (dy,dx)，空间像必须精确跟着平移。
    # 只有周期边界才满足这一点（孤立/零填充边界在视场边缘会不同）。
    if getattr(cfg, "BOUNDARY_MODE", "periodic") == "periodic":
        mask = polygons_to_pixels(
            [np.array([[-150.0, -100.0], [50.0, -100.0],
                       [50.0, 60.0], [-150.0, 60.0]]),
             np.array([[150.0, 100.0], [260.0, 100.0],
                       [260.0, 160.0], [150.0, 160.0]])],
            N, cfg.PIXEL_SIZE_NM, wrap=True)
        shift = (N // 4, -N // 5)      # 行、列方向的循环平移（像素）
        I_a = aerial_image(mask, cfg, pre)
        I_b = aerial_image(np.roll(mask, shift, axis=(0, 1)), cfg, pre)
        err = np.max(np.abs(I_a - np.roll(I_b, (-shift[0], -shift[1]),
                                         axis=(0, 1))))
        # 跨越视场边界的线条：上边缘的图形应环绕到下边缘，像连续无缝
        half_line = [np.array([[-40.0, N * cfg.PIXEL_SIZE_NM / 2 - 60.0],
                               [40.0, N * cfg.PIXEL_SIZE_NM / 2 - 60.0],
                               [40.0, N * cfg.PIXEL_SIZE_NM / 2 + 60.0],
                               [-40.0, N * cfg.PIXEL_SIZE_NM / 2 + 60.0]])]
        mask_wrap = polygons_to_pixels(half_line, N, cfg.PIXEL_SIZE_NM,
                                       wrap=True)
        I_w = aerial_image(mask_wrap, cfg, pre)
        row_top = I_w[2, :]          # 紧邻上边界
        row_bot = I_w[-3, :]         # 紧邻下边界（同一图形环绕后的部分）
        wrap_ok = np.max(np.abs(row_top - row_bot)) < 0.05
        report("周期边界条件", err < 1e-9 and wrap_ok,
               "循环平移不变性误差 %.1e；跨界图形上下边缘像差 %.3f"
               % (err, np.max(np.abs(row_top - row_bot))))

    print("自检结果: %d/%d PASS" % (sum(ok for _, ok, _ in results), len(results)))
    return all(ok for _, ok, _ in results)
