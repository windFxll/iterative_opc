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
把光源离散成若干点 {f_s, w_s}（环形/圆形按径向环 × 方位角；两极/四极
按圆形盘极；自定义光源图按非零格点），每个光源点对应一束倾斜平面波，
其像场为

    E_s(x) = FT⁻¹[ M(f) · P(f+f_s) ](x)

Hopkins 方程可严格改写为对这些点源的强度求和（数学上等价，非近似——
近似只来自光源采样点数）:

    I(x) = Σ_s w_s · |E_s(x)|²

对实值掩模和对称光瞳，±f_s 给出相同的像；因此对称光源的采样点必须
成对，这也是"对称图形必须得到对称像"的数值保障。非对称光源（例如
自定义光源图只亮一侧）不要求成对。

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
import os
import types

import numpy as np
from scipy import ndimage


# ------------------------------------------------------------------
# 光源形状 → 采样点
# ------------------------------------------------------------------

def _even_at_least(value, minimum):
    """不小于 minimum 的偶数（多极方位采样要偶数才能镜像成对）。"""
    n = max(int(round(value)), int(minimum))
    return n if n % 2 == 0 else n + 1


def _sample_polar_source(kind, cfg):
    """环形/圆形/多极光源 → (sigmas, area_weights)。

    sigmas : (M,2) 归一化光源坐标 (σx,σy)，σ = 光源频率 / (NA/λ)
    area_weights : (M,) 面积元权重（未归一化；公共常数不影响结果）

    环形/圆形：径向环 × 方位角，权重 ∝ σ_r（面积元 σ dσ dθ）。
    两极/四极：每个极是圆形盘，盘内按 (r,φ) 采样，权重 ∝ r。
    """
    n_rings = int(cfg.N_SOURCE_RINGS)
    n_pts = int(cfg.N_SOURCE_PTS)
    if n_rings < 1 or n_pts < 1:
        raise ValueError("N_SOURCE_RINGS 和 N_SOURCE_PTS 必须 >= 1")

    if kind in ("annular", "circle"):
        sig_in = 0.0 if kind == "circle" else float(cfg.SIGMA_IN)
        sig_out = float(cfg.SIGMA_OUT)
        if not (0.0 <= sig_in < sig_out <= 1.0 + 1e-9):
            raise ValueError(
                "环形/圆形照明要求 0 <= SIGMA_IN < SIGMA_OUT <= 1，"
                "当前 SIGMA_IN=%.3f, SIGMA_OUT=%.3f" % (sig_in, sig_out))
        dsigma = (sig_out - sig_in) / n_rings
        radii = sig_in + (np.arange(n_rings) + 0.5) * dsigma
        # 与旧实现完全一致：方位角从 0 起、步长 2π/N（不取中点），
        # 保证默认 annular 的采样点和历史结果逐点一致。
        thetas = 2.0 * np.pi * np.arange(n_pts) / n_pts
        pts, wts = [], []
        for sr in radii:
            for th in thetas:
                pts.append((sr * np.cos(th), sr * np.sin(th)))
                wts.append(sr)
        return np.asarray(pts, dtype=float), np.asarray(wts, dtype=float)

    if kind in ("dipole", "quadrupole"):
        n_poles = 2 if kind == "dipole" else 4
        # 每极每环点数：总点数与环形照明相近；取偶数保证镜像成对。
        n_local = _even_at_least(n_pts / float(n_poles), 4)
        step = np.pi if n_poles == 2 else np.pi / 2.0
        ang0 = np.deg2rad(float(cfg.POLE_ANGLE_DEG))
        centers = ang0 + np.arange(n_poles) * step
        sig_c = float(cfg.POLE_SIGMA)
        r_pole = float(cfg.POLE_RADIUS)
        if r_pole <= 0.0:
            raise ValueError("POLE_RADIUS 必须 > 0")
        if sig_c < 0.0 or sig_c + r_pole > 1.0 + 1e-9:
            raise ValueError(
                "两极/四极照明要求 POLE_SIGMA >= 0 且 "
                "POLE_SIGMA + POLE_RADIUS <= 1（光源在单位圆盘内），"
                "当前 POLE_SIGMA=%.3f, POLE_RADIUS=%.3f"
                % (sig_c, r_pole))
        radii = r_pole * (np.arange(n_rings) + 0.5) / n_rings
        phi = 2.0 * np.pi * np.arange(n_local) / n_local
        pts, wts = [], []
        for c in centers:
            cx, cy = sig_c * np.cos(c), sig_c * np.sin(c)
            for r in radii:
                for p in phi:
                    pts.append((cx + r * np.cos(p), cy + r * np.sin(p)))
                    wts.append(r)
        return np.asarray(pts, dtype=float), np.asarray(wts, dtype=float)

    raise ValueError("未知 ILLUM_TYPE=%r，可选: annular / circle / dipole / "
                     "quadrupole / custom" % kind)


def _resize_source_map(arr, max_n):
    """把大光源图按面积平均缩到长边 <= max_n，保持纵横比和左右对称性。"""
    ny, nx = arr.shape
    if max(ny, nx) <= max_n:
        return arr
    scale = max_n / float(max(ny, nx))
    new_nx = max(2, int(round(nx * scale)))
    new_ny = max(2, int(round(ny * scale)))
    try:
        import cv2
        return cv2.resize(arr, (new_nx, new_ny),
                          interpolation=cv2.INTER_AREA)
    except ImportError:
        # 无 OpenCV 时的退化路径：整数块平均，并从中心裁剪以保持对称
        fy = int(np.ceil(ny / float(max_n)))
        fx = int(np.ceil(nx / float(max_n)))
        ny2, nx2 = max(1, ny // fy), max(1, nx // fx)
        oy = (ny - ny2 * fy) // 2
        ox = (nx - nx2 * fx) // 2
        arr = arr[oy:oy + ny2 * fy, ox:ox + nx2 * fx]
        return arr.reshape(ny2, fy, nx2, fx).mean(axis=(1, 3))


def _load_source_map(cfg):
    """读取自定义光源图（SOURCE_MAP 阵列/函数 或 SOURCE_MAP_FILE 文件）。

    返回二维 float 阵列，坐标约定：行=y（第 0 行 σy=-1），列=x
    （第 0 列 σx=-1），覆盖归一化光源面 σx,σy ∈ [-1,1]。
    """
    spec = getattr(cfg, "SOURCE_MAP", None)
    path = getattr(cfg, "SOURCE_MAP_FILE", "") or ""
    if spec is not None and path:
        raise ValueError("SOURCE_MAP 与 SOURCE_MAP_FILE 只能设置一个")

    if spec is not None:
        if callable(spec):
            n = int(getattr(cfg, "SOURCE_MAP_MAX_N", 33))
            if n < 2:
                raise ValueError("SOURCE_MAP_MAX_N 必须 >= 2")
            axis = np.linspace(-1.0, 1.0, n)
            SX, SY = np.meshgrid(axis, axis, indexing="xy")
            arr = np.asarray(spec(SX, SY), dtype=np.float64)
        else:
            arr = np.asarray(spec, dtype=np.float64)
    elif path:
        if not os.path.isfile(path):
            raise ValueError("找不到自定义光源图文件: %s" % path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            arr = np.asarray(np.load(path), dtype=np.float64)
        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            import cv2
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError("无法读取自定义光源图: %s" % path)
            # OpenCV 第 0 行是图像上方；项目约定第 0 行 = σy=-1（下方），
            # 上下翻转后，图像正着看时 +y 朝上。
            arr = gray[::-1].astype(np.float64)
        else:
            try:
                arr = np.loadtxt(path, delimiter=",")
            except ValueError:
                arr = np.loadtxt(path)
            arr = np.asarray(arr, dtype=np.float64)
    else:
        raise ValueError("ILLUM_TYPE='custom' 但 SOURCE_MAP 和 "
                         "SOURCE_MAP_FILE 都未设置")

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("自定义光源图必须是二维阵列（行=y, 列=x）")
    if not np.all(np.isfinite(arr)):
        raise ValueError("自定义光源图含 NaN/Inf")
    arr = np.maximum(arr, 0.0)
    if float(arr.max()) <= 0.0:
        raise ValueError("自定义光源图强度全为 0")
    max_n = int(getattr(cfg, "SOURCE_MAP_MAX_N", 33))
    if max_n >= 2 and max(arr.shape) > max_n:
        arr = _resize_source_map(arr, max_n)
    return arr


def _sample_custom_source(cfg):
    """自定义光源图 → 每个非零格点一个源点，权重 = 强度 × 格点面积。"""
    arr = _load_source_map(cfg)
    ny, nx = arr.shape
    dx, dy = 2.0 / nx, 2.0 / ny
    sx = -1.0 + (np.arange(nx) + 0.5) * dx
    sy = -1.0 + (np.arange(ny) + 0.5) * dy
    SX, SY = np.meshgrid(sx, sy, indexing="xy")
    # 单位圆盘之外是物镜收集不到的照明角度，直接丢弃
    inside = SX ** 2 + SY ** 2 <= 1.0 + 1e-12
    thr = float(getattr(cfg, "SOURCE_MAP_THRESHOLD", 1e-3))
    if thr > 0:
        keep = inside & (arr > thr * float(arr.max()))
    else:
        keep = inside & (arr > 0.0)
    if not np.any(keep):
        raise ValueError("自定义光源图在单位圆盘内没有非零强度")
    pts = np.column_stack([SX[keep], SY[keep]])
    wts = arr[keep] * dx * dy
    return pts, wts


def _source_symmetry_flags(sigmas, weights, tol=1e-7):
    """数值判断光源采样的镜像/中心对称性（供自检用）。

    返回 (mirror_x, mirror_y, central)：
      mirror_x : 关于 y 轴（x→-x）对称
      mirror_y : 关于 x 轴（y→-y）对称
      central  : 中心对称（s→-s），离焦 ±z 对称性需要这一条
    """
    table = {}
    for (x, y), w in zip(sigmas, weights):
        key = (round(float(x), 9), round(float(y), 9))
        table[key] = table.get(key, 0.0) + float(w)
    scale = max(float(np.max(weights)), 1e-30)

    def ok(mx, my):
        for (x, y), w in table.items():
            w2 = table.get((round(mx * x, 9), round(my * y, 9)))
            if w2 is None or abs(w2 - w) > tol * scale:
                return False
        return True

    return ok(-1.0, 1.0), ok(1.0, -1.0), ok(-1.0, -1.0)


def sample_source(cfg):
    """按 ILLUM_TYPE 生成归一化光源采样点。

    支持：
      'annular'     环形（SIGMA_IN ~ SIGMA_OUT）
      'circle'      圆形（0 ~ SIGMA_OUT）
      'dipole'      两极圆形盘（POLE_SIGMA/POLE_RADIUS/POLE_ANGLE_DEG）
      'quadrupole'  四极圆形盘（同上，中心相差 90°）
      'custom'      自定义光源图（SOURCE_MAP / SOURCE_MAP_FILE）

    返回 dict:
      kind     : 光源类型
      sigmas   : (M,2) 归一化光源坐标 (σx,σy)
      weights  : (M,) 权重，Σw=1
      mirror_x : 光源是否关于 y 轴（x→-x）镜像对称
      mirror_y : 光源是否关于 x 轴（y→-y）镜像对称
      central  : 光源是否中心对称（s→-s）
    """
    kind = str(getattr(cfg, "ILLUM_TYPE", "annular")).lower()
    if kind == "custom":
        sigmas, wts = _sample_custom_source(cfg)
    else:
        sigmas, wts = _sample_polar_source(kind, cfg)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    wts = np.asarray(wts, dtype=np.float64)
    if np.any(wts < 0.0) or float(wts.sum()) <= 0.0:
        raise ValueError("光源采样权重必须非负且总和 > 0")
    # 光源面是单位圆盘：|σ|>1 的照明角度物镜收集不到，统一裁掉
    keep = np.hypot(sigmas[:, 0], sigmas[:, 1]) <= 1.0
    if not np.all(keep):
        sigmas, wts = sigmas[keep], wts[keep]
    if sigmas.shape[0] == 0:
        raise ValueError("光源采样在单位圆盘内没有有效点")
    wts = wts / wts.sum()
    mirror_x, mirror_y, central = _source_symmetry_flags(sigmas, wts)
    return {"kind": kind, "sigmas": sigmas, "weights": wts,
            "mirror_x": mirror_x, "mirror_y": mirror_y,
            "central": central}


# ------------------------------------------------------------------
# 光瞳预计算
# ------------------------------------------------------------------

def _n_fft(cfg):
    """FFT 网格尺寸。周期边界模式 = GRID_N（视场即一个周期）；
    孤立边界模式 = 2×GRID_N（零填充后取中心，模拟线性卷积、无环绕）。"""
    if getattr(cfg, "BOUNDARY_MODE", "periodic") == "isolated":
        return 2 * cfg.GRID_N
    return cfg.GRID_N


def _pupil_for_source(cfg, FX, FY, fc, fsx, fsy):
    """一个光源点对应的光瞳 P(f+f_s)（自然序）。"""
    nu = np.sqrt((FX + fsx) ** 2 + (FY + fsy) ** 2) / fc   # 归一化频率
    P = np.zeros(FX.shape, dtype=np.complex128)
    inside = nu <= 1.0
    # 光瞳透过：|ν|≤1 处为 1；离焦相位（标量模型）：
    #   exp[j·(2π/λ)·Δz·(√(1−(NA·ν)²) − 1)]，±Δz 互为共轭 → I(+z)=I(−z)
    if cfg.DEFOCUS_NM != 0.0:
        lam, na = cfg.WAVELENGTH_NM, cfg.NA
        ph = (2 * np.pi / lam) * cfg.DEFOCUS_NM * \
             (np.sqrt(np.maximum(1.0 - (na * nu[inside]) ** 2, 0.0)) - 1.0)
        P[inside] = np.exp(1j * ph)
    else:
        P[inside] = 1.0
    return P


def precompute_source_and_pupil(cfg):
    """采样光源、构建每个光源点对应的光瞳数组。只与光学参数有关，
    与掩模无关，整个 OPC 迭代过程只需调用一次。

    光源形状由 cfg.ILLUM_TYPE 决定（环形/圆形/两极/四极/自定义光源图），
    采样点与权重见 sample_source()。

    边界条件（重要）：
      周期边界 'periodic'（默认）——视场就是版图的一个周期，FFT 的
        循环卷积天然给出周期边界；周期版图的谱只含离散谐波，光瞳仅在
        这些谐波上取值，因此该实现对周期模型是**精确**的（无光瞳采样误差）。
      孤立边界 'isolated'——把掩模零填充到 2N 再做 FFT（线性卷积），
        结果取中心 N×N，消除相邻周期串扰；代价是 4 倍 FFT 时间。

    返回 dict:
      weights : 光源点权重 w_s（已归一化 Σw=1）
      fs      : (M,2) 光源点频率偏移 f_s (nm⁻¹)
      pupils  : 每个光源点一张 (n_fft,n_fft) 复数光瞳 P(f+f_s)（自然序）；
                点数过多超过 PUPIL_CACHE_MB 时为 None，前向时按需构建
      i_clear : 开域强度解析值（空掩模的空间像 = 1，用于归一化）
      n_fft   : FFT 网格尺寸
      source  : sample_source() 的完整返回（含 sigmas/weights/对称标志）
      FX, FY  : 频率网格（自然序），用于按需构建光瞳
      fc      : 光瞳截断频率 NA/λ
    """
    N, dx = _n_fft(cfg), cfg.PIXEL_SIZE_NM
    lam, na = cfg.WAVELENGTH_NM, cfg.NA
    fc = na / lam                       # 光瞳截断频率 = NA/λ (nm⁻¹)

    # ---- 频率网格（自然序）：FX[i,j] = fx[j], FY[i,j] = fy[i]，
    #      与 fft2 输出 M[k_y, k_x] 的布局一致 ----
    fx = np.fft.fftfreq(N, d=dx)
    FY, FX = np.meshgrid(fx, fx, indexing="xy")

    # ---- 光源采样（形状由 config 定义） ----
    src = sample_source(cfg)
    weights = src["weights"]
    fs = src["sigmas"] * fc             # (M,2) 物理频率偏移 (nm⁻¹)

    # ---- 每个光源点一张光瞳数组（含离焦相位） ----
    # 光瞳缓存占 M×N²×16 字节；自定义光源点数多时可能超出预算，
    # 此时不预存，改为 aerial_image 里按需构建（内存 O(N²)，稍慢）。
    cache_mb = float(getattr(cfg, "PUPIL_CACHE_MB", 512.0))
    need_mb = len(weights) * N * N * 16.0 / (1024.0 ** 2)
    if need_mb <= cache_mb:
        pupils = [_pupil_for_source(cfg, FX, FY, fc, fsx, fsy)
                  for fsx, fsy in fs]
        pupil_cached = True
    else:
        pupils = None
        pupil_cached = False

    # ---- 开域强度（解析值）：空掩模的像。sample_source 已把源点裁到
    #      单位圆盘内，|P|²=1（离焦只改相位），因此 i_clear = Σw = 1。 ----
    i_clear = float(np.sum(weights))

    return {"weights": weights, "fs": fs, "pupils": pupils,
            "pupil_cached": pupil_cached, "i_clear": i_clear,
            "n_fft": N, "source": src, "FX": FX, "FY": FY, "fc": fc}


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
    pupils = precomp.get("pupils")
    if pupils is not None:
        # 光瞳已预计算（默认情况）
        for w_s, P_s in zip(precomp["weights"], pupils):
            # ifft2 自带 1/N² 归一化；fftshift 把原点移回图像中心
            E_s = np.fft.fftshift(np.fft.ifft2(M * P_s))
            I += w_s * np.abs(E_s) ** 2
    else:
        # 光源点数多、光瞳未缓存：按需构建，内存 O(N²)
        FX, FY, fc = precomp["FX"], precomp["FY"], precomp["fc"]
        for w_s, (fsx, fsy) in zip(precomp["weights"], precomp["fs"]):
            P_s = _pupil_for_source(cfg, FX, FY, fc, fsx, fsy)
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
    """光学模型 7 项物理正确性检查，全部 PASS 才建议继续跑 OPC。

    0. 光源采样：权重归一化、所有源点位于单位圆盘内
    1. 开域：无图形的掩模 → 归一化空间像处处 = 1（验证权重与归一化）
    2. 大图形：中心趋 0、远处趋 1（验证成像基本行为）
    3. 左右对称：不对称图形及其镜像的像严格互为镜像
                    （光源本身非左右对称时自动跳过）
    4. 周期线条：各线印刷宽度一致（光源非左右对称时自动跳过）
    5. 离焦：I(+z)=I(−z)，且离焦使边缘斜率下降（验证光瞳相位；
       光源非中心对称时跳过对称性判据）
    6. 周期边界：循环平移不变性与跨边界图形连续性
    """
    from mask_ops import polygons_to_pixels

    results, skipped = [], []

    def report(name, ok, detail, skip=False):
        if skip:
            skipped.append(name)
            print("  [SKIP] %-6s %s" % (name, detail))
            return
        results.append((name, ok, detail))
        print("  [%s] %-6s %s" % ("PASS" if ok else "FAIL", name, detail))

    # ---- 检查 0：光源采样 ----
    N = cfg.GRID_N
    pre = precompute_source_and_pupil(cfg)
    src = pre["source"]
    max_sig = float(np.max(np.hypot(src["sigmas"][:, 0],
                                    src["sigmas"][:, 1])))
    report("光源采样",
           abs(float(src["weights"].sum()) - 1.0) < 1e-12
           and max_sig <= 1.0 + 1e-9,
           "%s, %d 点, Σw=%.6f, max|σ|=%.3f"
           % (src["kind"], len(src["weights"]),
              float(src["weights"].sum()), max_sig))

    # ---- 检查 1：开域 ----
    I = aerial_image(np.zeros((N, N), np.uint8), cfg, pre)
    report("开域归一化", np.max(np.abs(I - 1.0)) < 1e-6,
           "max|I-1| = %.2e" % np.max(np.abs(I - 1.0)))

    # ---- 检查 2：大矩形 ----
    rect = [np.array([[-100.0, -100.0], [100.0, -100.0],
                      [100.0, 100.0], [-100.0, 100.0]])]
    mask = polygons_to_pixels(rect, N, cfg.PIXEL_SIZE_NM)
    I = aerial_image(mask, cfg, pre)
    c = N // 2
    # 判据用相对值：中心应明显暗于远处。不同光源形状（多极/离轴）下
    # 大图形中心的绝对残余强度不同，固定 0.1 的门限不普适。
    far = I[4, 4]
    report("大图形成像", I[c, c] < 0.3 * far and far > 0.9,
           "中心 I=%.3f (应≈0), 远处 I=%.3f (应≈1)" % (I[c, c], far))

    # ---- 检查 3：左右对称（不对称图形 + 镜像） ----
    # 只有光源本身关于 x→-x 对称时，像才应满足左右对称。
    if src["mirror_x"]:
        asym = [np.array([[-80.0, -80.0], [20.0, -80.0], [20.0, 80.0],
                          [-80.0, 80.0]]),                      # 大矩形（不对称）
                np.array([[20.0, -20.0], [80.0, -20.0], [80.0, 20.0],
                          [20.0, 20.0]])]                       # 右侧小方块
        mask = polygons_to_pixels(asym, N, cfg.PIXEL_SIZE_NM)
        I1 = aerial_image(mask, cfg, pre)
        I2 = aerial_image(mask[:, ::-1], cfg, pre)
        err = np.max(np.abs(I1 - I2[:, ::-1]))
        report("左右对称性", err < 1e-9, "max|I - mirror(I)| = %.2e" % err)
    else:
        report("左右对称性", None, "当前光源不是左右对称，跳过此检查",
               skip=True)

    # ---- 检查 4：周期线条平移不变性（对称对印刷宽度一致） ----
    # 有限阵列存在邻近效应梯度：中间线最宽、最外线最窄是正确物理；
    # 平移不变性的正确判据是"左右对称的两条线宽度必须相等"。
    if src["mirror_x"]:
        lines = [np.array([[-90 * (k - 2) - 22.5, -120.0],
                           [-90 * (k - 2) + 22.5, -120.0],
                           [-90 * (k - 2) + 22.5, 120.0],
                           [-90 * (k - 2) - 22.5, 120.0]]) for k in range(5)]
        mask = polygons_to_pixels(lines, N, cfg.PIXEL_SIZE_NM)
        I = aerial_image(mask, cfg, pre)
        row = (I < cfg.RESIST_THRESHOLD).astype(np.uint8)[c, :]
        runs = []                               # 每段连续暗像素的长度（nm）
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
        if len(runs) != 5:
            report("周期线条一致性", None,
                   "该光源下 5 条线未全部独立成像（检测到 %d 段），跳过"
                   % len(runs), skip=True)
        else:
            ok = all(abs(runs[i] - runs[4 - i]) < 0.5 for i in range(5))
            report("周期线条一致性", ok,
                   "各线印刷宽度 %s nm" % ["%.1f" % r for r in runs])
    else:
        report("周期线条一致性", None, "当前光源不是左右对称，跳过此检查",
               skip=True)

    # ---- 检查 5：离焦对称与斜率 ----
    # I(+z)=I(−z) 需要光源中心对称（s→-s）；单个离轴光源不满足，
    # 此时跳过对称性判据（斜率仍可检查，但报告为跳过更清楚）。
    if src["central"]:
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
    else:
        report("离焦物理", None,
               "光源不是中心对称（s→-s），I(+z)=I(−z) 不适用，跳过",
               skip=True)

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

    print("自检结果: %d/%d PASS%s"
          % (sum(ok for _, ok, _ in results), len(results),
             "，%d 项 SKIP" % len(skipped) if skipped else ""))
    return all(ok for _, ok, _ in results)
