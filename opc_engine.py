# -*- coding: utf-8 -*-
"""
opc_engine.py —— 迭代法 OPC 引擎。

工作方式（经典边缘基 OPC，Cobb & Zakhor 思路）：
  目标图形 → 光栅化成初始掩模 → Hopkins 前向仿真得空间像 →
  阈值得印刷轮廓 → 沿每条掩模边法线采样求 EPE（边缘放置误差）→
  按反馈公式移动边缘 / 调整角落切口 → 重新光栅化 → 迭代。

数据模型（关键设计）
--------------------
- 掩模的真实状态是**浮点多边形**（nm），像素只是每轮迭代光栅化的派生视图。
  这样亚像素的边缘位移可以累积，不被 2nm 像素网格量化丢失。
- 每条边是一个 Segment：支撑直线 n·x = d（n 为单位外法向），移动即改 d；
  d_ref 是该边对应的**目标**直线偏移，EPE 以此为基准计算。
- 角落不单独写反馈代码：把角点替换成一条 45° 切口段（serif/hammerhead
  种子），切口段与普通边走完全相同的 EPE 反馈，自动沿角平分线方向生长。
  数学推导：设印刷轮廓沿角平分线相对目标角点偏移 t（圆化量），切口边长 s，
  则切口段 EPE = s/√2 + t，反馈 Δ=−η·EPE 的不动点 s* = √2·|t| > 0，
  即 serif 自动长到恰好补偿角点圆化——凸角/凹角公式完全一致。

符号约定（全文唯一）
--------------------
- 外法向 n 指向图形外部（背景侧）；多边形顶点逆时针（CCW，内部在行进方向左侧）。
- EPE > 0 = 印刷图形超出目标边界 → 反馈 Δ = −η·EPE 使边内收。
- 单位一律 nm。
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from mask_ops import (polygons_to_pixels, extract_polygon, classify_corner,
                      cross2,
                      cleanup_mask, check_design_rules)
from optics import aerial_image, apply_resist_blur, precompute_source_and_pupil


# ------------------------------------------------------------------
# 数据模型
# ------------------------------------------------------------------

@dataclass
class Segment:
    """一条可移动的掩模边（支撑直线 n·x = d）。"""
    n: np.ndarray              # 单位外法向 (2,)，指向图形外部
    d: float                   # 直线偏移 n·x = d（移动即改 d）
    p1: np.ndarray             # 当前起点 (2,) nm
    p2: np.ndarray             # 当前终点 (2,) nm
    d_ref: float               # 目标参考直线 n·x = d_ref（EPE 基准）
    is_cut: bool = False       # 是否为角落 45° 切口段（serif/hammerhead）
    corner_pt: np.ndarray = field(default=None, repr=False)  # 切口对应的目标角点
    s_sign: int = 1            # 切口直线偏移符号：凸角 +1（切口在角点外侧），
                               # 凹角 −1（切口在角点内侧，即 d = d_ref − s/√2）
    is_jog: bool = False       # 是否为连接断阶的人造小段（不参与反馈）
    prev_move: float = 0.0     # 上一轮位移（一阶惯性阻尼记忆）

    def length(self):
        return float(np.linalg.norm(self.p2 - self.p1))


def _all_segs(polys):
    """把所有多边形的段按固定顺序摊平（EPE/移动共用同一顺序）。"""
    out = []
    for poly in polys:
        out.extend(poly)
    return out


def _copy_polys(polys):
    """深拷贝多边形段状态（用于记录历史最佳）。"""
    return [[Segment(n=s.n.copy(), d=s.d, p1=s.p1.copy(), p2=s.p2.copy(),
                     d_ref=s.d_ref, is_cut=s.is_cut,
                     corner_pt=(s.corner_pt.copy() if s.corner_pt is not None
                                else None),
                     s_sign=s.s_sign, is_jog=s.is_jog, prev_move=s.prev_move)
             for s in poly] for poly in polys]


# ------------------------------------------------------------------
# 初始化：目标多边形 → 分段 + 角落切口
# ------------------------------------------------------------------

def _line_intersection(n1, d1, n2, d2):
    """求两条直线 n1·x=d1 与 n2·x=d2 的交点（解 2×2 线性方程组）。"""
    A = np.stack([n1, n2])
    b = np.array([d1, d2])
    return np.linalg.solve(A, b)


def _split_edge(p1, p2, cfg):
    """把一条边（两个顶点）拆成若干 Segment（每段 ≤16nm），返回段列表。

    拆分目的：一条长边各处的印刷偏差不同（线端回缩 vs 线中鼓出），
    需要局部自由度，所以每条长边分成若干可独立移动的小段。
    """
    L = float(np.linalg.norm(p2 - p1))
    dvec = (p2 - p1) / L
    n = np.array([dvec[1], -dvec[0]])       # 外法向（CCW 行进方向右侧）
    d = float(n @ p1)

    n_parts = max(1, int(np.ceil(L / cfg.MAX_EDGE_SEGMENT_NM)))
    segs = []
    for k in range(n_parts):
        t0, t1 = L * k / n_parts, L * (k + 1) / n_parts
        segs.append(Segment(n=n, d=d, p1=p1 + t0 * dvec, p2=p1 + t1 * dvec,
                            d_ref=d))
    return segs


def _make_cut_pair(seg_in, seg_out, v_corner, corner_kind, cfg):
    """在角点构造经典 OPC 结构：凸角外扩**方块 serif**、凹角**方形缺口
    hammerhead**（不是 45° 尖角倒角）。

    几何：设进入角点的边方向 û_a、离开角点的边方向 û_b（CCW 行进），
      V1 = V + s·û_a   （进边缩短端）
      V2 = V1 + s·û_b  （方块外角）
      V3 = V + s·û_b   （出边起点）
    凸角：沿两条邻边连通地外扩一个 s×s 方块（serif）；
    凹角：挖去一个 s×s 方块（hammerhead）。
    会就地缩短 seg_in（p2=V1）与 seg_out（p1=V3），返回两个切口段
    (cut_a: V1→V2, cut_b: V2→V3)，两者共享 corner_pt 对象以便配对。
    """
    s = cfg.SERIF_INIT_NM
    u_a = (v_corner - seg_in.p1) / np.linalg.norm(v_corner - seg_in.p1)
    u_b = (seg_out.p2 - v_corner) / np.linalg.norm(seg_out.p2 - v_corner)
    v1 = v_corner + s * u_a
    v2 = v1 + s * u_b
    v3 = v_corner + s * u_b
    seg_in.p2 = v1
    seg_out.p1 = v3

    # 符号：凸角（方块在角点外侧）+1；凹角（方块挖进图形内部）−1。
    # 两个切口段的 s↔d 关系一致：d = d_ref + s_sign·s（无 √2 因子）
    s_sign = 1 if corner_kind == "convex" else -1
    corner_ref = v_corner.copy()          # 共享对象：配对识别用
    d1 = v2 - v1
    d1 = d1 / np.linalg.norm(d1)
    n1 = np.array([d1[1], -d1[0]])
    cut_a = Segment(n=n1, d=float(n1 @ v1), p1=v1, p2=v2,
                    d_ref=float(n1 @ v_corner), is_cut=True,
                    corner_pt=corner_ref, s_sign=s_sign)
    d2 = v3 - v2
    d2 = d2 / np.linalg.norm(d2)
    n2 = np.array([d2[1], -d2[0]])
    cut_b = Segment(n=n2, d=float(n2 @ v2), p1=v2, p2=v3,
                    d_ref=float(n2 @ v_corner), is_cut=True,
                    corner_pt=corner_ref, s_sign=s_sign)
    return [cut_a, cut_b]


def _build_polygon(verts, cfg, corner_kinds):
    """从一个多边形顶点列表构建 Segment 列表（分段 + 角落切口）。

    做法：先逐边拆分成段组，再在每个角点处把切口段插到两条邻边之间
    （链式排列：边0组, 角1切口, 边1组, 角2切口, ..., 角0切口）。
    """
    n = len(verts)
    groups = [_split_edge(verts[k], verts[(k + 1) % n], cfg) for k in range(n)]

    chain = []          # [边0组, 切口对1, 边1组, 切口对2, ...]
    for k in range(n):
        chain.append(groups[k])
        kind = corner_kinds[(k + 1) % n]
        if kind is not None:
            seg_in = groups[k][-1]              # 进入角点(k+1)的段
            seg_out = groups[(k + 1) % n][0]    # 离开角点(k+1)的段
            chain.append(_make_cut_pair(seg_in, seg_out,
                                        verts[(k + 1) % n], kind, cfg))
    segs = []
    for item in chain:
        segs.extend(item if isinstance(item, list) else [item])
    return segs


def _detect_corner_kinds(verts, all_polys, min_edge_nm=0.0):
    """逐顶点判断角点类型（凸/凹/非角点）。用几何象限法，与环绕方向无关。

    min_edge_nm：角点两侧邻边都必须不短于该值才判定为角点——用于过滤
    像素提取版图上的光栅化台阶角（2nm 级锯齿角，不是真角点）。
    """
    kinds = []
    n = len(verts)
    for k in range(n):
        e1 = np.linalg.norm(verts[k] - verts[k - 1])
        e2 = np.linalg.norm(verts[(k + 1) % n] - verts[k])
        if min(e1, e2) < min_edge_nm:
            kinds.append(None)      # 邻边太短 → 台阶角，不是真角点
            continue
        kinds.append(classify_corner(verts[k], verts[k - 1], verts[(k + 1) % n],
                                     all_polys))
    return kinds


def init_polygons(target_polys, cfg):
    """从目标多边形初始化 OPC 段状态（含 serif 种子切口）。

    target_polys: List[np.ndarray]，目标图形顶点（nm，几何真值）
    """
    polys = []
    for verts in target_polys:
        verts = np.asarray(verts, dtype=float)
        kinds = _detect_corner_kinds(verts, target_polys, cfg.MIN_EDGE_SEGMENT_NM)
        polys.append(_build_polygon(verts, cfg, kinds))
    return polys


def init_polygons_from_pixels(mask, target_polys, cfg):
    """像素清理改动掩模后：从像素重提取多边形，并与目标几何重新配对。

    仅在清理触发时调用（罕见）。d_ref 通过"方向匹配 + 距离最近"找回
    目标参考直线；切口只在贴近目标角点的顶点处重建（避免给光栅化
    台阶的每个角都加 serif）。
    """
    # 目标边直线集合（外法向 + 偏移），用于配对
    ref_lines = []
    for verts in target_polys:
        for k in range(len(verts)):
            p1, p2 = verts[k], verts[(k + 1) % len(verts)]
            dvec = (p2 - p1) / np.linalg.norm(p2 - p1)
            n = np.array([dvec[1], -dvec[0]])
            ref_lines.append((n, float(n @ p1)))
    ref_corners = [v.copy() for verts in target_polys for v in verts]

    polys = []
    for verts in extract_polygon(mask, cfg.GRID_N, cfg.PIXEL_SIZE_NM):
        kinds = []
        for v in verts:
            near_corner = any(np.linalg.norm(v - c) < 2.0 for c in ref_corners)
            kinds.append(_corner_kind_at(verts, v, cfg.MIN_EDGE_SEGMENT_NM)
                         if near_corner else None)
        segs = _build_polygon(verts, cfg, kinds)
        # 找回 d_ref：方向一致且距离最近的目标边直线
        for seg in segs:
            if seg.is_cut or seg.is_jog:
                continue
            best_d, best = None, 1e9
            for n_t, d_t in ref_lines:
                if abs(np.dot(n_t, seg.n)) < np.cos(np.deg2rad(1.0)):
                    continue
                if abs(d_t - seg.d) < best:
                    best, best_d = abs(d_t - seg.d), d_t
            seg.d_ref = best_d if (best_d is not None and best < 20.0) else seg.d
        polys.append(segs)
    return polys


def _corner_kind_at(verts, v, min_edge_nm=0.0):
    """单个顶点角点判断的封装（用于像素重提取场景，含短边过滤）。"""
    n = len(verts)
    for k in range(n):
        if np.allclose(verts[k], v, atol=1e-6):
            if (min(np.linalg.norm(v - verts[k - 1]),
                    np.linalg.norm(verts[(k + 1) % n] - v)) < min_edge_nm):
                return None     # 邻边太短 → 台阶角
            return classify_corner(v, verts[k - 1], verts[(k + 1) % n],
                                   [verts])
    return None


# ------------------------------------------------------------------
# EPE 计算：法线采样 + 插值找阈值交点
# ------------------------------------------------------------------

def _find_crossing(aerial, p, n, cfg):
    """从掩模边上的点 p 沿外法向 n 扫描，找离 p 最近的阈值交点。

    返回 t_cross（nm，交点相对 p 的带符号距离，沿 n 为正）。
    找不到交点时返回 ±EPE_SEARCH_NM（封顶防反馈爆炸）。
    """
    N, dx = cfg.GRID_N, cfg.PIXEL_SIZE_NM
    search = cfg.EPE_SEARCH_NM
    ts = np.arange(-search, search + 0.5 * dx, 0.5 * dx)
    pts = p + ts[:, None] * n
    # map_coordinates 的坐标顺序是 (行=y, 列=x)，与物理 (x,y) 相反
    coords = np.stack([(pts[:, 1] / dx + N / 2 - 0.5),   # 行 i
                       (pts[:, 0] / dx + N / 2 - 0.5)], axis=0)  # 列 j
    I = ndimage.map_coordinates(aerial, coords, order=1, mode="nearest")
    s = I - cfg.RESIST_THRESHOLD
    changes = np.where(np.diff(np.sign(s)) != 0)[0]
    if len(changes) == 0:
        # 整条扫描线都在印刷区（暗）内 → 图形印大了 → 正 EPE，反之负
        return search if s[len(s) // 2] < 0 else -search
    # 取离 p 最近的交点，线性插值细分到亚像素
    k = changes[np.argmin(np.abs(ts[changes]))]
    f = -s[k] / (s[k + 1] - s[k])
    return float(ts[k] + f * (ts[k + 1] - ts[k]))


def _corner_epe(poly, i, aerial, cfg):
    """沿角平分线测量角点 EPE（方形 serif/hammerhead 的正确度量）。

    方形 serif 的两个切口面**平行于相邻边**，沿其法线扫描永远扫不到
    印刷轮廓（切口面整个落在轮廓外侧），必须改为：从目标角点沿角平分线
    扫描找印刷轮廓交点 t。
      t < 0：印刷轮廓在角点内侧（圆化）→ 凸角需要更大的 serif；
      t > 0：印刷轮廓鼓出角点/凹角缺口鼓包 → 凹角需要更深的 hammerhead。
    返回 t（带符号距离，nm），feedback Δ=−η·EPE 与重建的
    s += s_sign·Δ 组合后：
      凸角（s_sign=+1）：Δs = −η·t，t<0（圆化）时 serif 增大 ✓
      凹角（s_sign=−1）：Δs = +η·t，t>0（缺口鼓包）时 hammerhead 加深 ✓
    """
    n_seg = len(poly)
    seg = poly[i]
    corner = seg.corner_pt
    # 向两侧找真正的"边"（跳过断阶与切口），用它们的外法向求角平分线
    k = (i - 1) % n_seg
    while (poly[k].is_jog or poly[k].is_cut) and k != i:
        k = (k - 1) % n_seg
    n_in = poly[k].n
    k = (i + 1) % n_seg
    while (poly[k].is_jog or poly[k].is_cut) and k != i:
        k = (k + 1) % n_seg
    n_out = poly[k].n
    bis = n_in + n_out
    nb = np.linalg.norm(bis)
    if nb < 1e-9:
        return 0.0
    bis = bis / nb
    return float(_find_crossing(aerial, corner, bis, cfg))


def compute_epe(polys, aerial, cfg):
    """计算所有段的 EPE（段级中位数 + 全体采样点列表）。

    边缘碎片：沿外法线采样找阈值交点（段级中位数）。
    切口段（serif/hammerhead 的方形边）：沿角平分线从角点测量，
    同一角点的两段切口共用同一个角点 EPE（见 _corner_epe）。

    返回 (seg_epes, epe_all)：
      seg_epes : 与 _all_segs(polys) 同顺序的段级 EPE（nm）
      epe_all  : 全体采样点 EPE（用于统计 rms/max）
    """
    seg_epes, epe_all = [], []
    for poly in polys:
        n_seg = len(poly)
        for i, seg in enumerate(poly):
            if seg.is_jog:
                seg_epes.append(0.0)      # 人造连接段不参与反馈
                continue
            if seg.is_cut:
                e = _corner_epe(poly, i, aerial, cfg)
                seg_epes.append(e)
                epe_all.append(e)
                continue
            L = seg.length()
            if L < 1e-6:
                seg_epes.append(0.0)      # 退化段（零长度）不采样，防除零
                continue
            # 采样点：均匀分布，两端避让角点（避让量不超过半段长）
            ex = min(cfg.CORNER_EXCLUSION_NM, L / 4)
            K = max(1, int(round(L / cfg.PIXEL_SIZE_NM)))
            ts = np.linspace(ex, L - ex, K) if K > 1 else np.array([L / 2])
            dvec = (seg.p2 - seg.p1) / L
            epe_k = []
            for t in ts:
                p = seg.p1 + t * dvec
                t_cross = _find_crossing(aerial, p, seg.n, cfg)
                # EPE = 印刷轮廓位置在法向上的投影 − 目标参考线
                epe_k.append(float((p + t_cross * seg.n) @ seg.n - seg.d_ref))
            seg_epes.append(float(np.median(epe_k)))
            epe_all.extend(epe_k)
    return seg_epes, epe_all


# ------------------------------------------------------------------
# 反馈与移动
# ------------------------------------------------------------------

def compute_moves(seg_epes, polys, cfg, it):
    """反馈公式：Δ = −η·EPE → 一阶惯性阻尼 → 单步限幅。

    限幅 = min(MAX_EDGE_STEP_NM, |EPE|)：当 |EPE| 小于单步限幅时，
    一步大位移会直接越过零点（印刷轮廓跟随增益 < 1，EPE 符号翻转），
    与阻尼记忆叠加成极限环振荡。限制 |Δ| ≤ |EPE| 可保证增益 ≤ 1 时
    每次迭代都不翻号（新 EPE = EPE − gain·Δ，|Δ|≤|EPE| 则符号不变）。

    每段位移存入 seg.prev_move（同时作为下一轮阻尼的记忆），返回位移列表。
    """
    eta = cfg.ETA_EDGE
    if cfg.ANNEAL:
        eta = eta / (1.0 + it / 10.0)     # 增益退火：迭代后期更稳
    moves = []
    for seg, epe in zip(_all_segs(polys), seg_epes):
        if seg.is_jog:
            seg.prev_move = 0.0
            moves.append(0.0)
            continue
        raw = -eta * epe
        cmd = cfg.DAMPING * seg.prev_move + (1.0 - cfg.DAMPING) * raw
        # 限幅 = min(MAX_EDGE_STEP_NM, 2·|EPE|)：当 |EPE| 小于单步限幅时，
        # 一步大位移会直接越过零点（印刷轮廓跟随增益 < 1，EPE 符号翻转），
        # 与阻尼记忆叠加成极限环振荡。取 |Δ| ≤ 2·|EPE|：对增益 ≥ 0.5 的
        # 位置仍保证不翻号（新 EPE = EPE − gain·Δ 符号不变），同时避免
        # 限幅过保守导致的欠校正爬行（像素量化冻结）。
        step_lim = min(cfg.MAX_EDGE_STEP_NM, 2.0 * max(abs(epe), 0.25))
        move = float(np.clip(cmd, -step_lim, step_lim))
        # 累计偏置上限：角落区域印刷轮廓跟不上掩模偏置（分辨率极限），
        # 封顶 EPE 会驱动边缘无益地持续外扩——到限后停止同向位移
        if cfg.MAX_BIAS_NM > 0:
            bias = seg.d - seg.d_ref
            if abs(bias) >= cfg.MAX_BIAS_NM and move * bias > 0:
                move = 0.0
        seg.prev_move = move
        moves.append(move)
    return moves


def _collinear_run_len(segs, i, step):
    """从段 i 沿链方向 step 走，累计与它共线的连续边缘碎片总长
    （到切口/断阶/方向改变为止）。用于估算切口相邻的"整条边"长度。"""
    i = i % len(segs)
    n0 = segs[i].n
    total = 0.0
    k = i
    seen = set()
    while k % len(segs) not in seen:
        seen.add(k % len(segs))
        s = segs[k % len(segs)]
        if s.is_cut or s.is_jog:
            break
        if abs(cross2(n0, s.n)) > np.sin(np.deg2rad(1.0)):
            break
        total += s.length()
        k += step
    return total


def _rebuild_polygon(segs, cfg):
    """移动后重建多边形几何。

    连接规则（按相邻两段的类型）：
      - 切口 ↔ 边：切口**锚定在角点**上直接构造（V1 = 角点 + s·û_a，
        V2 = 角点 + s·û_b，角点 = 两条邻边移动后直线的交点）。不能靠
        "切口直线 ∩ 邻边直线"求交点——交点会滑过角点，把切口拉成
        倾斜长段、s 漂移为负值，几何退化。
      - 断阶 ↔ 边：直线交点（断阶线 ⊥ 边线，良态）。
      - 边 ↔ 边（近平行）：共线碎片，位移不同则插入新断阶。
      - 边 ↔ 边（不平行）：直线交点（角点处）。
    返回新的 Segment 列表（p1/p2 一致、法向重算、方向统一 CCW）。
    """
    n = len(segs)
    if n == 0:
        return []

    # ---- 1. 计算每段的新直线偏移（切口段在第 2 步统一处理） ----
    for seg in segs:
        if not seg.is_cut:
            seg.d = seg.d + seg.prev_move

    # ---- 2. 方形切口（serif/hammerhead）锚定构造（两遍） ----
    # 每个角点有一对切口段 (cut_a: V1→V2, cut_b: V2→V3)，共享 corner_pt：
    #   V1 = 角点 + s·û_a（进边缩短端）
    #   V2 = V1 + s·û_b（方块外角）
    #   V3 = 角点 + s·û_b（出边起点）
    # s ↔ d 为线性关系 d = d_ref + s_sign·s（无 √2 因子）。
    # 邻边方向必须从"边直线"（法向 n → 方向 d=(−n_y, n_x)）精确导出，
    # 不能用碎片几何端点——端点带断阶台阶噪声，方向逐年倾斜，
    # 与切口锚定互相放大（正反馈），最终折叠多边形。
    # 第一遍：找邻边（跳过断阶和同角的另一段切口）、算角点
    cut_info = {}       # i -> (corner, seg_in, seg_out, d_in, d_out)
    for i, seg in enumerate(segs):
        if not seg.is_cut:
            continue
        k = (i - 1) % n
        while (segs[k].is_jog
               or (segs[k].is_cut and segs[k].corner_pt is seg.corner_pt)):
            k = (k - 1) % n
        seg_in = segs[k]
        k = (i + 1) % n
        while (segs[k].is_jog
               or (segs[k].is_cut and segs[k].corner_pt is seg.corner_pt)):
            k = (k + 1) % n
        seg_out = segs[k]
        corner = _line_intersection(seg_in.n, seg_in.d, seg_out.n, seg_out.d)
        d_in = np.array([-seg_in.n[1], seg_in.n[0]])
        if np.dot(d_in, corner - seg_in.p1) < 0:
            d_in = -d_in
        d_out = np.array([-seg_out.n[1], seg_out.n[0]])
        if np.dot(d_out, seg_out.p2 - corner) < 0:
            d_out = -d_out
        cut_info[i] = (corner, seg_in, seg_out, d_in, d_out)

    # 沿链向两侧找"下一个不同角点的切口"，用两个角点间的距离作为
    # "整条邻边长度"（不能用共线碎片游程：两端 serif 增长会把中间
    # 碎片吃短，游程自我收缩，把 serif 压回小尺寸，角落修不动）
    corner_dist = {}
    for i in cut_info:
        corner = cut_info[i][0]
        for step in (-1, +1):
            k = (i + step) % n
            guard = 0
            while guard < len(segs) and not (
                    segs[k].is_cut
                    and segs[k].corner_pt is not segs[i].corner_pt):
                k = (k + step) % n
                guard += 1
            if segs[k].is_cut and k != i:
                corner_dist[(i, step)] = np.linalg.norm(corner - cut_info[k][0])
            else:
                corner_dist[(i, step)] = 1e9    # 整圈都没找到（防御）

    # 第二遍：按"角点对"更新切口尺寸、锚定 V1/V2/V3、吞噬被覆盖的碎片
    cut_pts = {}
    drop = set()        # 被切口滑动完全覆盖、需从链中删除的碎片索引
    for i, (corner, seg_in, seg_out, d_in, d_out) in cut_info.items():
        # 成对处理：跳过"对的第二段"（其数据与第一段相同）
        prev_is_partner = (segs[(i - 1) % n].is_cut
                           and segs[(i - 1) % n].corner_pt is segs[i].corner_pt)
        if prev_is_partner:
            continue
        partner = ((i + 1) % n
                   if (segs[(i + 1) % n].is_cut
                       and segs[(i + 1) % n].corner_pt is segs[i].corner_pt)
                   else None)
        seg = segs[i]
        # 尺寸更新：s_old = s_sign·(d − d_ref)；两段的位移取平均，
        # 保证方块保持正方形（s 沿两个方向同步增长）
        s_old = seg.s_sign * (seg.d - seg.d_ref)
        m_avg = seg.prev_move
        if partner is not None:
            m_avg = 0.5 * (seg.prev_move + segs[partner].prev_move)
        # 上限 = [SERIF_MAX, 短邻边长度的一半 − 2nm] 的较小者：
        # 一条边两端各有一个切口，两边都自由增长会把整条边吃掉
        L_in = corner_dist.get((i, -1), 1e9)
        L_out = corner_dist.get((i, +1), 1e9)
        s_cap = max(1.0, 0.5 * min(L_in, L_out) - 2.0)
        s = np.clip(s_old + seg.s_sign * m_avg,
                    1.0, min(cfg.SERIF_MAX_NM, s_cap))
        seg.d = seg.d_ref + seg.s_sign * s
        if partner is not None:
            segs[partner].d = segs[partner].d_ref + segs[partner].s_sign * s
        # 方块几何
        v1 = corner + s * d_in
        v2 = v1 + s * d_out
        v3 = corner + s * d_out
        cut_pts[i] = (v1, v2)
        if partner is not None:
            cut_pts[partner] = (v2, v3)

        # 切口滑动吞噬碎片：s 可能超过单个碎片长度（8~16nm），锚定点
        # 滑过碎片端点时该碎片被完全覆盖，必须从链中删除，否则碎片
        # 反转成负长度、多边形折叠（表现为 serif 被有效性检查压回）。
        k = (i - 1) % n
        while True:
            s2 = segs[k]
            if s2.is_cut:
                break
            if not s2.is_jog:
                if s <= np.linalg.norm(s2.p1 - corner) + 1e-6:
                    break               # 远端仍在切口外侧 → 部分保留
                drop.add(k)             # 远端被覆盖 → 吞掉继续
            k = (k - 1) % n
        k = ((partner if partner is not None else i) + 1) % n
        while True:
            s2 = segs[k]
            if s2.is_cut:
                break
            if not s2.is_jog:
                if s <= np.linalg.norm(s2.p2 - corner) + 1e-6:
                    break
                drop.add(k)
            k = (k + 1) % n

    # 删除被吞噬的碎片，并重映射 cut_pts 的索引
    if drop:
        old_to_new = {}
        keep = []
        for idx, s in enumerate(segs):
            if idx in drop:
                continue
            old_to_new[idx] = len(keep)
            keep.append(s)
        segs = keep
        n = len(segs)
        cut_pts = {old_to_new[i]: v for i, v in cut_pts.items()
                   if i in old_to_new}

    # ---- 3. 段与段之间的连接点 ----
    # 平行判定用 1° 角容差（不能用机器精度）：同一长边上的相邻碎片
    # 因断阶台阶差会产生 ~1° 的轻微不共线，若按"非平行"求交点，
    # 两条近平行线的交点在数万 nm 之外 → 多边形产生尖刺/蝴蝶结。
    PARALLEL_TOL = np.sin(np.deg2rad(1.0))
    pt_prev = {}      # 段 i 的新 p1（与段 i-1 的连接点）
    jogs = {}         # 段 i-1 之后插入的断阶: (A, B) = (prev 新终点, seg 新起点)
    for i in range(n):
        prev, seg = segs[i - 1], segs[i]
        if prev.is_cut:
            pt = cut_pts[(i - 1) % n][1]
        elif seg.is_cut:
            pt = cut_pts[i][0]
        elif prev.is_jog or seg.is_jog:
            if abs(cross2(prev.n, seg.n)) < PARALLEL_TOL:
                # 防御：切口吞噬碎片后两个断阶可能相邻且平行（都垂直于
                # 同一条边），交点退化 → 取两端点中点
                pt = 0.5 * (prev.p2 + seg.p1)
            else:
                pt = _line_intersection(prev.n, prev.d, seg.n, seg.d)
        elif abs(cross2(prev.n, seg.n)) < PARALLEL_TOL:
            # 共线碎片：位移不同则用断阶连接（两点在过旧顶点的垂线上）
            V_old = seg.p1
            A = V_old + prev.prev_move * prev.n      # prev 的新终点
            B = V_old + seg.prev_move * seg.n        # seg 的新起点
            # 切口邻域不建断阶：断阶与切口锚定几何相互作用会折叠链序。
            # 改为合并（取平均位移，牺牲一点局部自由度换几何稳定）。
            near_cut = (segs[(i - 2) % n].is_cut or segs[(i + 1) % n].is_cut)
            if np.linalg.norm(A - B) < 0.25 or near_cut:
                pt = 0.5 * (A + B)                   # 合并
            else:
                jogs[i - 1] = (A, B)
                pt = None
        else:
            pt = _line_intersection(prev.n, prev.d, seg.n, seg.d)
        pt_prev[i] = pt

    # ---- 4. 组装新段列表（断阶跟在它的前驱段之后） ----
    new_segs = []
    for i in range(n):
        seg = segs[i]
        seg.p1 = (np.asarray(jogs[i - 1][1], dtype=float)
                  if (i - 1) in jogs else np.asarray(pt_prev[i], dtype=float))
        seg.p2 = (np.asarray(jogs[i][0], dtype=float)
                  if i in jogs else np.asarray(pt_prev[(i + 1) % n],
                                               dtype=float))
        new_segs.append(seg)
        if i in jogs:
            A, B = jogs[i]
            new_segs.append(Segment(n=np.zeros(2), d=0.0,
                                    p1=np.asarray(A, dtype=float),
                                    p2=np.asarray(B, dtype=float),
                                    d_ref=0.0, is_jog=True))

    # ---- 5. 丢弃过短的断阶，并把链缝合（每段终点 = 下一段起点） ----
    new_segs = [s for s in new_segs if not (s.is_jog and s.length() < 0.25)]
    m = len(new_segs)
    for i in range(m):
        nxt = new_segs[(i + 1) % m]
        if np.linalg.norm(new_segs[i].p2 - nxt.p1) > 0.5:
            mid = 0.5 * (new_segs[i].p2 + nxt.p1)
            new_segs[i].p2 = mid
            nxt.p1 = mid

    # ---- 4. 统一 CCW ----
    # 关键：边缘碎片与切口的法向（n）**永远不按几何端点重算**——
    # 端点含断阶台阶噪声，重算法向会使碎片逐年倾斜，相邻碎片倾斜
    # 差超过平行容差后交点飞出、多边形折叠（正反馈）。碎片法向保持
    # 原始边缘方向（断阶几何 = 平行碎片 + 垂直断阶，严格无倾斜）。
    # 只有断阶（人造连接段）按几何重算法向。
    from mask_ops import _signed_area
    verts = np.array([s.p1 for s in new_segs])
    if _signed_area(verts) < 0:
        new_segs = new_segs[::-1]
        for s in new_segs:
            s.p1, s.p2 = s.p2, s.p1
            s.n = -s.n          # 方向反转 → 法向与直线偏移同步反转
            s.d = -s.d
            if s.is_cut:
                s.s_sign = -s.s_sign
    for s in new_segs:
        if s.is_jog:
            L = s.length()
            if L < 1e-6:
                continue        # 退化断阶：保留原直线
            dvec = (s.p2 - s.p1) / L
            s.n = np.array([dvec[1], -dvec[0]])
            s.d = float(s.n @ s.p1)
            s.d_ref = s.d
    return new_segs


def _backup_polygon(segs):
    """备份一个多边形的全部可变状态（用于无效重建后的回退）。"""
    return [(s.n.copy(), s.d, s.p1.copy(), s.p2.copy(), s.prev_move)
            for s in segs]


def _restore_polygon(segs, backup):
    """从备份恢复多边形状态。"""
    for s, (n, d, p1, p2, mv) in zip(segs, backup):
        s.n, s.d, s.p1, s.p2, s.prev_move = n, d, p1, p2, mv


def _has_fold(rebuilt):
    """检测针状折返：相邻两段行进方向几乎反向（点积 < −0.99）。

    shapely 的 is_valid 查不出针状折返（它是合法的简单多边形），
    但针状折返是"切口吃掉整条边"后的链序翻转，必须单独检测。
    """
    for i, s in enumerate(rebuilt):
        d1 = s.p2 - s.p1
        d2 = rebuilt[(i + 1) % len(rebuilt)].p2 - rebuilt[(i + 1) % len(rebuilt)].p1
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        if np.dot(d1, d2) / (n1 * n2) < -0.99:
            return True
    return False


def apply_moves(polys, cfg):
    """用每段的 prev_move 重建多边形几何（位移已由 compute_moves 写好）。

    安全网：重建后做几何有效性检查（自交 + 针状折返），不通过则把
    位移减半重试（最多 3 次），仍无效就放弃本轮移动（恢复原几何）。
    """
    import shapely.geometry as sg

    out = []
    for poly in polys:
        backup = _backup_polygon(poly)
        for attempt in range(3):
            rebuilt = _rebuild_polygon(poly, cfg)
            verts = np.array([s.p1 for s in rebuilt])
            ok = (len(verts) >= 3 and not _has_fold(rebuilt)
                  and sg.Polygon(verts).is_valid)
            if ok:
                out.append(rebuilt)
                break
            # 无效：恢复原几何，位移减半再试
            _restore_polygon(poly, backup)
            for s in poly:
                s.prev_move *= 0.5
        else:
            # 3 次都无效：放弃本轮移动，原样保留上一轮的有效几何。
            # （不能做零位移重建——重建对含断阶/切口的几何不幂等，
            #   会把有效几何重建成无效。）
            _restore_polygon(poly, backup)
            for s in poly:
                s.prev_move = 0.0
            out.append(poly)
    return out


# ------------------------------------------------------------------
# 迭代主循环
# ------------------------------------------------------------------

def _contour_dev(printed, target_polys, cfg):
    """印刷轮廓与目标版图的对称平均偏差（nm）——最贴近"看得见的质量"
    的指标。用作历史最佳状态的判据：EPE 的 rms 由角点过渡区样本主导，
    在 serif 过冲时可能"变好"，但视觉上轮廓反而变差。"""
    import shapely.geometry as sg
    cont = extract_polygon(printed.astype(np.uint8), cfg.GRID_N,
                           cfg.PIXEL_SIZE_NM)
    if not cont:
        return 1e9
    cont_geo = sg.MultiPolygon([sg.Polygon(p) for p in cont])
    tgt_geo = sg.MultiPolygon([sg.Polygon(p) for p in target_polys])
    d1 = np.mean([tgt_geo.boundary.distance(sg.Point(q))
                  for p in cont for q in p])
    tpts = np.concatenate([np.linspace(p, np.roll(p, -1, axis=0), 40)[:-1].
                          reshape(-1, 2) for p in target_polys])
    d2 = np.mean([cont_geo.boundary.distance(sg.Point(q)) for q in tpts])
    return float(0.5 * (d1 + d2))


def run_opc(target_pixels, target_polys, cfg):
    """完整的迭代 OPC 主循环。

    流程：目标 → 初始掩模 → 前向仿真 → EPE → 收敛判断 →
          移动边缘/角落 → 重新光栅化 → （清理）→ 下一轮。
    返回 dict：最终掩模像素、多边形状态、空间像、印刷图、EPE 历史。
    """
    from visualize import save_frame

    pre = precompute_source_and_pupil(cfg)          # 光源/光瞳只算一次
    polys = init_polygons(target_polys, cfg)        # 含 serif 种子的段状态
    mask = polygons_to_pixels([np.array([s.p1 for s in poly])
                               for poly in polys],
                              cfg.GRID_N, cfg.PIXEL_SIZE_NM)
    history = []        # (rms, max|EPE|) 每轮一条
    best_dev, no_improve = np.inf, 0
    best_state = None   # 历史最佳状态（角点区域可能来回振荡，返回最佳）
    seg_epes, aerial, printed = [], None, None

    for it in range(cfg.MAX_ITER):
        # ---- 前向模型：空间像 → 胶模糊 → 阈值印刷 ----
        aerial = apply_resist_blur(aerial_image(mask, cfg, pre), cfg)
        printed = (aerial < cfg.RESIST_THRESHOLD).astype(np.uint8)

        # ---- EPE 与收敛判断 ----
        seg_epes, epe_all = compute_epe(polys, aerial, cfg)
        rms = float(np.sqrt(np.mean(np.square(epe_all))))
        mx = float(np.max(np.abs(epe_all)))
        history.append((rms, mx))
        # 真实轮廓偏差：用于"历史最佳"判据（比 rms 更贴近视觉质量）
        dev = _contour_dev(printed, target_polys, cfg)
        print("迭代 %2d:  RMS EPE = %.2f nm,  轮廓偏差 = %.2f nm"
              % (it, rms, dev))

        # 存帧必须在移动之前：画的是"本轮评估的几何状态 + 它的 EPE"，
        # 移动之后段列表结构变化（断阶增减），旧 EPE 会标错到新段上
        if cfg.SAVE_FRAMES:
            save_frame(cfg.OUT_DIR, it, target_pixels, target_polys, mask,
                       aerial, printed, polys, seg_epes, history, cfg)

        if dev < best_dev - 1e-9:
            best_dev, no_improve = dev, 0
            best_state = (mask.copy(), _copy_polys(polys), aerial.copy(),
                          printed.copy(), list(seg_epes), dev)
        else:
            no_improve += 1
            if no_improve >= 10:
                print("  收敛：连续 10 轮轮廓偏差无改善（最佳 %.2f nm）"
                      % best_dev)
                break

        # ---- 反馈移动 + 重建 ----
        compute_moves(seg_epes, polys, cfg, it)
        polys = apply_moves(polys, cfg)
        mask = polygons_to_pixels([np.array([s.p1 for s in poly])
                                   for poly in polys],
                                  cfg.GRID_N, cfg.PIXEL_SIZE_NM)

        # ---- 像素辅助操作 + 设计规则报告 ----
        # 清理只处理小碎屑（≤3px 的岛/洞，来自断阶/小切口光栅化的
        # 亚分辨率产物）。改动很小时**不重建多边形状态**（重建会重置
        # serif 生长、断阶和参考配对，代价远大于收益）：直接使用清理后
        # 的像素做仿真即可，下一轮光栅化会重现碎屑、再被清理，无害。
        mask_old = mask
        mask, changed = cleanup_mask(mask, cfg)
        if changed:
            n_changed = int(np.abs(mask.astype(int) - mask_old.astype(int)).sum())
            if n_changed > 10:
                print("  像素清理大改动（%d 像素），重提取多边形并重配目标参考"
                      % n_changed)
                polys = init_polygons_from_pixels(mask, target_polys, cfg)
        drc = check_design_rules(mask, cfg)
        if not drc["ok"]:
            print("  [DRC] min_width=%.1f nm (限 %.1f), min_space=%.1f nm (限 %.1f)"
                  % (drc["min_width"], cfg.MIN_WIDTH_NM,
                     drc["min_space"], cfg.MIN_SPACE_NM))

    # 返回历史最佳状态（若最后几轮在角点区振荡，最佳可能出现在中间轮）
    if best_state is not None:
        mask, polys, aerial, printed, seg_epes, best_dev = best_state
        print("  输出历史最佳状态：轮廓偏差 %.2f nm" % best_dev)

    return {"mask": mask, "polys": polys, "aerial": aerial,
            "printed": printed, "history": history,
            "seg_epes": seg_epes, "target": target_pixels,
            "target_polys": target_polys, "best_dev": best_dev}
