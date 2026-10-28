# -*- coding: utf-8 -*-
"""
mask_ops.py —— 像素版图与多边形之间的转换、清理、设计规则检查。

全项目唯一的坐标换算约定（重要！）：
  像素数组 mask[i, j]，i 为行（对应 y），j 为列（对应 x），
  像素 (i,j) 中心的物理坐标:
      x = (j - GRID_N/2 + 0.5) * dx
      y = (i - GRID_N/2 + 0.5) * dx
  即原点在图像正中心，且带"半像素偏移"——这样网格关于原点严格对称
  （若不偏移，x 轴两侧像素到原点的距离不对称，会引入 0.5px 系统误差）。
  所有坐标换算必须走 x_of_j / y_of_i / j_of_x / i_of_y，不许手写公式。

多边形顶点坐标是网格"顶点"（不是像素中心），换算时无半像素偏移：
      x = (j - GRID_N/2) * dx
"""
import numpy as np
import cv2
from scipy import ndimage


# ------------------------------------------------------------------
# 坐标换算（全项目唯一入口）
# ------------------------------------------------------------------

def x_of_j(j, grid_n, dx):
    """像素列号 j → 该像素中心的物理 x (nm)。"""
    return (np.asarray(j, dtype=float) - grid_n / 2 + 0.5) * dx


def y_of_i(i, grid_n, dx):
    """像素行号 i → 该像素中心的物理 y (nm)。"""
    return (np.asarray(i, dtype=float) - grid_n / 2 + 0.5) * dx


def j_of_x(x, grid_n, dx):
    """物理 x (nm) → 像素列号（浮点，可为负/出界）。"""
    return np.asarray(x, dtype=float) / dx + grid_n / 2 - 0.5


def i_of_y(y, grid_n, dx):
    """物理 y (nm) → 像素行号（浮点，可为负/出界）。"""
    return np.asarray(y, dtype=float) / dx + grid_n / 2 - 0.5


def vertex_x_of_j(j, grid_n, dx):
    """网格顶点（列线）j → 物理 x (nm)。顶点坐标无半像素偏移。"""
    return (np.asarray(j, dtype=float) - grid_n / 2) * dx


def vertex_y_of_i(i, grid_n, dx):
    """网格顶点（行线）i → 物理 y (nm)。顶点坐标无半像素偏移。"""
    return (np.asarray(i, dtype=float) - grid_n / 2) * dx


# ------------------------------------------------------------------
# 光栅化：多边形 → 像素
# ------------------------------------------------------------------

def polygons_to_pixels(polygons_nm, grid_n, dx, wrap=False):
    """把多边形列表（nm 坐标）光栅化成像素数组（1=图形）。

    用 OpenCV fillPoly 填充。注意 OpenCV 的图像约定：
    像素坐标为 (列, 行) 即 (x, y) 的交换，行对应物理 y、列对应物理 x。

    wrap=True：周期边界模式——跨出视场的部分按 ±grid_n 平移后重复绘制，
    使跨越边界的图形自动环绕到对面（与 FFT 的周期边界一致）。
    """
    canvas = np.zeros((grid_n, grid_n), np.uint8)
    if not polygons_nm:
        return canvas
    shifts = ((-grid_n, 0, grid_n) if wrap else (0,))
    for poly in polygons_nm:
        col = j_of_x(poly[:, 0], grid_n, dx)   # x → 列（像素中心坐标系）
        row = i_of_y(poly[:, 1], grid_n, dx)   # y → 行
        base = np.stack([col, row], axis=-1)
        for sx in shifts:
            for sy in shifts:
                pts = (base + np.array([sx, sy], dtype=float)).astype(np.float32)
                cv2.fillPoly(canvas,
                             [np.round(pts).astype(np.int32).reshape(-1, 1, 2)],
                             1)
    return canvas


# ------------------------------------------------------------------
# 像素 → 多边形提取
# ------------------------------------------------------------------

def extract_polygon(mask_pixels, grid_n, dx):
    """从像素版图提取多边形边界（闭合环，nm 坐标）。

    用 OpenCV findContours 做边界追踪（等价于二值 marching squares 的
    轴对齐结果，但处理斜角接触、环追踪等细节都由库代劳，更可靠）。
    findContours 返回的轮廓点位于"像素顶点"上（即网格线交点），
    因此物理坐标用无半像素偏移的 vertex_* 换算。

    返回顶点按逆时针排列（外环；有洞时内环顺时针，本项目的图形无洞）。
    """
    contours, _ = cv2.findContours(
        mask_pixels.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    polys = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        # cnt: (k,1,2)，[...,0]=列(x)，[...,1]=行(y)
        pts = np.stack([vertex_x_of_j(cnt[:, 0, 0], grid_n, dx),
                        vertex_y_of_i(cnt[:, 0, 1], grid_n, dx)], axis=-1)
        pts = _remove_redundant(pts)          # 去重复点与共线中间点
        if len(pts) < 3:
            continue
        # findContours 在图像坐标（y 向下）里外环为逆时针，
        # 换到物理坐标（y 向上）后变成顺时针 → 统一反转为逆时针
        if _signed_area(pts) < 0:
            pts = pts[::-1]
        polys.append(pts)
    return polys


def _signed_area(poly):
    """鞋带公式带符号面积：>0 逆时针。"""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def cross2(a, b):
    """二维叉积（标量）。numpy 2.x 的 np.cross 不再支持二维向量，
    全项目统一用它做共线/平行判定。"""
    return float(a[0] * b[1] - a[1] * b[0])


def _remove_redundant(pts, tol=1e-6):
    """去掉首尾重复与共线中间点（共线判定用叉积，避免角度除法）。"""
    if len(pts) < 3:
        return pts
    # findContours 的轮廓首尾重复（闭合），先去尾点，再按环处理
    if np.allclose(pts[0], pts[-1], atol=tol):
        pts = pts[:-1]
    if len(pts) < 3:
        return pts
    n = len(pts)
    keep = []
    for k in range(n):
        a = pts[(k - 1) % n]    # 前一点
        b = pts[k]
        c = pts[(k + 1) % n]    # 后一点
        v1, v2 = b - a, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < tol or n2 < tol:
            continue        # b 与 a 或 c 重合 → 丢弃 b
        # 叉积 ≈ 0 且同向（点积 > 0）→ b 在直线中间 → 丢弃
        if abs(cross2(v1, v2)) < tol * n1 * n2 and np.dot(v1, v2) > 0:
            continue
        keep.append(b)
    return np.array(keep)


# ------------------------------------------------------------------
# 角落检测（基于几何 + 象限填充数，与环方向无关）
# ------------------------------------------------------------------

def classify_corner(vertex, prev, nxt, polygons_nm, tol_deg=0.5):
    """判断顶点是否为 90° 角点，返回 'convex' / 'concave' / None（非角点）。

    方法（与环绕方向无关，可靠）：
      1. 相邻两边必须互相垂直（点积 ≈ 0），否则不是直角角点；
      2. 取角点四周 4 个象限的试探点（±dx/2 偏移），统计落在图形
         内部的象限数：占 1 个象限 = 凸角，占 3 个象限 = 凹角，
         占 2 个象限 = 直边（非角点）。
    用 shapely 对多边形做精确包含判断，不受光栅化影响。
    """
    import shapely.geometry

    e1 = vertex - prev      # 指向角点的边方向
    e2 = nxt - vertex       # 离开角点的边方向
    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    # 垂直判定：|cosθ| 足够小（直角角点的邻边夹角为 90°）
    cos = abs(np.dot(e1, e2)) / (n1 * n2)
    if cos > np.sin(np.deg2rad(tol_deg)):
        return None

    # 象限试探点（±dx/2 偏移量级即可，用边长的 1/4 与 dx 取小更稳妥）
    d = min(n1, n2, 2.0) * 0.25
    sxs = np.array([-d, d, -d, d])
    sys = np.array([d, d, -d, -d])
    pts = vertex + np.stack([sxs, sys], axis=-1)

    if isinstance(polygons_nm, shapely.geometry.base.BaseGeometry):
        geo = polygons_nm
    else:
        # 多边形列表 → shapely 并集（多图形时角点包含判断需整个版图）
        geo = shapely.ops.unary_union(
            [shapely.geometry.Polygon(p) for p in polygons_nm])
    inside = sum(1 for p in pts if geo.contains(shapely.geometry.Point(p)))
    if inside == 1:
        return "convex"
    if inside == 3:
        return "concave"
    return None


# ------------------------------------------------------------------
# 形态学清理（像素辅助操作）
# ------------------------------------------------------------------

def cleanup_mask(mask, cfg):
    """保守的像素级清理：去孤立小岛、填小空洞（拓扑保护）。返回 (新mask, 是否改动)。

    只在极端情况下触发（如自定义版图含噪点），正常 OPC 迭代几乎不会改动。
    拓扑保护：绝不拆分/合并主图形——小岛只在"不是最大连通域"时才删。
    """
    changed = False
    m = mask.copy()

    if cfg.CLEANUP_ISLANDS:
        labels, nlab = ndimage.label(m)
        if nlab > 1:
            sizes = ndimage.sum(m, labels, range(1, nlab + 1))
            main = np.argmax(sizes) + 1
            for lab in range(1, nlab + 1):
                if lab != main and sizes[lab - 1] <= cfg.MIN_ISLAND_PX:
                    m[labels == lab] = 0
                    changed = True

    if cfg.CLEANUP_HOLES:
        # 背景（0）的连通域中，不接触图像边界的才是"洞"
        bg_labels, nbg = ndimage.label(1 - m)
        border_labs = set(np.unique(np.concatenate(
            [bg_labels[0, :], bg_labels[-1, :],
             bg_labels[:, 0], bg_labels[:, -1]])))
        for lab in range(1, nbg + 1):
            if lab in border_labs:
                continue
            size = int(np.sum(bg_labels == lab))
            if size <= cfg.MIN_HOLE_PX:
                m[bg_labels == lab] = 1
                changed = True

    if cfg.CLEANUP_OPEN:
        kernel = np.ones((3, 3), np.uint8)
        m2 = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        if m2.sum() != m.sum():
            m, changed = m2, True

    return m, changed


# ------------------------------------------------------------------
# 设计规则检查（DRC，默认仅报告）
# ------------------------------------------------------------------

def check_design_rules(mask, cfg):
    """用距离变换估计最小线宽/最小间距，返回 dict。默认只报告不强制。"""
    dx = cfg.PIXEL_SIZE_NM
    # 最小线宽：图形内部各点到边界最大距离的 2 倍（近似，轴对齐图形上精确）
    dt_in = ndimage.distance_transform_edt(mask) * dx
    min_width = 2 * dt_in.max()
    # 最小间距：背景内同理
    dt_out = ndimage.distance_transform_edt(1 - mask) * dx
    min_space = 2 * dt_out.max()
    ok = (min_width >= cfg.MIN_WIDTH_NM) and (min_space >= cfg.MIN_SPACE_NM)
    return {"min_width": min_width, "min_space": min_space, "ok": ok}
