# -*- coding: utf-8 -*-
"""
geometry.py —— 目标版图的生成与读写。

职责：
  1. 生成参数化的示例目标图形（多边形顶点列表，单位 nm，图形中心在原点）；
  2. 从文件读取自定义像素版图（.txt 的 0/1 矩阵 或 .png 灰度图）；
  3. 可选的 GDS 导出。

坐标约定：多边形顶点为 float 数组，形状 (k,2)，列为 (x, y)，单位 nm。
所有示例图形均以原点为中心、关于 x/y 轴对称放置，便于观察对称性。
"""
import os

import cv2
import numpy as np
import shapely.affinity
import shapely.geometry
import shapely.ops


# ------------------------------------------------------------------
# 参数化示例图形生成器
# ------------------------------------------------------------------
# 每个生成器返回 List[np.ndarray]：多边形顶点列表（可能有多个多边形），
# 顶点按逆时针顺序排列（shapely 保证外环逆时针），掩模内部在行进方向左侧。

def _rect(x0, y0, x1, y1):
    """轴对齐矩形 [x0,x1]×[y0,y1] 的逆时针顶点数组。"""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _to_vertex_lists(geom):
    """把 shapely 几何对象转成多边形顶点列表（支持 MultiPolygon）。"""
    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
    out = []
    for g in geoms:
        coords = np.asarray(g.exterior.coords)[:-1]  # 去掉闭合重复点
        # 保证逆时针（shapely 外环即逆时针，这里双保险）
        if _signed_area(coords) < 0:
            coords = coords[::-1]
        out.append(coords)
    return out


def _signed_area(poly):
    """鞋带公式求多边形带符号面积：>0 逆时针（CCW），<0 顺时针。"""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _union_rects(rects):
    """多个矩形的并集 → 多边形顶点列表。用 shapely 做几何布尔运算。"""
    polys = [shapely.geometry.Polygon(r) for r in rects]
    return _to_vertex_lists(shapely.ops.unary_union(polys))


def l_shape(cfg):
    """L 形：一竖臂 + 一横臂在左下角重合。看凸角 serif 与凹角 hammerhead。"""
    w = cfg.L_WIDTH_NM
    arm = cfg.L_ARM_NM
    rects = [
        _rect(-w / 2, -arm / 2, w / 2, arm / 2),        # 竖臂（对称居中）
        _rect(-arm / 2, -w / 2, w / 2, w / 2),          # 横臂（向左伸出）
    ]
    return _union_rects(rects)


def t_shape(cfg):
    """T 形：顶部横梁 + 中央竖杆。横梁两端是凸角，梁杆交接处是凹角。"""
    w = cfg.T_WIDTH_NM
    bar, stem = cfg.T_BAR_NM, cfg.T_STEM_NM
    rects = [
        _rect(-bar / 2, stem / 2 - w, bar / 2, stem / 2),   # 横梁（顶部）
        _rect(-w / 2, -stem / 2, w / 2, stem / 2),          # 竖杆（中央）
    ]
    return _union_rects(rects)


def dense_lines(cfg):
    """密集线条阵列：N 条竖直等间距线条。看线条周期邻近效应与边缘位移。"""
    w, pitch, n = cfg.LINES_W, cfg.LINES_PITCH, cfg.LINES_N
    x0 = -pitch * (n - 1) / 2.0
    return [_rect(x0 + k * pitch - w / 2, -200.0, x0 + k * pitch + w / 2, 200.0)
            for k in range(n)]


def contact_array(cfg):
    """接触孔阵列：n×n 个方孔。强二维邻近效应，角落圆化明显。"""
    s, pitch, n = cfg.CONT_SIZE, cfg.CONT_PITCH, cfg.CONT_N
    x0 = -pitch * (n - 1) / 2.0
    return [_rect(x0 + i * pitch - s / 2, x0 + j * pitch - s / 2,
                  x0 + i * pitch + s / 2, x0 + j * pitch + s / 2)
            for i in range(n) for j in range(n)]


def line_end_gap(cfg):
    """线端间隙：两条水平线端对端、中间留小间隙。看线端回缩修正。"""
    w, ln, g = cfg.GAP_W, cfg.GAP_LEN, cfg.GAP
    return [
        _rect(-g / 2 - ln, -w / 2, -g / 2, w / 2),   # 左侧线（线端在 x=-g/2）
        _rect(g / 2, -w / 2, g / 2 + ln, w / 2),     # 右侧线（线端在 x=+g/2）
    ]


def isolated_line(cfg):
    """孤立线条：单独一条竖直线。与密集线条对比可看邻近效应。"""
    w, ln = cfg.ISO_W, cfg.ISO_LEN
    return [_rect(-w / 2, -ln / 2, w / 2, ln / 2)]


# ------------------------------------------------------------------
# 自定义版图读写
# ------------------------------------------------------------------

def load_mask_txt(path):
    """读取 0/1 文本矩阵（空格/逗号/任意空白分隔），返回 uint8 像素数组。"""
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("版图 txt 必须是二维 0/1 矩阵")
    return (arr >= 0.5).astype(np.uint8)


def load_mask_png(path):
    """读取灰度 PNG：像素值 ≥128 记为图形(1)，否则记为背景(0)。"""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("无法读取图像文件: %s" % path)
    return (gray >= 128).astype(np.uint8)


def load_custom_mask(path, grid_n):
    """按扩展名读取自定义像素版图（txt 0/1 矩阵 或 png 灰度图）。"""
    ext = os.path.splitext(path)[1].lower()
    mask = load_mask_txt(path) if ext == ".txt" else load_mask_png(path)
    if mask.shape != (grid_n, grid_n):
        raise ValueError(
            "自定义版图尺寸 %s 与 GRID_N=%d 不一致：请在 config.py 里把 "
            "GRID_N 改成版图边长，或用图像软件把版图缩放成 %d×%d。"
            % (mask.shape, grid_n, grid_n, grid_n))
    return mask


def export_gds(polygons_nm, path):
    """可选：把掩模多边形导出为 GDS 文件（gdstk，单位换算 nm→μm）。"""
    import gdstk
    lib = gdstk.Library()
    cell = lib.new_cell("OPC_MASK")
    for poly in polygons_nm:
        cell.add(gdstk.Polygon(poly / 1000.0, layer=1))
    lib.write_gds(path)


# ------------------------------------------------------------------
# 主入口：按 config 生成目标版图
# ------------------------------------------------------------------

def make_target(cfg, out_dir=None):
    """按 config.DEMO_NAME 生成目标版图。

    返回:
      target_pixels : uint8 数组 (GRID_N, GRID_N)，1=图形（不透明），0=背景
      target_polys  : List[np.ndarray]，目标图形多边形顶点（nm，几何真值）
    """
    from mask_ops import polygons_to_pixels, extract_polygon

    generators = {
        "l_shape": l_shape,
        "t_shape": t_shape,
        "dense_lines": dense_lines,
        "contact_array": contact_array,
        "line_end_gap": line_end_gap,
        "isolated_line": isolated_line,
    }
    if cfg.DEMO_NAME == "custom":
        mask = load_custom_mask(cfg.CUSTOM_MASK_FILE, cfg.GRID_N)
        polys = extract_polygon(mask, cfg.GRID_N, cfg.PIXEL_SIZE_NM)
        if not polys:
            raise ValueError("自定义版图中没有检测到任何图形")
        target_pixels = mask
        target_polys = polys
    elif cfg.DEMO_NAME in generators:
        polys = generators[cfg.DEMO_NAME](cfg)
        target_pixels = polygons_to_pixels(polys, cfg.GRID_N, cfg.PIXEL_SIZE_NM)
        target_polys = polys
    else:
        raise ValueError("未知 DEMO_NAME=%r，可选: %s"
                         % (cfg.DEMO_NAME, sorted(generators) + ["custom"]))
    return target_pixels, target_polys
