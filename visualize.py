# -*- coding: utf-8 -*-
"""
visualize.py —— 结果可视化。

每帧输出一张总览图（3×3 网格）：
  第 1 行: 目标版图 | 当前掩模 | 空间像热图
  第 2 行: 印刷轮廓对比 | EPE 图 | （目标轮廓参考）
  第 3 行: EPE 收敛曲线（占满整行）

配色约定（遵循数据可视化规范）：
  - 空间像热图: viridis 顺序色带（连续量 0~1，单一色相，感知均匀）
  - EPE 图: RdBu_r 发散色带 + 对称零点（正负极性数据双色相+中性中点）
  - 轮廓对比: 目标/印刷/掩模三条线用固定颜色 + 图例（身份不只靠颜色）
  - 收敛曲线: 单轴（rms 与 max|EPE| 同单位 nm，不做双轴图）
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")     # 无窗口环境，直接存 PNG
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from mask_ops import extract_polygon

# 固定色（色盲安全，Okabe-Ito 系）
C_TARGET = "#D55E00"      # 目标轮廓（朱红）
C_PRINT = "#0072B2"       # 印刷轮廓（蓝）
C_MASK = "#009E73"        # 掩模轮廓（绿）
C_RMS = "#0072B2"         # rms 曲线（蓝）
C_MAX = "#D55E00"         # max|EPE| 曲线（朱红）


def setup_style():
    """全局绘图样式（中文显示 + 负号）。在模块导入时调用一次。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.grid"] = False


def _limits(cfg):
    """视场物理坐标范围 (xmin, xmax, ymin, ymax)，单位 nm。"""
    half = cfg.GRID_N * cfg.PIXEL_SIZE_NM / 2.0
    return -half, half, -half, half


def _imshow(ax, img, cfg, cmap):
    """按物理坐标画像素图（原点在中心，y 轴向上）。"""
    xmin, xmax, ymin, ymax = _limits(cfg)
    ax.imshow(img, cmap=cmap, extent=[xmin, xmax, ymin, ymax],
              origin="lower", interpolation="nearest")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")


def _draw_polygons(ax, polys_nm, color, lw=1.2, label=None):
    """画多边形轮廓线（自动闭合）。返回图例句柄。"""
    from matplotlib.patches import Polygon as MplPolygon
    handles = []
    for poly in polys_nm:
        h = MplPolygon(np.asarray(poly, dtype=float), closed=True,
                       fill=False, edgecolor=color, linewidth=lw)
        ax.add_patch(h)
        handles.append(h)
    if label is not None and handles:
        # 用 Line2D 做一个代表句柄（图例里一条线而不是一个多边形块）
        handles = [Line2D([0], [0], color=color, lw=lw, label=label)]
    return handles


def _segments_of(polys):
    """把段状态转成 (顶点数组, 段线段数组)，供轮廓绘制与 EPE 着色。"""
    verts, lines = [], []
    for poly in polys:
        verts.append(np.array([s.p1 for s in poly]))
        lines.extend([(s.p1, s.p2) for s in poly])
    return verts, lines


def _draw_overview(outdir, fname, title, target, target_polys, mask, aerial,
                   printed, polys, seg_epes, history, cfg):
    """画一张完整的 3×3 总览图并存盘。"""
    xmin, xmax, ymin, ymax = _limits(cfg)

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.72],
                          hspace=0.32, wspace=0.30)

    # ---- 1) 目标版图 ----
    ax = fig.add_subplot(gs[0, 0])
    _imshow(ax, target, cfg, cmap="binary")
    ax.set_title("目标版图")

    # ---- 2) 当前掩模 + 掩模轮廓 ----
    ax = fig.add_subplot(gs[0, 1])
    _imshow(ax, mask, cfg, cmap="binary")
    verts, _ = _segments_of(polys)
    _draw_polygons(ax, verts, C_MASK, lw=1.0)
    ax.set_title("当前掩模版图（绿线=修正后多边形）")

    # ---- 3) 空间像热图（顺序色带 + 掩模轮廓参考） ----
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(aerial, cmap="viridis", vmin=0.0, vmax=1.0,
                   extent=[xmin, xmax, ymin, ymax], origin="lower")
    _draw_polygons(ax, verts, "white", lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("归一化强度（开域=1）")
    ax.set_title("空间像（Hopkins 前向模型）")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")

    # ---- 4) 印刷轮廓对比（图例，身份不只靠颜色） ----
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor("white")
    printed_cont = extract_polygon(printed, cfg.GRID_N, cfg.PIXEL_SIZE_NM)
    _draw_polygons(ax, target_polys, C_TARGET, lw=1.6)
    _draw_polygons(ax, printed_cont, C_PRINT, lw=1.6)
    _draw_polygons(ax, verts, C_MASK, lw=0.9)
    ax.legend(handles=[
        Line2D([0], [0], color=C_TARGET, lw=1.6, label="目标轮廓"),
        Line2D([0], [0], color=C_PRINT, lw=1.6, label="印刷轮廓（阈值 %.1f）"
               % cfg.RESIST_THRESHOLD),
        Line2D([0], [0], color=C_MASK, lw=0.9, label="修正后掩模"),
    ], loc="upper right", framealpha=0.9)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("印刷轮廓 vs 目标")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")

    # ---- 5) EPE 图：目标版图 vs 成像后图形的带符号偏差 ----
    # 印刷轮廓按"与目标的带符号距离"着色（发散色带+对称零点）：
    # 在目标内部（印小）= 蓝，在目标外部（印大）= 红，贴合 = 中性色。
    import shapely.geometry as sg
    ax = fig.add_subplot(gs[1, 1])
    _draw_polygons(ax, target_polys, C_TARGET, lw=1.4)
    printed_cont = extract_polygon(printed, cfg.GRID_N, cfg.PIXEL_SIZE_NM)
    tgt_geo = sg.MultiPolygon([sg.Polygon(p) for p in target_polys])
    lines, devs = [], []
    for poly in printed_cont:
        for k in range(len(poly)):
            p1, p2 = poly[k], poly[(k + 1) % len(poly)]
            mid = 0.5 * (p1 + p2)
            pt = sg.Point(mid)
            d = tgt_geo.boundary.distance(pt)
            # 带符号偏差：中点在目标内部 = 印小（负），外部 = 印大（正）
            devs.append(-d if tgt_geo.contains(pt) else d)
            lines.append((p1, p2))
    if lines:
        m = max(1.0, float(np.max(np.abs(devs))))
        norm = TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)   # 0 居中为中性色
        lc = LineCollection(lines, cmap="RdBu_r", norm=norm, linewidths=2.2)
        lc.set_array(np.asarray(devs))
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("EPE (nm)")
        ax.legend(handles=[Line2D([0], [0], color=C_TARGET, lw=1.4,
                                  label="目标版图")],
                  loc="upper right", framealpha=0.9)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title("边缘放置误差 EPE（印刷轮廓着色：红=印大 蓝=印小）")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")

    # ---- 6) 印刷图（参考） ----
    ax = fig.add_subplot(gs[1, 2])
    _imshow(ax, printed, cfg, cmap="binary")
    _draw_polygons(ax, target_polys, C_TARGET, lw=0.9)
    ax.set_title("印刷图形（阈值化空间像）")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")

    # ---- 7) 收敛曲线（单轴） ----
    ax = fig.add_subplot(gs[2, :])
    its = np.arange(len(history))
    rms = np.array([h[0] for h in history])
    mx = np.array([h[1] for h in history])
    ax.plot(its, rms, "-o", color=C_RMS, lw=1.8, ms=4, label="RMS EPE")
    ax.plot(its, mx, "-s", color=C_MAX, lw=1.8, ms=4, label="max|EPE|")
    ax.axhline(cfg.EPE_TOL_NM, color="gray", ls="--", lw=1.0,
               label="容差 %.1f nm" % cfg.EPE_TOL_NM)
    if len(history):
        ax.annotate("%.2f nm" % rms[-1], (its[-1], rms[-1]),
                    textcoords="offset points", xytext=(-8, 10), color=C_RMS)
        ax.annotate("%.2f nm" % mx[-1], (its[-1], mx[-1]),
                    textcoords="offset points", xytext=(-8, -14), color=C_MAX)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("EPE (nm)")          # rms 与 max 同单位，单轴即可
    ax.set_xlim(0, max(len(history) - 1, 1))
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("EPE 收敛曲线")

    fig.suptitle(title, fontsize=14)
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, fname), bbox_inches="tight")
    plt.close(fig)


def save_frame(outdir, it, target, target_polys, mask, aerial, printed,
               polys, seg_epes, history, cfg):
    """保存单次迭代的总览帧 iter_%03d.png。"""
    _draw_overview(outdir, "iter_%03d.png" % it,
                   "迭代 %d / %d" % (it, cfg.MAX_ITER),
                   target, target_polys, mask, aerial, printed,
                   polys, seg_epes, history, cfg)


def save_summary(outdir, target, target_polys, mask, aerial, printed,
                 polys, seg_epes, history, cfg):
    """保存最终汇总图 summary.png。"""
    _draw_overview(outdir, "summary.png",
                   "迭代法 OPC 最终结果（%s，%d 轮迭代）"
                   % (cfg.DEMO_NAME, len(history)),
                   target, target_polys, mask, aerial, printed,
                   polys, seg_epes, history, cfg)
