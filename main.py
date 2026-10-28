# -*- coding: utf-8 -*-
"""
main.py —— 迭代法 OPC 模型入口。

用法:
    conda activate env
    python main.py            # 跑默认示例（config.py 里 DEMO_NAME 指定的图形）
    python main.py --selftest # 先跑光学模型自检（5 项物理检查），再跑 OPC

所有参数都在 config.py 里调整（光刻参数、图形尺寸、迭代设置等），
本文件不需要改动。
"""
import os
import sys

import numpy as np

import config
import geometry
from mask_ops import polygons_to_pixels
from opc_engine import run_opc
from optics import self_check
from visualize import setup_style, save_summary


def main():
    setup_style(config)
    os.makedirs(config.OUT_DIR, exist_ok=True)

    # ---- 可选：光学模型自检 ----
    if "--selftest" in sys.argv:
        print("=== 光学模型自检（Hopkins/Abbe 前向） ===")
        if not self_check(config):
            print("自检未通过，请先检查 optics.py 或参数设置")
            return

    # ---- 1. 生成目标版图 ----
    print("=== 迭代法 OPC 演示: %s ===" % config.DEMO_NAME)
    target_pixels, target_polys = geometry.make_target(config)
    print("目标图形: %d 个多边形, %d 个像素 (视场 %.0f nm, 像素 %.1f nm)"
          % (len(target_polys), int(target_pixels.sum()),
             config.GRID_N * config.PIXEL_SIZE_NM, config.PIXEL_SIZE_NM))
    print("光刻参数: λ=%.0f nm, NA=%.2f, %s 照明 σ=%.1f~%.1f, 阈值 %.2f%s"
          % (config.WAVELENGTH_NM, config.NA, config.ILLUM_TYPE,
             config.SIGMA_IN, config.SIGMA_OUT, config.RESIST_THRESHOLD,
             ", 离焦 %.0f nm" % config.DEFOCUS_NM
             if config.DEFOCUS_NM else ""))

    # ---- 2. 迭代 OPC ----
    result = run_opc(target_pixels, target_polys, config)

    # ---- 3. 输出 ----
    mask, polys = result["mask"], result["polys"]
    aerial, printed = result["aerial"], result["printed"]
    history = result["history"]

    save_summary(config.OUT_DIR, target_pixels, target_polys, mask, aerial,
                 printed, polys, result["seg_epes"], history, config)

    # 最终掩模像素存成 0/1 文本（可作为自定义版图输入再读回）
    np.savetxt(os.path.join(config.OUT_DIR, "final_mask.txt"),
               mask.astype(int), fmt="%d")

    with open(os.path.join(config.OUT_DIR, "history.txt"), "w",
              encoding="utf-8") as f:
        f.write("# iter  rms_epe(nm)  max_abs_epe(nm)\n")
        for i, (rms, mx) in enumerate(history):
            f.write("%4d  %.4f  %.4f\n" % (i, rms, mx))

    # 可选：GDS 导出（掩模多边形，单位换算 nm→μm）
    try:
        geometry.export_gds([np.array([s.p1 for s in poly]) for poly in polys],
                            os.path.join(config.OUT_DIR, "final_mask.gds"))
        print("GDS 导出: output/final_mask.gds")
    except ImportError:
        pass

    # 可选：迭代动画 GIF
    if config.SAVE_GIF:
        try:
            import imageio
            frames = [imageio.imread(os.path.join(config.OUT_DIR, f))
                      for f in sorted(os.listdir(config.OUT_DIR))
                      if f.startswith("iter_") and f.endswith(".png")]
            if frames:
                imageio.mimsave(os.path.join(config.OUT_DIR, "opc.gif"),
                                frames, duration=0.4)
                print("动画导出: output/opc.gif")
        except ImportError:
            pass

    # ---- 4. 报告 ----
    print("\n=== 结果报告 ===")
    if history:
        rms0, mx0 = history[0]
        print("迭代轮数: %d" % len(history))
        print("RMS EPE:  %.2f nm -> %.2f nm（末轮）"
              % (rms0, history[-1][0]))
        print("max|EPE|:  %.2f nm -> %.2f nm"
              % (mx0, history[-1][1]))
    # 印刷轮廓与目标的对称平均偏差（初始掩模 vs 输出掩模）——最直观的
    # 质量指标（与 run_opc 内部"历史最佳"判据同口径，可直接比较）
    from optics import precompute_source_and_pupil, aerial_image, apply_resist_blur
    from opc_engine import _contour_dev
    try:
        pre = precompute_source_and_pupil(config)
        printed0 = (apply_resist_blur(aerial_image(target_pixels, config, pre),
                                      config) < config.RESIST_THRESHOLD)
        dev0 = _contour_dev(printed0.astype(np.uint8), target_polys, config)
        print("印刷轮廓平均偏差: %.1f nm -> %.1f nm"
              % (dev0, result.get("best_dev", dev0)))
    except Exception:
        pass
    print("输出目录: %s/  (summary.png 汇总图, iter_*.png 逐轮帧, "
          "history.txt 收敛数据, final_mask.txt 最终掩模)" % config.OUT_DIR)
    print("提示: 改 config.py 的 DEMO_NAME 可切换示例图形；")
    print("      入门顺序建议 elbow_array（周期+角点）→ dense_lines → "
          "contact_array → line_end_array。")


if __name__ == "__main__":
    main()
