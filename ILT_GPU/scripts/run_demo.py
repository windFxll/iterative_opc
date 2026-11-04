import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ilt_gpu.layouts import LAYOUTS, load_target
from ilt_gpu.optics import AbbeImaging
from ilt_gpu.optimization import PixelILT
from ilt_gpu.profiles import make_run_profile
from ilt_gpu.resist import hard_resist
from ilt_gpu.utils.io import (save_binary_png, save_gray_png, save_history,
                              to_numpy)
from ilt_gpu.utils.metrics import binary_iou, contour_mean_distance_nm
from ilt_gpu.utils.visualization import save_demo_summary


def main():
    parser = argparse.ArgumentParser(description="Differentiable GPU ILT demo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--profile", choices=("demo", "iccad-native"),
                        default="demo")
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="realistic")
    parser.add_argument("--target", type=Path, default=None,
                        help="GLP/NPY/PNG target; overrides --layout")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        if args.target is not None:
            args.output = ROOT / "output" / f"{args.target.stem}_{args.profile}"
        else:
            args.output = ROOT / "output" / (
                "demo" if args.layout == "realistic" else "complex"
            )
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        print(f"GPU: {torch.cuda.get_device_name(device)} capability="
              f"{torch.cuda.get_device_capability(device)}")

    profile = make_run_profile(args.profile, args.iterations)
    optics_cfg = profile.optics
    opt_cfg = profile.optimization
    target_name = args.target.stem if args.target is not None else args.layout
    target_np = (load_target(args.target, optics_cfg.grid_size, optics_cfg.pixel_size_nm)
                 if args.target is not None
                 else LAYOUTS[args.layout](optics_cfg.grid_size, optics_cfg.pixel_size_nm))
    target = torch.as_tensor(target_np, device=device)

    print(f"Optimize: {optics_cfg.grid_size}x{optics_cfg.grid_size} @ "
          f"{optics_cfg.pixel_size_nm:.1f} nm, {optics_cfg.source_count} source points, "
          f"{optics_cfg.boundary_mode}")
    imaging = AbbeImaging(optics_cfg, device)
    with torch.no_grad():
        initial_mask = target.clone()
        initial_aerial_opt = imaging(initial_mask)
        initial_printed_opt = hard_resist(initial_aerial_opt, optics_cfg.resist_threshold)

    result = PixelILT(imaging, opt_cfg).optimize(target)

    # Verification uses the denser sampling selected by the run profile.
    verify_cfg = profile.verification
    print(f"Verify: {verify_cfg.source_count} source points")
    del imaging
    if device.type == "cuda":
        torch.cuda.empty_cache()
    verify_start = time.perf_counter()
    verifier = AbbeImaging(verify_cfg, device)
    with torch.no_grad():
        initial_aerial = verifier(initial_mask)
        initial_printed = hard_resist(initial_aerial, verify_cfg.resist_threshold)
        optimized_aerial = verifier(result.optimized_mask)
        optimized_printed = hard_resist(optimized_aerial, verify_cfg.resist_threshold)
    verify_seconds = time.perf_counter() - verify_start

    arrays = {
        "target": target_np,
        "initial_mask": to_numpy(initial_mask),
        "initial_aerial": to_numpy(initial_aerial),
        "initial_printed": to_numpy(initial_printed),
        "optimized_mask": to_numpy(result.optimized_mask),
        "optimized_aerial": to_numpy(optimized_aerial),
        "optimized_printed": to_numpy(optimized_printed),
    }
    initial_iou = binary_iou(initial_printed, target)
    optimized_iou = binary_iou(optimized_printed, target)
    initial_contour = contour_mean_distance_nm(arrays["initial_printed"], target_np,
                                                optics_cfg.pixel_size_nm)
    optimized_contour = contour_mean_distance_nm(arrays["optimized_printed"], target_np,
                                                  optics_cfg.pixel_size_nm)

    save_binary_png(args.output / "01_target.png", arrays["target"])
    save_binary_png(args.output / "02_initial_printed.png", arrays["initial_printed"])
    save_binary_png(args.output / "03_optimized_mask.png", arrays["optimized_mask"])
    save_gray_png(args.output / "04_optimized_aerial.png", arrays["optimized_aerial"])
    save_binary_png(args.output / "05_optimized_printed.png", arrays["optimized_printed"])
    np.save(args.output / "optimized_mask.npy", arrays["optimized_mask"])
    np.savetxt(args.output / "optimized_mask.txt", arrays["optimized_mask"], fmt="%d")
    save_history(args.output / "history.csv", result.history)
    save_demo_summary(args.output / "summary.png", history=result.history,
                      pixel_size_nm=optics_cfg.pixel_size_nm, **arrays)

    peak_mb = (torch.cuda.max_memory_allocated(device) / 2**20
               if device.type == "cuda" else 0.0)
    report = (
        f"device={device}\n"
        f"profile={profile.name}\n"
        f"target={target_name}\n"
        f"grid_size={optics_cfg.grid_size}\n"
        f"pixel_size_nm={optics_cfg.pixel_size_nm}\n"
        f"gpu={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}\n"
        f"optimization_seconds={result.elapsed_seconds:.3f}\n"
        f"verification_seconds={verify_seconds:.3f}\n"
        f"peak_gpu_memory_mb={peak_mb:.1f}\n"
        f"best_step={result.best_step}\n"
        f"steps_completed={result.steps_completed}\n"
        f"stop_reason={result.stop_reason}\n"
        f"initial_iou={initial_iou:.6f}\n"
        f"optimized_iou={optimized_iou:.6f}\n"
        f"initial_contour_mean_nm={initial_contour:.4f}\n"
        f"optimized_contour_mean_nm={optimized_contour:.4f}\n"
    )
    (args.output / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
