import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ilt_gpu.layouts import area_downsample_binary, parse_glp, rasterize_glp
from ilt_gpu.utils.io import save_binary_png


def main():
    parser = argparse.ArgumentParser(description="Rasterize an ICCAD GLP benchmark")
    parser.add_argument("glp", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-size", type=int, default=1024)
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    shapes = parse_glp(args.glp)
    native = rasterize_glp(args.glp, grid_size=args.native_size, pixel_size_nm=1.0)
    working = area_downsample_binary(native, args.factor)

    stem = args.glp.stem
    np.save(args.output / f"{stem}_1024x1024_1nm.npy", native)
    np.save(args.output / f"{stem}_256x256_4nm.npy", working)
    save_binary_png(args.output / f"{stem}_1024x1024_1nm.png", native)
    save_binary_png(args.output / f"{stem}_256x256_4nm.png", working)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), constrained_layout=True)
    for ax, image, title in (
        (axes[0], native, "Native ICCAD target (1024², 1 nm/px)"),
        (axes[1], working, "ILT_GPU target (256², 4 nm/px)"),
    ):
        ax.imshow(image, origin="lower", extent=(0, 1024, 0, 1024),
                  cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_aspect("equal")
    fig.suptitle(f"ICCAD-2013 {stem}: {len(shapes)} M1 shapes")
    fig.savefig(args.output / f"{stem}_comparison.png", dpi=180)
    plt.close(fig)

    native_area = int(native.sum())
    represented_area = int(working.sum() * args.factor**2)
    print(f"GLP={args.glp.resolve()}")
    print(f"shapes={len(shapes)} native_area_nm2={native_area} "
          f"working_area_nm2={represented_area} area_error={represented_area-native_area}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
