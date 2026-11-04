import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ilt_gpu.config import OpticsConfig
from ilt_gpu.layouts import LAYOUTS
from ilt_gpu.utils.io import save_binary_png


def main():
    parser = argparse.ArgumentParser(description="Generate an ILT target without optimizing it")
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="complex")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output" / "complex_layout_preview")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = OpticsConfig()
    target = LAYOUTS[args.layout](cfg.grid_size, cfg.pixel_size_nm)
    save_binary_png(args.output / "target.png", target)
    np.save(args.output / "target.npy", target)

    half = cfg.grid_size * cfg.pixel_size_nm / 2.0
    fig, ax = plt.subplots(figsize=(8.6, 8.2), constrained_layout=True)
    ax.imshow(target, origin="lower", extent=(-half, half, -half, half),
              cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Complex ILT target: serpentine interconnects and dense lines")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.text(0.02, 0.02,
            "grid 256×256 | pixel 4 nm | field 1.024 µm\n"
            "trace CD 48/56/64 nm | dense L/S 40/32 nm",
            transform=ax.transAxes, color="tab:blue", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.7"})
    fig.savefig(args.output / "target_overview.png", dpi=180)
    plt.close(fig)

    fill = float(target.mean())
    print(f"layout={args.layout} shape={target.shape} pixel={cfg.pixel_size_nm:.1f}nm "
          f"field={2 * half:.1f}nm fill={fill:.4f}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
