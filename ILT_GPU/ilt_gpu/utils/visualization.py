from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_demo_summary(path, target, initial_mask, initial_aerial, initial_printed,
                      optimized_mask, optimized_aerial, optimized_printed, history,
                      pixel_size_nm):
    path = Path(path)
    half = target.shape[0] * pixel_size_nm / 2.0
    extent = (-half, half, -half, half)
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(1, 1, 0.75))
    panels = [
        (target, "Target", "gray_r", 0, 1),
        (initial_mask, "Initial mask", "gray_r", 0, 1),
        (initial_aerial, "Initial aerial image", "inferno", 0, 1),
        (initial_printed, "Initial hard resist", "gray_r", 0, 1),
        (target, "Target (reference)", "gray_r", 0, 1),
        (optimized_mask, "GPU ILT mask", "gray_r", 0, 1),
        (optimized_aerial, "Verified aerial image", "inferno", 0, 1),
        (optimized_printed, "Verified hard resist", "gray_r", 0, 1),
    ]
    for idx, (image, title, cmap, vmin, vmax) in enumerate(panels):
        ax = fig.add_subplot(grid[idx // 4, idx % 4])
        ax.imshow(image, origin="lower", extent=extent, cmap=cmap,
                  vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x (nm)")
        if idx % 4 == 0:
            ax.set_ylabel("y (nm)")

    ax_loss = fig.add_subplot(grid[2, :2])
    steps = [row["step"] for row in history]
    ax_loss.plot(steps, [row["loss"] for row in history], marker="o")
    ax_loss.set_title("Differentiable optimization loss")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(alpha=0.25)

    ax_iou = fig.add_subplot(grid[2, 2:])
    ax_iou.plot(steps, [row["iou"] for row in history], marker="o", color="tab:green")
    ax_iou.set_title("Hard-mask validation IoU")
    ax_iou.set_xlabel("step")
    ax_iou.set_ylabel("IoU")
    ax_iou.set_ylim(0, 1)
    ax_iou.grid(alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)

