import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ilt_gpu.config import OpticsConfig
from ilt_gpu.optics import AbbeImaging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    cfg = OpticsConfig(grid_size=96, pixel_size_nm=4.0,
                       source_rings=2, source_points_per_ring=8,
                       source_batch_size=4)
    model = AbbeImaging(cfg, device)
    mask = torch.zeros((cfg.grid_size, cfg.grid_size), device=device, requires_grad=True)
    mask.data[36:60, 28:68] = 1.0
    aerial = model(mask)
    loss = aerial.square().mean()
    loss.backward()
    assert torch.isfinite(aerial).all()
    assert mask.grad is not None and torch.isfinite(mask.grad).all()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"PASS device={device} name={device_name} "
          f"aerial=[{float(aerial.min()):.4f},{float(aerial.max()):.4f}] "
          f"grad_mean={float(mask.grad.abs().mean()):.3e}")


if __name__ == "__main__":
    main()

