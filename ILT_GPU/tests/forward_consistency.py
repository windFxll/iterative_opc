"""Compare the GPU operator against the parent NumPy reference implementation."""
import argparse
from pathlib import Path
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PARENT))

from ilt_gpu.config import OpticsConfig
from ilt_gpu.optics import AbbeImaging
from optics import aerial_image, precompute_source_and_pupil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    n = 96
    rng = np.random.default_rng(7)
    mask = (rng.random((n, n)) > 0.82).astype(np.float64)
    numpy_cfg = types.SimpleNamespace(
        GRID_N=n, PIXEL_SIZE_NM=4.0, WAVELENGTH_NM=193.0, NA=1.35,
        ILLUM_TYPE="annular", SIGMA_IN=0.6, SIGMA_OUT=0.9,
        N_SOURCE_RINGS=2, N_SOURCE_PTS=8, DEFOCUS_NM=0.0,
        BOUNDARY_MODE="isolated",
    )
    reference = aerial_image(mask, numpy_cfg,
                             precompute_source_and_pupil(numpy_cfg))

    torch_cfg = OpticsConfig(
        grid_size=n, pixel_size_nm=4.0,
        source_rings=2, source_points_per_ring=8, source_batch_size=4,
    )
    model = AbbeImaging(torch_cfg, device)
    output = model(torch.as_tensor(mask, dtype=torch.float32,
                                   device=device)).detach().cpu().numpy()
    rms = float(np.sqrt(np.mean((reference - output) ** 2)))
    maximum = float(np.max(np.abs(reference - output)))
    assert rms < 1e-6 and maximum < 2e-6
    print(f"PASS NumPy/GPU forward RMS={rms:.3e}, max={maximum:.3e}")


if __name__ == "__main__":
    main()

