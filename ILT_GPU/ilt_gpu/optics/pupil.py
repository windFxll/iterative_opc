import torch

from ..config import OpticsConfig
from .source import annular_source_samples


def build_shifted_pupils(config: OpticsConfig, device: torch.device):
    """Build complex64 pupils P(f + f_s) and normalized source weights."""
    n = config.fft_size
    freq = torch.fft.fftfreq(n, d=config.pixel_size_nm, device=device)
    fy, fx = torch.meshgrid(freq, freq, indexing="xy")
    cutoff = config.numerical_aperture / config.wavelength_nm

    source_xy, source_weights = annular_source_samples(
        config.sigma_in, config.sigma_out,
        config.source_rings, config.source_points_per_ring,
    )
    pupils = []
    for sx_norm, sy_norm in source_xy:
        sx = cutoff * float(sx_norm)
        sy = cutoff * float(sy_norm)
        nu = torch.sqrt((fx + sx).square() + (fy + sy).square()) / cutoff
        inside = nu <= 1.0
        if config.defocus_nm:
            phase = ((2.0 * torch.pi / config.wavelength_nm) * config.defocus_nm
                     * (torch.sqrt(torch.clamp(1.0 -
                         (config.numerical_aperture * nu).square(), min=0.0)) - 1.0))
            pupil = torch.where(inside, torch.polar(torch.ones_like(phase), phase),
                                torch.zeros_like(phase, dtype=torch.complex64))
        else:
            # With zero defocus the pupil is strictly binary.  Keeping it bool
            # avoids wasting 8x memory on complex zeros and ones.
            pupil = inside
        pupils.append(pupil)

    pupils = torch.stack(pupils, dim=0)
    weights = torch.as_tensor(source_weights, device=device)
    if pupils.is_complex():
        dc_power = pupils[:, 0, 0].real.square() + pupils[:, 0, 0].imag.square()
    else:
        dc_power = pupils[:, 0, 0].to(torch.float32)
    clear_intensity = torch.sum(weights * dc_power).clamp_min(1e-12)
    return pupils, weights, clear_intensity

