import torch
from torch import nn
from torch.nn import functional as F

from ..config import OpticsConfig
from ..resist.models import gaussian_blur2d
from .pupil import build_shifted_pupils


class AbbeImaging(nn.Module):
    """Differentiable partially coherent Abbe imaging operator.

    Input mask convention: 1 = opaque dark feature, 0 = clear field.
    """

    def __init__(self, config: OpticsConfig, device: torch.device | str):
        super().__init__()
        self.config = config
        device = torch.device(device)
        pupils, weights, clear_intensity = build_shifted_pupils(config, device)
        self.register_buffer("pupils", pupils, persistent=False)
        self.register_buffer("weights", weights, persistent=False)
        self.register_buffer("clear_intensity", clear_intensity, persistent=False)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 2 or mask.shape != (self.config.grid_size, self.config.grid_size):
            raise ValueError(f"expected mask [{self.config.grid_size}, {self.config.grid_size}]")

        transmission = 1.0 - mask.to(dtype=torch.float32)
        n = self.config.grid_size
        n_fft = self.config.fft_size
        if n_fft != n:
            total = n_fft - n
            left = total // 2
            right = total - left
            transmission = F.pad(transmission[None, None],
                                 (left, right, left, right), value=1.0)[0, 0]

        spectrum = torch.fft.fft2(torch.fft.ifftshift(transmission))
        intensity = torch.zeros((n_fft, n_fft), device=mask.device, dtype=torch.float32)
        batch = max(1, self.config.source_batch_size)
        for start in range(0, self.pupils.shape[0], batch):
            stop = min(start + batch, self.pupils.shape[0])
            field = torch.fft.fftshift(
                torch.fft.ifft2(spectrum[None] * self.pupils[start:stop]),
                dim=(-2, -1),
            )
            power = field.real.square() + field.imag.square()
            intensity = intensity + torch.sum(
                self.weights[start:stop, None, None] * power, dim=0)

        if n_fft != n:
            offset = (n_fft - n) // 2
            intensity = intensity[offset:offset + n, offset:offset + n]
        intensity = intensity / self.clear_intensity

        sigma_px = self.config.resist_blur_sigma_nm / self.config.pixel_size_nm
        if sigma_px > 0:
            intensity = gaussian_blur2d(
                intensity, sigma_px,
                circular=self.config.boundary_mode == "periodic",
            )
        return intensity

