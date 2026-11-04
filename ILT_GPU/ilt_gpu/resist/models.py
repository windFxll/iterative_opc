import math

import torch
from torch.nn import functional as F


def soft_resist(aerial: torch.Tensor, threshold: float, beta: float) -> torch.Tensor:
    """Differentiable dark-feature resist transfer."""
    return torch.sigmoid(beta * (threshold - aerial))


def hard_resist(aerial: torch.Tensor, threshold: float) -> torch.Tensor:
    return (aerial < threshold).to(aerial.dtype)


def gaussian_blur2d(image: torch.Tensor, sigma_px: float,
                    circular: bool = False) -> torch.Tensor:
    if sigma_px <= 0:
        return image
    radius = max(1, int(math.ceil(3.0 * sigma_px)))
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (x / sigma_px).square())
    kernel = kernel / kernel.sum()
    data = image[None, None]
    mode = "circular" if circular else "replicate"
    data = F.pad(data, (radius, radius, 0, 0), mode=mode)
    data = F.conv2d(data, kernel.view(1, 1, 1, -1))
    data = F.pad(data, (0, 0, radius, radius), mode=mode)
    data = F.conv2d(data, kernel.view(1, 1, -1, 1))
    return data[0, 0]

