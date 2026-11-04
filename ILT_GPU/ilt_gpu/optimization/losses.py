import torch
from torch.nn import functional as F


def boundary_weight_map(target: torch.Tensor, edge_weight: float,
                        radius_px: int = 2) -> torch.Tensor:
    data = target[None, None]
    radius_px = max(1, int(radius_px))
    kernel_size = 2 * radius_px + 1
    dilated = F.max_pool2d(data, kernel_size=kernel_size, stride=1,
                          padding=radius_px)
    eroded = -F.max_pool2d(-data, kernel_size=kernel_size, stride=1,
                           padding=radius_px)
    boundary = (dilated - eroded).clamp(0.0, 1.0)[0, 0]
    return 1.0 + edge_weight * boundary


def total_variation(mask: torch.Tensor) -> torch.Tensor:
    return ((mask[1:, :] - mask[:-1, :]).abs().mean()
            + (mask[:, 1:] - mask[:, :-1]).abs().mean())


def ilt_loss(mask, printed_soft, target, weights,
             tv_weight, binary_weight, area_weight):
    image = ((printed_soft - target).square() * weights).mean()
    tv = total_variation(mask)
    binary = (mask * (1.0 - mask)).mean()
    area = (mask.mean() - target.mean()).square()
    total = image + tv_weight * tv + binary_weight * binary + area_weight * area
    return total, {
        "image": image.detach(), "tv": tv.detach(),
        "binary": binary.detach(), "area": area.detach(),
    }

