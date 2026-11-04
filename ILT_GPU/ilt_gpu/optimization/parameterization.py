import torch
from torch import nn

from ..resist.models import gaussian_blur2d


class FilteredMask(nn.Module):
    """Trainable mask logits followed by a manufacturing low-pass filter."""

    def __init__(self, target: torch.Tensor, filter_sigma_px: float,
                 border_margin_px: int = 0):
        super().__init__()
        initial = torch.where(target > 0.5, torch.full_like(target, 1.5),
                              torch.full_like(target, -1.5))
        self.logits = nn.Parameter(initial)
        self.filter_sigma_px = filter_sigma_px
        window = torch.ones_like(target)
        if border_margin_px > 0:
            margin = int(border_margin_px)
            window[:margin, :] = 0.0
            window[-margin:, :] = 0.0
            window[:, :margin] = 0.0
            window[:, -margin:] = 0.0
        self.register_buffer("design_window", window)

    def forward(self, beta: float) -> torch.Tensor:
        filtered = gaussian_blur2d(self.logits, self.filter_sigma_px)
        return torch.sigmoid(beta * filtered) * self.design_window

