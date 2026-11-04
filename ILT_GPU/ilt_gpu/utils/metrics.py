import numpy as np
import torch
from scipy import ndimage


def binary_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    aa, bb = a > 0.5, b > 0.5
    union = torch.logical_or(aa, bb).sum().clamp_min(1)
    return float(torch.logical_and(aa, bb).sum() / union)


def contour_mean_distance_nm(a, b, pixel_size_nm: float) -> float:
    a = np.asarray(a) > 0.5
    b = np.asarray(b) > 0.5
    a_edge = np.logical_xor(a, ndimage.binary_erosion(a))
    b_edge = np.logical_xor(b, ndimage.binary_erosion(b))
    if not a_edge.any() or not b_edge.any():
        return float("inf")
    d_to_b = ndimage.distance_transform_edt(~b_edge)
    d_to_a = ndimage.distance_transform_edt(~a_edge)
    return float(0.5 * (d_to_b[a_edge].mean() + d_to_a[b_edge].mean())
                 * pixel_size_nm)

