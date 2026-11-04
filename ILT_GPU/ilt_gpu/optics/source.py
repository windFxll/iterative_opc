import numpy as np


def annular_source_samples(sigma_in: float, sigma_out: float,
                            rings: int, points_per_ring: int):
    """Return normalized source coordinates and area-correct radial weights."""
    if rings <= 0 or points_per_ring <= 0:
        raise ValueError("source rings and points_per_ring must be positive")
    if points_per_ring % 2:
        raise ValueError("points_per_ring must be even to preserve +/- symmetry")
    if not 0.0 <= sigma_in < sigma_out:
        raise ValueError("expected 0 <= sigma_in < sigma_out")

    ds = (sigma_out - sigma_in) / rings
    radii = sigma_in + (np.arange(rings, dtype=np.float64) + 0.5) * ds
    angles = 2.0 * np.pi * np.arange(points_per_ring) / points_per_ring
    xy, weights = [], []
    for radius in radii:
        for angle in angles:
            xy.append((radius * np.cos(angle), radius * np.sin(angle)))
            weights.append(radius)
    weights = np.asarray(weights, dtype=np.float32)
    weights /= weights.sum()
    return np.asarray(xy, dtype=np.float32), weights

