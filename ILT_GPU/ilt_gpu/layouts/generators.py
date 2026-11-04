import numpy as np


def _nm_to_index(value_nm: float, grid_size: int, pixel_size_nm: float) -> int:
    return int(round(value_nm / pixel_size_nm + (grid_size - 1) / 2.0))


def _rect(mask, x0, y0, x1, y1, pixel_size_nm):
    n = mask.shape[0]
    j0 = max(0, _nm_to_index(min(x0, x1), n, pixel_size_nm))
    j1 = min(n, _nm_to_index(max(x0, x1), n, pixel_size_nm) + 1)
    i0 = max(0, _nm_to_index(min(y0, y1), n, pixel_size_nm))
    i1 = min(n, _nm_to_index(max(y0, y1), n, pixel_size_nm) + 1)
    mask[i0:i1, j0:j1] = 1.0


def _manhattan_trace(mask, points, width_nm, pixel_size_nm):
    """Rasterize an axis-aligned polyline with square, gap-free corners."""
    half_width = width_nm / 2.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if x0 == x1:
            _rect(mask, x0 - half_width, min(y0, y1) - half_width,
                  x0 + half_width, max(y0, y1) + half_width, pixel_size_nm)
        elif y0 == y1:
            _rect(mask, min(x0, x1) - half_width, y0 - half_width,
                  max(x0, x1) + half_width, y0 + half_width, pixel_size_nm)
        else:
            raise ValueError("Manhattan traces require horizontal/vertical segments")


def _square(mask, x, y, size_nm, pixel_size_nm):
    half = size_nm / 2.0
    _rect(mask, x - half, y - half, x + half, y + half, pixel_size_nm)


def realistic_interconnect(grid_size: int, pixel_size_nm: float) -> np.ndarray:
    """Three isolated Manhattan nets with pads, bends, and landing regions."""
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    trace, pad, land = 96.0, 120.0, 112.0
    xpad, ypad = 350.0, 320.0
    hw, hp, hl = trace / 2, pad / 2, land / 2

    def rect(x0, y0, x1, y1):
        _rect(mask, x0, y0, x1, y1, pixel_size_nm)

    def pad_at(x, y): rect(x - hp, y - hp, x + hp, y + hp)
    def land_at(x, y): rect(x - hl, y - hl, x + hl, y + hl)
    def hseg(x0, x1, y): rect(min(x0, x1), y - hw, max(x0, x1), y + hw)
    def vseg(x, y0, y1): rect(x - hw, min(y0, y1), x + hw, max(y0, y1))

    for sign in (-1.0, 1.0):
        y_outer, y_inner, xb = sign * ypad, sign * 240.0, 190.0
        pad_at(-xpad, y_outer); pad_at(xpad, y_outer)
        hseg(-xpad, -xb, y_outer); vseg(-xb, y_outer, y_inner)
        hseg(-xb, xb, y_inner); vseg(xb, y_inner, y_outer)
        hseg(xb, xpad, y_outer); land_at(-xb, y_inner); land_at(xb, y_inner)

    xb, yb = 95.0, 70.0
    pad_at(-xpad, 0.0); pad_at(xpad, 0.0)
    hseg(-xpad, -xb, 0.0); vseg(-xb, 0.0, yb)
    hseg(-xb, xb, yb); vseg(xb, 0.0, yb); hseg(xb, xpad, 0.0)
    land_at(-xb, yb); land_at(xb, yb)
    return mask


def complex_interconnect(grid_size: int, pixel_size_nm: float) -> np.ndarray:
    """Stress layout: three serpentine nets plus a 40/32 nm line-space grating.

    Coordinates are physical nanometres.  The many concave/convex corners probe
    corner rounding, while the lower-right grating probes dense-pattern imaging.
    """
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)

    upper = [
        (-440, 340), (-330, 340), (-330, 205), (-170, 205),
        (-170, 350), (-20, 350), (-20, 235), (135, 235),
        (135, 365), (285, 365), (285, 245), (435, 245),
    ]
    middle = [
        (-445, 40), (-355, 40), (-355, -85), (-215, -85),
        (-215, 85), (-75, 85), (-75, -45), (65, -45),
        (65, 105), (205, 105), (205, -35), (345, -35),
        (345, 75), (445, 75),
    ]
    lower = [
        (-440, -250), (-350, -250), (-350, -390), (-220, -390),
        (-220, -225), (-90, -225), (-90, -365), (40, -365),
        (40, -245), (140, -245),
    ]

    for points, width in ((upper, 64.0), (middle, 56.0), (lower, 48.0)):
        _manhattan_trace(mask, points, width, pixel_size_nm)
        _square(mask, *points[0], 96.0, pixel_size_nm)
        _square(mask, *points[-1], 96.0, pixel_size_nm)

    # Dense vertical line-space array: nominal CD=40 nm, space=32 nm.
    for x in (220.0, 292.0, 364.0, 436.0):
        _rect(mask, x - 20.0, -430.0, x + 20.0, -210.0, pixel_size_nm)

    # Small staggered landing array adds compact 2-D features without touching
    # the neighboring nets.
    for x, y in ((105.0, -105.0), (165.0, -145.0), (225.0, -105.0)):
        _square(mask, x, y, 48.0, pixel_size_nm)

    return mask

