from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GlpShape:
    kind: str
    layer: str
    values: tuple[float, ...]


def parse_glp(path, layer="M1"):
    """Read RECT/PGON geometry from the simple ICCAD GLP format."""
    shapes = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if len(fields) < 4 or fields[0] not in {"RECT", "PGON"}:
            continue
        kind, shape_layer = fields[0], fields[2]
        if shape_layer != layer:
            continue
        values = tuple(float(value) for value in fields[3:])
        if kind == "RECT" and len(values) != 4:
            raise ValueError(f"Malformed RECT in {path}: {raw_line}")
        if kind == "PGON" and (len(values) < 6 or len(values) % 2):
            raise ValueError(f"Malformed PGON in {path}: {raw_line}")
        shapes.append(GlpShape(kind, shape_layer, values))
    if not shapes:
        raise ValueError(f"No {layer} RECT/PGON shapes found in {path}")
    return shapes


def _fill_polygon_at_pixel_centers(canvas, xy):
    """Scan-convert a polygon using half-open intervals at pixel centres."""
    height, width = canvas.shape
    points = list(zip(xy[0::2], xy[1::2]))
    edges = list(zip(points, points[1:] + points[:1]))
    y_min = max(0, int(np.floor(min(y for _, y in points))))
    y_max = min(height, int(np.ceil(max(y for _, y in points))))
    for row in range(y_min, y_max):
        y = row + 0.5
        intersections = []
        for (x0, y0), (x1, y1) in edges:
            if (y0 <= y < y1) or (y1 <= y < y0):
                intersections.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
        intersections.sort()
        for left, right in zip(intersections[0::2], intersections[1::2]):
            col0 = max(0, int(np.ceil(left - 0.5)))
            col1 = min(width, int(np.ceil(right - 0.5)))
            canvas[row, col0:col1] = 1


def rasterize_glp(path, grid_size=1024, pixel_size_nm=1.0, layer="M1"):
    """Rasterize GLP geometry onto a binary lower-left-origin array."""
    shapes = parse_glp(path, layer=layer)
    canvas = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for shape in shapes:
        values = tuple(value / pixel_size_nm for value in shape.values)
        if shape.kind == "RECT":
            x, y, width, height = values
            x0 = max(0, int(np.ceil(x - 0.5)))
            x1 = min(grid_size, int(np.ceil(x + width - 0.5)))
            y0 = max(0, int(np.ceil(y - 0.5)))
            y1 = min(grid_size, int(np.ceil(y + height - 0.5)))
            canvas[y0:y1, x0:x1] = 1
        else:
            _fill_polygon_at_pixel_centers(canvas, values)
    return canvas.astype(np.float32)


def area_downsample_binary(array, factor, threshold=0.5):
    """Downsample a binary target by block coverage and return binary pixels."""
    array = np.asarray(array, dtype=np.float32)
    if factor == 1:
        return (array > threshold).astype(np.float32)
    height, width = array.shape
    if height % factor or width % factor:
        raise ValueError("Array dimensions must be divisible by factor")
    coverage = array.reshape(height // factor, factor, width // factor, factor).mean((1, 3))
    return (coverage >= threshold).astype(np.float32)


def load_target(path, grid_size, pixel_size_nm):
    """Load GLP, NPY, or grayscale PNG as a binary optimization target."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".glp":
        factor = int(round(pixel_size_nm))
        if factor < 1 or not np.isclose(pixel_size_nm, factor):
            raise ValueError("GLP coverage loading currently requires integer nm pixels")
        native = rasterize_glp(path, grid_size=grid_size * factor, pixel_size_nm=1.0)
        return area_downsample_binary(native, factor)
    if suffix == ".npy":
        target = np.load(path)
    elif suffix in {".png", ".bmp", ".tif", ".tiff"}:
        target = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported target format: {suffix}")
    if target.shape != (grid_size, grid_size):
        raise ValueError(f"Expected {(grid_size, grid_size)}, got {target.shape}")
    return (target > 0.5).astype(np.float32)
