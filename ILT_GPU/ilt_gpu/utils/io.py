import csv
from pathlib import Path

import numpy as np
from PIL import Image


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def save_binary_png(path, array):
    data = (np.asarray(array) > 0.5).astype(np.uint8) * 255
    Image.fromarray(data, mode="L").save(path)


def save_gray_png(path, array):
    data = np.clip(np.asarray(array), 0.0, 1.0)
    Image.fromarray(np.uint8(data * 255), mode="L").save(path)


def save_history(path, history):
    path = Path(path)
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

