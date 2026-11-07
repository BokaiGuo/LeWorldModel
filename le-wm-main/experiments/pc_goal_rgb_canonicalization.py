"""Deterministic goal-RGB-only image translation for the declared PointMaze view."""
import numpy as np


def goal_shift(goal):
    image = np.asarray(goal)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("requires 224x224 uint8 RGB")
    rgb = image.astype(np.int16)
    red, green, blue = (rgb[..., i] for i in range(3))
    mask = (red > 150) & (green > 80) & (10 * red > 12 * green) & (10 * red > 14 * blue)
    y, x = np.where(mask)
    if len(x) < 1000:
        raise ValueError("static maze boundary missing")
    box = (int(x.min()), int(y.min()), int(x.max()), int(y.max()))
    width, height = box[2] - box[0] + 1, box[3] - box[1] + 1
    if not (80 <= width <= 180 and 80 <= height <= 180):
        raise ValueError("unexpected maze boundary dimensions")
    dx = int(np.rint(111.5 - (box[0] + box[2]) / 2))
    dy = int(np.rint(125.5 - (box[1] + box[3]) / 2))
    if max(abs(dx), abs(dy)) > 20:
        raise ValueError("unsupported camera displacement")
    return {"dx": dx, "dy": dy, "bbox": box, "mask_pixels": int(len(x))}


def translate(image, dx, dy):
    value = np.asarray(image)
    if value.shape != (224, 224, 3) or value.dtype != np.uint8 or not isinstance(dx, int) or not isinstance(dy, int):
        raise ValueError("expected RGB image and integer shift")
    if max(abs(dx), abs(dy)) > 20:
        raise ValueError("unsupported translation")
    # Nearest-edge extension is explicit; the valid mask marks observed pixels.
    y = np.arange(224) - dy
    x = np.arange(224) - dx
    valid = ((y >= 0) & (y < 224))[:, None] & ((x >= 0) & (x < 224))[None, :]
    output = value[np.clip(y, 0, 223)[:, None], np.clip(x, 0, 223)[None, :]]
    return output, valid
