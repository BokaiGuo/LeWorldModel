"""Declared RGB-only color masks and margins for this PushT renderer."""
import numpy as np


def masks(pixels):
    pixels = np.asarray(pixels)
    if pixels.shape != (224, 224, 3) or pixels.dtype != np.uint8:
        raise ValueError("uint8 RGB224 required")
    r, g, b = pixels.astype(np.int16).transpose(2, 0, 1)
    block = (g-r > 5) & (b-g > 5) & (b-r < 80) & (r > 70) & (b < 235)
    pusher = (b-r > 100) & (b-g > 60) & (b > 200)
    return {"object": block, "pusher": pusher}


def inspect(pixels, margin=8):
    if type(margin) is not int or not 0 <= margin < 112:
        raise ValueError("margin")
    result = {}
    for name, mask in masks(pixels).items():
        y, x = np.nonzero(mask)
        box = None if len(x) == 0 else [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
        result[name] = dict(pixels=int(len(x)), bbox=box,
                           interior=box is not None and min(box[:2]) >= margin
                           and max(box[2:]) < 224-margin)
    result["eligible"] = all(result[k]["interior"] for k in ("object", "pusher"))
    result["margin_pixels"] = margin
    result["scope"] = "renderer_specific_RGB_screen_not_occlusion_or_shape_completeness_certificate"
    return result
