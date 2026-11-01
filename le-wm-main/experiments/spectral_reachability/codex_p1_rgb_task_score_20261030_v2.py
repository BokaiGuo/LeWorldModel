"""Image-only task features for the fixed PushT renderer; no simulator access.

Inputs are current RGB and task goal RGB, not simulator pose/goal state. The
separate object and pusher features are not a native-success classifier or a
calibrated scalar objective. Exact-color masks are renderer-specific and can be
biased by occlusion even when the visibility checks pass.
"""
import hashlib
import math

import numpy as np
from scipy.ndimage import distance_transform_edt, label

BLUE = (65, 105, 225)
BLOCK = (119, 136, 153)


def _image(value):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.shape != (224, 224, 3):
        raise ValueError("native uint8 RGB224 required")
    return value


def _mask(image, color, minimum, *, single):
    mask = np.all(image == color, axis=-1)
    y, x = np.nonzero(mask)
    if len(x) < minimum or x.min() == 0 or x.max() == 223 or y.min() == 0 or y.max() == 223:
        raise ValueError("mask absent, too small, or clipped")
    _, components = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if single and components != 1:
        raise ValueError("pusher mask fragmented")
    return mask, {"count": int(len(x)), "centroid_xy": [float(x.mean()), float(y.mean())],
                  "bounds_xyxy": [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
                  "components": int(components)}


def score(current_rgb, goal_rgb):
    """Return RGB-only measurements; refuse absent/clipped/fragmented markers."""
    current, goal = _image(current_rgb), _image(goal_rgb)
    cm, cb = _mask(current, BLUE, 8, single=True)
    gm, gb = _mask(goal, BLUE, 8, single=True)
    co, cob = _mask(current, BLOCK, 8, single=False)
    go, gob = _mask(goal, BLOCK, 8, single=False)
    pusher = math.dist(cb["centroid_xy"], gb["centroid_xy"])
    # Symmetric mean nearest-mask-pixel distance retains shape/orientation
    # information but is not an object-pose estimate.
    object_shape = (float(distance_transform_edt(~go)[co].mean())
                    + float(distance_transform_edt(~co)[go].mean())) / 2
    pixel_mse = float(np.mean((current.astype(np.float32)-goal.astype(np.float32))**2))
    return {"pusher_centroid_distance_pixels": pusher, "object_shape_chamfer_pixels": object_shape,
            "rgb_pixel_mse": pixel_mse, "current_pusher": cb, "goal_pusher": gb,
            "current_object": cob, "goal_object": gob,
            "current_pixel_sha256": hashlib.sha256(current.tobytes()).hexdigest(),
            "goal_pixel_sha256": hashlib.sha256(goal.tobytes()).hexdigest(),
            "native_success": None, "occlusion_free_certified": False}
