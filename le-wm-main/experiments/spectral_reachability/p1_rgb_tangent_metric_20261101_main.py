"""Goal-RGB geometric tangent metric, without simulator state or fitting."""
import numpy as np
from PIL import Image

from . import p1_rgb_visibility_20261101_main as visibility

AXES = ("object_x", "object_y", "object_angle", "pusher_x", "pusher_y")


def perturb(pixels, axis, amount):
    if axis not in AXES or not np.isfinite(amount) or abs(amount) > 1:
        raise ValueError("fixed geometric perturbation")
    if not visibility.inspect(pixels)["eligible"]:
        raise ValueError("goal boundary visibility")
    name = "object" if axis.startswith("object") else "pusher"
    mask = visibility.masks(pixels)[name]
    y, x = np.nonzero(mask)
    rgba = np.concatenate((pixels, (mask * 255).astype(np.uint8)[..., None]), axis=-1)
    layer = Image.fromarray(rgba, "RGBA")
    if axis == "object_angle":
        layer = layer.rotate(5.0 * amount, resample=Image.Resampling.BICUBIC,
                             center=(float(x.mean()), float(y.mean())))
    else:
        dx = 4.0 * amount if axis.endswith("_x") else 0.0
        dy = 4.0 * amount if axis.endswith("_y") else 0.0
        layer = layer.transform((224,224), Image.Transform.AFFINE, (1,0,-dx,0,1,-dy),
                                resample=Image.Resampling.BICUBIC)
    background = pixels.copy()
    background[mask] = 255
    return np.asarray(Image.alpha_composite(Image.fromarray(background).convert("RGBA"), layer)
                      .convert("RGB")).copy()


def construct(jacobian):
    jacobian = np.asarray(jacobian, np.float64)
    if jacobian.shape != (192,5) or not np.isfinite(jacobian).all():
        raise ValueError("finite 192x5 tangent")
    u, s, _ = np.linalg.svd(jacobian, full_matrices=False)
    if s[-1] <= 0 or s[0] / s[-1] > 1e6:
        raise ValueError("unresolved geometric directions")
    ridge = .01 * s[0] ** 2
    gains = s / (s*s + ridge)
    # Weak residual penalty retains sensitivity outside the five local directions.
    q = (u * gains[None,:]**2) @ u.T + (.01 / s[0]**2) * (np.eye(192) - u @ u.T)
    q = .5 * (q+q.T)
    q *= 192 / np.trace(q)
    values, vectors = np.linalg.eigh(q)
    if values.min() <= 0:
        raise ValueError("positive definite task metric")
    sqrt = (vectors*np.sqrt(values)[None,:]) @ vectors.T
    return sqrt, dict(singular_values=s.tolist(), ridge=float(ridge),
                      eigenvalues=values.tolist(), trace=float(np.trace(q)))
