"""Atlas v3: explicitly explore local task-response nullspace directions.

This extends, rather than modifies, the sealed v2 candidate bank. A nullspace
direction is only a first-order statement about the chosen model metric; it is
not evidence that real control capability changes.
"""

import numpy as np

from .d2p_atlas_search_v2 import AtlasSearch


def task_nullspace(jacobian, sqrt_q, action_scale):
    """Return raw-action null directions for Q^1/2 G R^-1/2.

    Full right singular vectors are necessary only for a wide matrix. Avoid
    full U for the common tall visual-latent Jacobian.
    """
    jacobian, sqrt_q, action_scale = map(
        lambda value: np.asarray(value, dtype=np.float64),
        (jacobian, sqrt_q, action_scale),
    )
    if (jacobian.ndim != 2 or min(jacobian.shape) < 1
            or sqrt_q.shape != (jacobian.shape[0], jacobian.shape[0])
            or action_scale.shape != (jacobian.shape[1],)
            or not all(np.isfinite(a).all() for a in (jacobian, sqrt_q, action_scale))
            or np.any(action_scale <= 0)):
        raise ValueError("finite aligned weighted response geometry required")
    weighted = (sqrt_q @ jacobian) * action_scale
    _, singular, vh = np.linalg.svd(weighted, full_matrices=weighted.shape[0] < weighted.shape[1])
    tolerance = singular[0] * max(weighted.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > tolerance))
    return vh[rank:] * action_scale


class NullspaceAtlasSearch(AtlasSearch):
    def __init__(self, compiler, config, *, null_modes=2):
        super().__init__(compiler, config)
        if type(null_modes) is not int or null_modes < 1:
            raise ValueError("positive null mode bound required")
        self.null_modes = null_modes

    def _bank(self, context, rng):
        bank = super()._bank(context, rng)
        c = self.compiler
        shape = (c.config.packet_steps, len(c.scale))
        zero = np.zeros(shape)
        jacobian = c._call("jacobian", context, zero)
        directions = task_nullspace(jacobian, c.q, np.tile(c.scale, shape[0]))
        for i, raw in enumerate(directions[:self.null_modes]):
            direction = raw.reshape(shape)
            direction = direction / max(float(np.max(np.abs(direction))), 1e-12)
            for sign in (-1, 1):
                proposed = sign * c.config.prefix_radius * direction
                # Preserve direction while finding the largest legal raw step.
                scale = 1.0
                upper = np.broadcast_to(c.high, shape)
                lower = np.broadcast_to(c.low, shape)
                positive, negative = proposed > 0, proposed < 0
                if positive.any():
                    scale = min(scale, float(np.min(upper[positive] / proposed[positive])))
                if negative.any():
                    scale = min(scale, float(np.min(lower[negative] / proposed[negative])))
                bank.append((f"null_{i}_{sign:+d}", proposed * scale))
        return bank

    def search(self, context):
        result = super().search(context)
        result["schema"] = "d2p_atlas_nullspace.v3"
        result["configuration"]["null_modes"] = self.null_modes
        return result
