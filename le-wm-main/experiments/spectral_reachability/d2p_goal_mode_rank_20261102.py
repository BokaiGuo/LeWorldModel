"""Task-relevant damped spectral mode ranking for an already weighted Jacobian.

This pure numerical component does not select or execute physical actions.
"""

import numpy as np


def rank_goal_modes(jacobian, residual, damping):
    """Rank modes by decrease in the damped local quadratic objective.

    ``jacobian`` and ``residual`` must already use the same task weighting;
    action variables must already be whitened by the declared action metric.
    The returned ranking applies to the local *unconstrained* problem only.
    """
    matrix = np.asarray(jacobian, dtype=np.float64)
    target = np.asarray(residual, dtype=np.float64)
    if (matrix.ndim != 2 or target.shape != (matrix.shape[0],)
            or not np.isfinite(matrix).all() or not np.isfinite(target).all()
            or not np.isfinite(damping) or damping <= 0):
        raise ValueError("invalid weighted Jacobian, residual, or positive damping")
    left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    projections = left.T @ target
    coefficients = singular * projections / (singular**2 + damping)
    contributions = (singular**2 / (singular**2 + damping)) * projections**2
    return {
        "left": left,
        "singular": singular,
        "right_t": right_t,
        "coefficients": coefficients,
        "contributions": contributions,
        "order": np.argsort(-contributions, kind="stable"),
    }
