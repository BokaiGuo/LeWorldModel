"""Development-only task-residual spectral prefix proposals.

This module does not select a program or consume physical outcomes.  Proposals
must be checked by the caller's RGB guard and nonlinear model rollout.
"""
import numpy as np


def proposals(anchor, jacobian, sqrt_metric, residual, *, radius, max_modes,
              ridge=0.01, relative_cutoff=1e-6):
    """Rank resolved response modes by their damped projection on the residual.

    Both signs of each selected mode are emitted: the local linear ranking is
    only a proposal mechanism, and is not reliable across contact changes.
    """
    anchor, jacobian, sqrt_metric, residual = (
        np.asarray(x, dtype=np.float64)
        for x in (anchor, jacobian, sqrt_metric, residual)
    )
    if any(not np.isfinite(x).all() for x in (anchor, jacobian, sqrt_metric, residual)):
        raise ValueError("finite inputs required")
    if anchor.shape != (5, 2) or jacobian.ndim != 2 or jacobian.shape[1] != 10:
        raise ValueError("expected a five-step anchor and [latent,10] Jacobian")
    dimension = jacobian.shape[0]
    if not dimension or sqrt_metric.shape != (dimension, dimension) or residual.shape != (dimension,):
        raise ValueError("metric or residual shape mismatch")
    if np.any(np.abs(anchor) > 1):
        raise ValueError("anchor outside action bounds")
    if not np.isfinite(radius) or not 0 < radius <= 1:
        raise ValueError("invalid radius")
    if isinstance(max_modes, bool) or not isinstance(max_modes, (int, np.integer)) or max_modes < 1:
        raise ValueError("max_modes must be a positive integer")
    if not np.isfinite(ridge) or ridge <= 0 or not 0 < relative_cutoff < 1:
        raise ValueError("invalid ridge or cutoff")

    u, singular, vh = np.linalg.svd(sqrt_metric @ jacobian, full_matrices=False)
    if not singular.size or singular[0] <= 1e-12:
        raise ValueError("no resolved model action response")
    rank = int(np.count_nonzero(singular > singular[0] * relative_cutoff))
    weighted_residual = sqrt_metric @ residual
    coefficients = singular[:rank] / (singular[:rank] ** 2 + ridge) * (u[:, :rank].T @ weighted_residual)
    order = np.argsort(-np.abs(coefficients), kind="stable")[:max_modes]
    records = []
    for mode in order:
        direction = vh[mode].reshape(5, 2).copy()
        if direction.flat[np.argmax(np.abs(direction))] < 0:
            direction *= -1
        direction /= np.max(np.abs(direction))
        for sign in (-1, 1):
            actions = np.clip(anchor + sign * radius * direction, -1, 1).astype("<f4")
            records.append({
                "mode": int(mode), "sign": sign, "singular_value": float(singular[mode]),
                "residual_coefficient_magnitude": float(abs(coefficients[mode])),
                "actions": actions,
            })
    return records
