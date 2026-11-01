"""Numeric dynamics-derived proposals; no model, physics, or outcome access."""
import numpy as np


def finite(value):
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("finite numeric input required")
    return array


def spectral_pair(anchor, jacobian, sqrt_metric, radius=0.25):
    """Propose both signs of the leading task-weighted action response mode."""
    anchor, jacobian, q = map(finite, (anchor, jacobian, sqrt_metric))
    if anchor.shape != (5, 2) or jacobian.shape[1:] != (10,):
        raise ValueError("five-step prefix and flattened action Jacobian required")
    if q.shape != (jacobian.shape[0], jacobian.shape[0]) or not 0 < radius <= 1:
        raise ValueError("metric shape or trust radius")
    if np.max(np.abs(anchor)) > 1:
        raise ValueError("bounded anchor required")
    _, singular, vh = np.linalg.svd(q @ jacobian, full_matrices=False)
    if singular[0] <= 1e-12:
        raise ValueError("no resolved model action response")
    direction = vh[0].reshape(5, 2)
    # Canonical sign makes records stable; both signs remain in the bank.
    if direction.flat[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    direction /= np.max(np.abs(direction))
    proposals = [np.clip(anchor + sign * radius * direction, -1, 1).astype("<f4")
                 for sign in (-1, 1)]
    return proposals, {"singular_values": singular.tolist(), "direction": direction.tolist(),
                       "radius": radius, "source": "task_weighted_model_action_jacobian"}


def suffix_bank(response, residual, sqrt_metric, ridge=0.01):
    """A fixed bounded damped least-squares bank, evaluated nonlinearly later."""
    response, residual, q = map(finite, (response, residual, sqrt_metric))
    if response.ndim != 2 or response.shape[1] != 20:
        raise ValueError("ten-step two-dimensional response required")
    if residual.shape != (response.shape[0],) or q.shape != (len(residual), len(residual)):
        raise ValueError("residual/metric mismatch")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("positive finite ridge required")
    u, s, vh = np.linalg.svd(q @ response, full_matrices=False)
    delta = vh.T @ ((s / (s * s + ridge)) * (u.T @ (q @ residual)))
    delta /= max(1.0, float(np.max(np.abs(delta))))
    return np.stack([(scale * delta).reshape(10, 2) for scale in (0, .25, .5, 1)]).astype("<f4")


def costs(vectors, goal, sqrt_metric):
    vectors, goal, q = map(finite, (vectors, goal, sqrt_metric))
    if vectors.ndim != 2 or goal.shape != (vectors.shape[1],) or q.shape != (len(goal), len(goal)):
        raise ValueError("score shape mismatch")
    return np.mean(((vectors - goal) @ q.T) ** 2, axis=1)


def select(scores):
    scores = finite(scores)
    if scores.ndim != 1 or not scores.size:
        raise ValueError("nonempty score vector required")
    return int(np.argmin(scores))
