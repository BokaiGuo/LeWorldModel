"""Deterministic model-side response diversity for fixed prefix candidates.

The caller must supply action Jacobians in the same action chart and a common,
predeclared task direction. This module never consumes physical outcomes and
does not evaluate non-greediness or authorize any scientific stage.
"""
import numpy as np


def select(jacobians, task_direction, stage_scores, count):
    """Choose the stage-best candidate, then farthest response vectors.

    Features are J_i.T @ r for one shared direction r; J_i already contains
    the caller's fixed target weighting and action scaling. Squared distances
    use the original feature scale, not per-candidate normalization. Ties
    are broken by original index. All candidates and selection distances are
    returned for auditing, and a zero response is not called a capability gain.
    """
    j = np.asarray(jacobians, dtype=np.float64)
    r = np.asarray(task_direction, dtype=np.float64)
    scores = np.asarray(stage_scores, dtype=np.float64)
    if (j.ndim != 3 or j.shape[0] < 1 or j.shape[1] < 1 or j.shape[2] < 1
            or r.shape != (j.shape[1],) or scores.shape != (j.shape[0],)
            or not isinstance(count, int) or isinstance(count, bool)
            or count < 1 or count > j.shape[0]
            or not np.isfinite(j).all() or not np.isfinite(r).all()
            or not np.isfinite(scores).all() or np.linalg.norm(r) == 0):
        raise ValueError("invalid common task response inputs")
    features = np.einsum("nza,z->na", j, r)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite task responses")
    distances = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=-1)
    if not np.isfinite(distances).all():
        raise ValueError("nonfinite response distances")
    selected = [int(np.argmin(scores))]
    selection_distance = [None]
    while len(selected) < count:
        remaining = [i for i in range(j.shape[0]) if i not in selected]
        nearest = np.min(distances[remaining][:, selected], axis=1)
        choice = int(np.argmax(nearest))
        selected.append(remaining[choice])
        selection_distance.append(float(nearest[choice]))
    return {
        "selected_indices": selected,
        "selection_min_squared_distance": selection_distance,
        "response_features": features.tolist(),
        "stage_scores": scores.tolist(),
        "task_direction": r.tolist(),
        "interpretation": "model-side common-direction action response diversity only",
        "P1_admission": False,
    }
