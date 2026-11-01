"""Temporary task metric from supplied goal-image embeddings, never outcomes.

Projects out the finite span of specified task-equivalent goals. This is not a
learned readout and does not promise global agent invariance or object fidelity.
It requires an explicit goal-image set; a single RGB goal cannot supply it.
"""
from __future__ import annotations

import numpy as np


def compile_metric(goal_vectors):
    goals = np.asarray(goal_vectors, dtype=np.float64)
    if goals.ndim != 2 or goals.shape[0] < 2 or goals.shape[1] < 2 or not np.isfinite(goals).all():
        raise ValueError("at least two finite task-equivalent goal vectors required")
    directions = (goals[1:]-goals[0]).T
    basis, singular, _ = np.linalg.svd(directions, full_matrices=False)
    tolerance = (float(singular[0]) if len(singular) else 0.) * max(directions.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(singular > tolerance))
    if rank == goals.shape[1]:
        raise ValueError("task metric would discard the entire latent space")
    nuisance = basis[:, :rank]
    projection = np.eye(goals.shape[1])-nuisance@nuisance.T
    return projection, {"rank": rank, "singular_values": singular.tolist(), "rank_tolerance": tolerance,
                        "goal_variants": len(goals), "latent_dim": goals.shape[1],
                        "finite_goal_span_only": True, "physical_outcomes_used": False,
                        "additional_training": False}
