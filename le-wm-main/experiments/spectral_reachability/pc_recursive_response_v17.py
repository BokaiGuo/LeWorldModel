"""Score-augmented response proposals; original nonlinear score stays final."""
from dataclasses import asdict

import numpy as np

from experiments.spectral_reachability import pc_recursive_response_v13 as v13


def compile_program(predict, target, config, *, score_scale, score_target, carry=None):
    """Augment chart with the legal planner score, without extra predictions.

    Both scale and target are explicit because the score need not be a
    nonnegative MSE. This modifies proposal geometry only: candidate and
    final scores remain exactly the caller's original scalar score.
    """
    chart_target = np.asarray(target, np.float64)
    if (chart_target.ndim != 1 or not chart_target.size or not np.isfinite(chart_target).all()
            or not np.isfinite(score_scale) or score_scale <= 0 or not np.isfinite(score_target)):
        raise ValueError("invalid score-augmented geometry")

    def augmented(actions):
        features, scores = predict(actions)
        features, scores = np.asarray(features, np.float64), np.asarray(scores, np.float64)
        if features.shape != (len(actions), len(chart_target)) or scores.shape != (len(actions),):
            raise ValueError("invalid predictor output")
        return np.concatenate((features, score_scale * scores[:, None]), axis=1), scores

    extended_target = np.concatenate((chart_target, [score_scale * score_target]))
    result = v13.compile_program(augmented, extended_target, config, carry=carry)
    return {**result, "method": "recursive_response_v17_score_augmented_development",
            "score_geometry": {"scale": float(score_scale), "target": float(score_target),
                               "chart_target": chart_target.tolist(), "config": asdict(config)}}
