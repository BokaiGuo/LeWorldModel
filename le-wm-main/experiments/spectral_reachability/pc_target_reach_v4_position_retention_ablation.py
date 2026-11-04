"""Single-process retention ablation for the sealed v4 development method."""
import numpy as np

from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as v4

METHOD = "target_reach_fixed_terminal_position_retention_ablation_v1"


def decide(arm, root, step, target, predict, meter, carry=None):
    """Preserve v4 probes and budget, replacing only retention ranking.

    This scoped monkeypatch is for a single-process diagnostic runner only;
    it is not thread-safe and must not be used in production.
    """
    cached = {"positions": None}

    def capture(actions):
        positions, scores = predict(actions)
        if np.asarray(actions).shape == (9, 5, 2):
            cached["positions"] = np.asarray(positions).copy()
        return positions, scores

    def position_retain(scores, responses, number=4):
        if cached["positions"] is None or len(scores) != len(cached["positions"]):
            raise ValueError("position retention candidate bank mismatch")
        return core.retain(scores, cached["positions"], number)

    original = v4.retain
    try:
        v4.retain = position_retain
        return v4.decide(arm, root, step, target, capture, meter, carry)
    finally:
        v4.retain = original
