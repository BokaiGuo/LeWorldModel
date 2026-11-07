import numpy as np
import pytest

from experiments import pc_tworoom_failure_case_temporal_20261107_codex as diag


def test_sealed_action_order_and_identity():
    selected = np.full((25, 2), 0.25, dtype=np.float32)
    bank = {"r38_b": np.zeros((8, 8, 25, 2), dtype=np.float32)}
    prior = {"r38_actions": np.stack([selected] * 4)}
    plan = {"rows": [{"root": 38, "decisions": {"lookahead": {"action_sha256": diag.b.q.digest(selected)}}}]}
    result = diag.actions(38, bank, prior, plan)
    assert result.shape == (65, 25, 2)
    assert np.array_equal(result[-1], selected)
    assert np.count_nonzero(result[:-1]) == 0


def test_wrong_selected_action_fails_closed():
    selected = np.zeros((25, 2), dtype=np.float32)
    bank = {"r38_b": np.zeros((8, 8, 25, 2), dtype=np.float32)}
    prior = {"r38_actions": np.stack([selected] * 4)}
    plan = {"rows": [{"root": 38, "decisions": {"lookahead": {"action_sha256": "incorrect"}}}]}
    with pytest.raises(ValueError, match="identity"):
        diag.actions(38, bank, prior, plan)
