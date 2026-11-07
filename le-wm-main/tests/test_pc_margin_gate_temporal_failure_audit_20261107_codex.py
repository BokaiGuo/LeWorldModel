import numpy as np
import pytest

from experiments.pc_margin_gate_temporal_failure_audit_20261107_codex import classify


def test_first_step_and_end_loss():
    a = np.zeros((8, 8), dtype=bool)
    b = a.copy()
    a[0, 0] = a[0, 1] = True
    at = np.zeros((8, 8, 25), dtype=bool)
    bt = at.copy()
    at[0, 0, 14] = True
    bt[0, 1, 24] = True
    result = classify(a, b, at, bt)
    assert result["changed"] == 2
    assert result["direct_only_first_steps"] == [15]
    assert result["gated_only_first_steps"] == [25]
    assert result["direct_only_success_lost_by_end"] == 1


def test_reject_unchanged_success_disagreement():
    a = np.zeros((8, 8), dtype=bool)
    at = np.zeros((8, 8, 25), dtype=bool)
    bt = at.copy()
    at[0, 0, 0] = True
    with pytest.raises(ValueError):
        classify(a, a, at, bt)
