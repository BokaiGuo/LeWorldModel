import math

import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_dev303_v3_model_evaluation_contract_20261102_exclusive as m,
)

def prepared():
    labels = [f"rgb_boundary_approach_{i}" for i in range(4)] + [
        "zero", "task_greedy_direct", "random_control"]
    return {"schema": m.NAME + ".prepare.v1", "status": "development_prepared_not_executable",
            "P1_admission": False, "physical_execution_authorized": False,
            "candidates": [m._row(i, label, np.zeros((5, 2)), selectable=i < 4)
                           for i, label in enumerate(labels)]}


def test_action_shape_bounds_hash_and_nonfinite():
    row = m._row(0, "rgb_boundary_approach_0", np.zeros((5, 2)), selectable=True)
    assert row["actions_sha256"] == m.digest(np.zeros((5, 2), dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="bounded"):
        m._row(0, "x", np.full((5, 2), 1.01), selectable=True)
    with pytest.raises(ValueError, match="bounded"):
        m._row(0, "x", np.full((5, 2), np.nan), selectable=True)


def test_selection_is_stable_and_never_selects_controls():
    result = m.select(prepared(), [2, 1, 1, 3, -50, -60, -70])
    assert result["selected_index"] == 1
    assert result["control_indices"] == {"zero": 4, "task_greedy_direct": 5, "random": 6}
    assert result["ledger_verified"] is False and result["P1_admission"] is False


@pytest.mark.parametrize("scores", [[0] * 6, [0] * 8, [0] * 6 + [math.nan],
                                    [0] * 6 + [None], [0] * 6 + [True]])
def test_missing_or_bad_score_abstains(scores):
    with pytest.raises(ValueError, match="complete finite"):
        m.select(prepared(), scores)


def test_untrusted_prepare_or_ledger_bytes_rejected():
    with pytest.raises(ValueError, match="SHA"):
        m.build(b"{}", b"", b"{}", b"{}", "unverified")


def test_action_tampering_rejected():
    value = prepared()
    value["candidates"][0]["actions"][0][0] = 0.4
    with pytest.raises(ValueError, match="identity or SHA"):
        m.select(value, [0] * 7)
