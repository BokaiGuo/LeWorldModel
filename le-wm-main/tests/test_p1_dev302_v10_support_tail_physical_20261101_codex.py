import numpy as np

from experiments.spectral_reachability import (
    p1_dev302_v10_support_tail_physical_20261101_codex as subject,
)


def test_v10_physical_contract_has_four_equal_horizon_plans():
    value = subject.build()
    assert [row["role"] for row in value["roles"]] == list(subject.ROLES)
    assert [row["candidate_index"] for row in value["roles"]] == [16, 0, 4, 7]
    assert all(np.asarray(row["actions"]).shape == (15, 2) for row in value["roles"])
    assert value["same_replanning_budget"] is True
    assert value["same_feedback_count"] is True
    assert value["P1_admission"] is False
