import numpy as np

from experiments.spectral_reachability import (
    p1_dev302_rgb_cp5_tail_physical_20261101_codex as subject,
)


def test_contract_has_common_actual_prefix_and_fixed_tail_roster():
    value = subject.build()
    assert [row["role"] for row in value["roles"]] == list(subject.ROLES)
    plans = [np.asarray(row["actions"], np.float32) for row in value["roles"]]
    assert all(plan.shape == (10, 2) for plan in plans)
    assert all(np.array_equal(plan[:5], plans[0][:5]) for plan in plans)
    assert value["same_actual_prefix_all_arms"] is True
    assert value["post_cp5_physical_outcomes_used_for_selection"] is False
    assert value["development_only"] is True
    assert value["fair_total_cost"] is False
    assert value["P1_admission"] is False
