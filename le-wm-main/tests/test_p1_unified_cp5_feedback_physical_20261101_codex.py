import numpy as np

from experiments.spectral_reachability import p1_unified_cp5_feedback_physical_20261101_codex as subject


def test_contract_has_fixed_six_plans_and_no_physical_neural_calls():
    value = subject.build()
    assert value["roles"] == list(subject.base.ROLES)
    assert value["new_neural_calls_during_physical"] == 0
    assert value["post_cp5_physical_outcomes_used_for_selection"] is False
    assert len(value["roots"]) == 2
    for root in value["roots"]:
        assert set(root["candidates"]) == set(subject.base.ROLES)
        for plan in root["candidates"].values():
            raw = np.asarray(plan, np.float32)
            assert raw.shape == (3, 5, 2)
            assert np.isfinite(raw).all()
            assert np.max(np.abs(raw)) <= 1


def test_root1_greedy_preserves_early_success_plan():
    value = subject.build()
    root1 = next(root for root in value["roots"] if root["slot"] == 1)
    assert root1["decisions"]["same_bank_greedy"] == {
        "kind": "original_early_success", "stop_step_source": 4
    }
