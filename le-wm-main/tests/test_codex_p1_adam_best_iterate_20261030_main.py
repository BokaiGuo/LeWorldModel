import pytest

from experiments.spectral_reachability.codex_p1_adam_best_iterate_20261030_main import select_best


def test_selects_intermediate_and_preserves_earliest_tie():
    scores = [10.0] * 21
    scores[4] = scores[12] = 1.0
    assert select_best(scores) == 4


def test_rejects_nonfinite_or_incomplete_history():
    with pytest.raises(ValueError):
        select_best([0.0] * 20)
    with pytest.raises(ValueError):
        select_best([0.0] * 20 + [float("nan")])


def test_saved_plan_and_physical_pair_reconstruct():
    from experiments.spectral_reachability import codex_p1_adam_best_iterate_compare_20261030_main as comparison
    from experiments.spectral_reachability import codex_p1_adam_best_iterate_physical_20261030_main as physical

    data, _ = comparison.compare.plan.gap.prior.gap.read(physical.PLAN, physical.SHA)
    for row in data["rows"]:
        assert row["selected_step"] == select_best(row["objective_history"])
        assert row["selected_score"] == row["objective_history"][row["selected_step"]]
        assert row["selected_score"] <= min(row["baseline_score"], row["optimized_score"]) + 1e-6
    result = comparison.build()
    assert len(result["rows"]) == 24
    assert result["summary"]["comparable"] == 22
    assert all(r["new_minus_trust"] is None for r in result["rows"]
               if r["trust_status"] == "native_stopped" or r["new_status"] == "native_stopped")
