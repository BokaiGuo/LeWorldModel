from experiments.spectral_reachability import codex_p1_adam_best_iterate_20261030_main as selector
from experiments.spectral_reachability import codex_p1_unreg_best_compare_20261030_main as comparison


def test_unregularized_saved_selection_and_physical_pairs():
    read = comparison.compare.plan.gap.prior.gap.read
    plan, _ = read(comparison.physical.PLAN, comparison.physical.SHA)
    assert plan["penalty_weight"] == 0
    assert len(plan["rows"]) == 24
    for row in plan["rows"]:
        assert row["selected_step"] == selector.select_best(row["objective_history"])
        assert row["selected_score"] == row["objective_history"][row["selected_step"]]
    result = comparison.build()
    assert result["summary"]["fixed"] == 24
    assert result["summary"]["comparable"] == 22
    assert result["summary"]["trust_success"] == 2
    assert result["summary"]["unreg_success"] == 1
    assert all(row["unreg_minus_trust"] is None for row in result["rows"]
               if row["trust_status"] == "native_stopped" or row["unreg_status"] == "native_stopped")
