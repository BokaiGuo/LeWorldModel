import pytest

from experiments.spectral_reachability import p1_dev303_residual_spectral_physical_20261102_main as runner


def test_sealed_report_produces_complete_panel_and_two_preselected_arms():
    contract = runner.build()
    assert len(contract["specs"]) == 42
    assert sum(s["selected_before_physics"] for s in contract["specs"]) == 2
    assert {s["metric"] for s in contract["specs"]} == {"latent", "goal_tangent"}
    assert contract["P1_admission"] is False
    assert contract["feedback"] == "open_loop_15_actions_not_cp5_replanning"


def test_incomplete_or_invalid_report_rejected():
    with pytest.raises(Exception):
        runner.specs_from_report({"status": "model_only_complete", "P1_admission": False,
                                  "rows": [{"index": 1}]})
