import numpy as np

from experiments.gdd_wm import run_pointmaze_repairability_frontier_pilot as pilot


def _row(root: str, split: str, candidate: str, action: float) -> dict:
    return {
        "instance_id": root,
        "split": split,
        "candidate_id": candidate,
        "source_state_raw": [0.0, 0.0, 0.0, 0.0],
        "goal_raw": [1.0, 0.0],
        "source_latent_model": [0.1, 0.2],
        "action_sequence": [[action, 0.0]] * 5,
    }


def test_assign_roles_makes_train_gdd_calibration_outcome_root_disjoint():
    rows = []
    for index in range(12):
        split = "train" if index < 8 else ("calibration" if index < 10 else "outcome")
        rows.extend([_row(f"r{index}", split, f"a{index}", 0.1), _row(f"r{index}", split, f"b{index}", -0.1)])
    roles = pilot._assign_roles(rows)
    assert set(roles.values()) == {"train", "gdd", "calibration", "outcome"}
    assert len(roles) == 12


def test_preselection_contract_uses_scores_before_outcome_costs():
    group = [_row("r", "outcome", "left", 0.1), _row("r", "outcome", "right", -0.1)]
    scores = {"r": {"psd": np.asarray([2.0, 1.0]), "mlp": np.asarray([1.0, 2.0]), "naive": np.asarray([2.0, 1.0])}}
    costs = {"r": np.asarray([4.0, 0.0])}
    result = pilot._condition_outcome({"r": group}, scores, costs, {"r": 0.9}, threshold=0.5)
    row = result["rows"]["r"]
    assert row["routed_pick"] == row["mlp_pick"] == 1
    assert row["repair_outcome_mlp_beats_psd"] is True
    assert result["outcome_cost_used_for_selection"] is False


def test_audit_rejects_missing_preselection_provenance(tmp_path):
    from experiments.gdd_wm.audit_pointmaze_repairability_frontier import audit

    path = tmp_path / "report.json"
    path.write_text("{}", encoding="utf-8")
    result = audit(path)
    assert result["status"] == "rejected_contract_audit"


def test_summary_states_whether_frozen_preselection_predicted_repair():
    result = {
        "outcome_cost_used_for_selection": False,
        "rows": {
            "r": {
                "psd_regret": 0.2,
                "mlp_regret": 0.0,
                "naive_regret": 0.2,
                "routed_regret": 0.0,
                "repair_outcome_mlp_beats_psd": True,
                "router_selected_mlp_pre_outcome": True,
            },
            "r2": {
                "psd_regret": 0.0,
                "mlp_regret": 0.1,
                "naive_regret": 0.0,
                "routed_regret": 0.0,
                "repair_outcome_mlp_beats_psd": False,
                "router_selected_mlp_pre_outcome": False,
            },
        },
    }
    summary = pilot._summary(result)
    assert summary["preselection_prediction_verdict"]["status"] == "confirmed_in_this_pilot"
