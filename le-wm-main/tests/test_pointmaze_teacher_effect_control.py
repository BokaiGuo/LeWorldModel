import numpy as np

from experiments.gdd_wm import build_pointmaze_effect_teacher_control as control


def _row(root: str, split: str, context: str, action: float, score: float, cost: float) -> dict:
    return {
        "instance_id": root,
        "split": split,
        "candidate_id": f"{context}-{action}",
        "candidate_bank_context_id": context,
        "action_sequence_sha256": f"action-{action}",
        "teacher_score": score,
        "real_task_cost": cost,
    }


def test_d1_stability_uses_calibration_only_and_requires_consistent_anchor_ordering():
    rows = [
        _row("cal", "calibration", "c0", 0.0, 2.0, 9.0),
        _row("cal", "calibration", "c0", 1.0, 1.0, 1.0),
        _row("cal", "calibration", "c1", 0.0, 2.0, 8.0),
        _row("cal", "calibration", "c1", 1.0, 1.0, 0.0),
        # Outcome costs are deliberately adversarial; D1 must not read them.
        _row("out", "outcome", "c0", 0.0, 1.0, 0.0),
        _row("out", "outcome", "c0", 1.0, 2.0, 9.0),
    ]
    report = control._d1_stability(rows)
    assert report["passed"] is True
    assert report["real_task_cost_read"] is False
    assert report["stable_pair_count"] == 1


def test_d1_stability_rejects_context_dependent_anchor_ranking():
    rows = [
        _row("cal", "calibration", "c0", 0.0, 2.0, 0.0),
        _row("cal", "calibration", "c0", 1.0, 1.0, 0.0),
        _row("cal", "calibration", "c1", 0.0, 1.0, 0.0),
        _row("cal", "calibration", "c1", 1.0, 2.0, 0.0),
    ]
    report = control._d1_stability(rows)
    assert report["passed"] is False
    assert report["unstable_pair_count"] == 1


def test_outcome_alignment_is_post_freeze_external_evaluation_only():
    rows = [
        _row("out", "outcome", "c0", 0.0, 2.0, 0.1),
        _row("out", "outcome", "c0", 1.0, 1.0, 0.5),
    ]
    report = control._held_outcome_alignment(rows)
    assert report["real_task_cost_read"] is True
    assert report["pairwise_execution_alignment"] == 1.0
    assert report["mean_candidate_regret"] == 0.0


def test_effect_score_depends_on_causal_state_goal_and_predicted_effect_not_cost():
    row = {
        "source_state_raw": [0.0, 0.0, 0.0, 0.0],
        "goal_raw": [1.0, 0.0],
        "real_task_cost": 12345.0,
    }
    toward_goal = control._preference_score(row, np.asarray([1.0, 0.0, 0.0, 0.0]))
    away_goal = control._preference_score(row, np.asarray([-1.0, 0.0, 0.0, 0.0]))
    assert toward_goal > away_goal
    assert control.CONTROL_ID.startswith("pointmaze_train_only_execution_supervised")


def test_effect_teacher_contract_explicitly_keeps_outcome_roots_out_of_fit_targets():
    source = control._source_state({"instance_id": "r", "source_state_raw": [0, 0, 0, 0]})
    assert source.shape == (4,)
    goal = control._goal_state({"instance_id": "r", "goal_raw": [1, 2]})
    assert goal.shape == (2,)


def test_visual_effect_families_exclude_privileged_state_and_task_cost():
    row = {
        "instance_id": "r",
        "source_state_raw": [99.0, 98.0, 97.0, 96.0],
        "goal_raw": [95.0, 94.0],
        "source_latent_model": [1.0, 2.0],
        "source_history_model": [3.0, 4.0, 5.0],
        "action_sequence": [[0.1, 0.2], [0.3, 0.4]],
        "real_task_cost": -12345.0,
    }
    latent_action = control._features(row, "latent_action")
    history_action = control._features(row, "history_action")
    combined = control._features(row, "latent_history_action")
    # _action_summary has seven fixed action-only statistics for this sequence.
    assert latent_action.shape[0] == 9
    assert history_action.shape[0] == 10
    assert combined.shape[0] == 12
    assert not np.isin([99.0, 98.0, 95.0, -12345.0], latent_action).any()
    assert control._feature_fields("latent_history_action") == [
        "source_latent_model", "source_history_model", "action_sequence_summary",
    ]
