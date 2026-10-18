from __future__ import annotations

import numpy as np

from experiments.run_live_cube_rgb_macro_ranker import (
    _approach_observation,
    _mode_conditioned_reclose_decision,
    _online_effect_action,
    _effect_probe_required,
    _support_probe_candidate_index,
    _rgb_block_goal_error,
    _online_macro_bank,
    _factorized_online_macro_bank,
    _multiphase_factorized_online_macro_bank,
    _physical_features,
    _rgb_approach_action,
)
from experiments.run_live_cube_rgb_pick_place import _approach_action, _greedy_effect_action
from experiments.audit_cube_macro_action_ranking import _macro_bank


def test_rgb_approach_action_is_bounded_and_has_five_controls() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:12, 24:28, 0] = 220
    image[8:12, 24:28, 1] = 20
    image[8:12, 24:28, 2] = 20
    # The frozen RGB rule must remain a bounded action even with incomplete
    # effector visibility.
    action = _rgb_approach_action(image)
    assert action.shape == (5,)
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_approach_observation_marks_missing_effector_without_fabricating_error() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:12, 24:28, 0] = 220
    observation = _approach_observation(image)
    assert observation["block_valid"] is True
    assert observation["effector_valid"] is False
    assert observation["approach_error_norm"] is None


def test_physical_ranker_features_use_one_chart_for_current_previous_and_goal() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:12, 24:28, 0] = 220
    branch = {
        "current_image": image,
        "previous_image": image.copy(),
        "goal_image": image.copy(),
        "previous_action": np.zeros(5, dtype=np.float32),
    }
    bank = _macro_bank(length=20, direction_count=4, approach_steps=4)
    features = _physical_features([branch], bank, "cpu")
    assert features.shape == (len(bank), 47)
    assert np.isfinite(features.numpy()).all()


def test_online_macro_bank_uses_validated_cube_gripper_close_sign() -> None:
    bank = _online_macro_bank()
    assert bank.shape == (5, 20, 5)
    assert np.all(bank[1:, 10:14, 4] == 1.0)
    assert np.all(bank[1:, :10, 4] == 0.0)
    long_bank = _online_macro_bank(length=40)
    close_indices = np.flatnonzero(long_bank[1, :, 4] > 0.5)
    assert len(close_indices) >= 4
    low_gain = _online_macro_bank(push_scale=0.25)
    assert np.max(np.abs(low_gain[1:, 14:, :2])) <= 0.25


def test_factorized_bank_separates_approach_and_push_directions() -> None:
    bank = _factorized_online_macro_bank(push_scale=0.25)
    assert bank.shape == (49, 20, 5)
    pairs = {
        (tuple(np.round(row[:4, :2].mean(axis=0), 5)),
         tuple(np.round(row[14:, :2].mean(axis=0), 5)))
        for row in bank[1:]
    }
    assert len(pairs) == 48  # 16 direction pairs x 3 declared push gains
    assert len({pair[0] for pair in pairs}) == 4
    assert len({
        tuple(np.round(np.asarray(pair[1]) / max(np.linalg.norm(pair[1]), 1e-8), 5))
        for pair in pairs
    }) == 4
    assert np.all(bank[1:, 10:14, 4] == 1.0)
    assert np.max(np.abs(bank[1:, 14:, :2])) <= 0.25


def test_multiphase_bank_has_two_persistent_push_phases() -> None:
    bank = _multiphase_factorized_online_macro_bank(length=20, push_scale=0.25)
    assert bank.shape == (129, 20, 5)
    assert np.all(bank[:, 10:, 4] >= 0.0)
    assert any(
        not np.allclose(row[10:15, :2].mean(axis=0), row[15:, :2].mean(axis=0))
        for row in bank[1:]
    )


def test_rgb_block_goal_error_uses_physical_red_chart() -> None:
    current = np.zeros((32, 32, 3), dtype=np.uint8)
    goal = np.zeros((32, 32, 3), dtype=np.uint8)
    current[8:12, 8:12, 0] = 220
    goal[8:12, 24:28, 0] = 220
    error, valid = _rgb_block_goal_error(current, goal)
    assert valid is True
    assert error > 0.5


def test_mode_conditioned_reclose_fails_closed_on_missing_rgb_progress() -> None:
    should_reclose, reason, progress = _mode_conditioned_reclose_decision(
        [{"before": 0.20, "after": 0.20}], 1.0e-3,
    )
    assert should_reclose is True
    assert reason == "insufficient_rgb_progress"
    assert np.isclose(progress, 0.0)


def test_mode_conditioned_reclose_keeps_persistent_contact_after_progress() -> None:
    should_reclose, reason, progress = _mode_conditioned_reclose_decision(
        [{"before": 0.20, "after": 0.18}], 1.0e-3,
    )
    assert should_reclose is False
    assert reason == "progress_ok"
    assert np.isclose(progress, 0.02)


def test_online_effect_action_fails_closed_on_rank_one_support() -> None:
    action, diagnostics = _online_effect_action(
        [np.asarray([1.0, 0.0])] * 4,
        [np.asarray([0.1, 0.0])] * 4,
        np.asarray([1.0, 0.0]),
        np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
    )
    assert action is None
    assert diagnostics["status"] == "insufficient_support"


def test_online_effect_action_fails_closed_before_first_observation() -> None:
    action, diagnostics = _online_effect_action(
        [], [], np.asarray([1.0, 0.0]), np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert action is None
    assert diagnostics["rows"] == 0


def test_online_effect_action_selects_supported_progress_direction() -> None:
    actions = [
        np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]),
        np.asarray([0.0, 1.0]), np.asarray([0.0, -1.0]),
    ]
    deltas = [
        np.asarray([0.2, 0.0]), np.asarray([-0.2, 0.0]),
        np.asarray([0.0, 0.1]), np.asarray([0.0, -0.1]),
    ]
    selected, diagnostics = _online_effect_action(
        actions, deltas, np.asarray([1.0, 0.0]), np.asarray(actions),
    )
    assert diagnostics["status"] == "ok"
    assert np.allclose(selected, [1.0, 0.0])


def test_online_effect_action_rejects_rank_two_but_zero_strength_map() -> None:
    actions = [
        np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]),
        np.asarray([0.0, 1.0]), np.asarray([0.0, -1.0]),
    ]
    selected, diagnostics = _online_effect_action(
        actions, [np.zeros(2) for _ in actions], np.asarray([1.0, 0.0]),
        np.asarray(actions),
    )
    assert selected is None
    assert diagnostics["status"] == "weak_effect"
    assert diagnostics["mapping_norm"] == 0.0


def test_support_probe_forces_an_orthogonal_candidate_after_rank_one_history() -> None:
    bank = _online_macro_bank()
    index = _support_probe_candidate_index(bank, [np.asarray([1.0, 0.0])] * 4)
    assert index in {3, 4}


def test_support_probe_can_reexcite_rank_two_but_weak_effect_history() -> None:
    bank = _online_macro_bank()
    history = [
        np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]),
        np.asarray([0.0, 1.0]), np.asarray([0.0, -1.0]),
    ]
    index = _support_probe_candidate_index(bank, history, force=True)
    assert index in {1, 2, 3, 4}


def test_effect_probe_is_required_for_nonweak_rank_one_history() -> None:
    actions = [np.asarray([1.0, 0.0])] * 4
    deltas = [np.asarray([0.2, 0.0])] * 4
    required, rank = _effect_probe_required(actions, deltas)
    assert required is True
    assert rank == 1


def test_pick_place_hybrid_approach_is_bounded_and_uses_rgb_fallback() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    # Red target and visible effector marker are sufficient for the RGB chart
    # helpers; the controller must remain safe even with a missing world chart.
    image[14:18, 22:26, 0] = 220
    goal = image.copy()
    action = _approach_action(
        image,
        pixel_mapping=None,
        approach_model=None,
        goal_image=goal,
        controller="hybrid",
        pixel_gain=0.25,
        world_gain=1.0,
    )
    assert action.shape == (5,)
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_greedy_effect_action_stays_inside_observed_support() -> None:
    mapping = np.asarray([[0.2, 0.0], [0.0, 0.1]], dtype=np.float32)
    action = _greedy_effect_action(np.asarray([1.0, 0.2]), mapping, 0.1)
    assert action is not None
    assert np.isclose(np.linalg.norm(action), 0.1)
    assert np.count_nonzero(action) == 1
