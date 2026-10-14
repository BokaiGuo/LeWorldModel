import numpy as np

from experiments.run_pusht_repaired_predictor_mpc import (
    CalibratedResidualAdapter,
    block_cost,
    make_action_sequence,
    proposal_sequences,
    staged_proposal_sequences,
    task_state_cost,
)
from experiments.run_pusht_two_stage_oracle_planning import (
    angle_error,
    block_cost as oracle_block_cost,
    joint_cost,
)


def test_public_action_sequence_preserves_history_and_horizon():
    past = np.asarray([[0.1, -0.2], [0.2, -0.1]], dtype=np.float32)
    future = np.zeros((4, 2), dtype=np.float32)
    encoded = make_action_sequence(past, future)
    assert encoded.shape == (6, 10)
    assert np.isfinite(encoded).all()


def test_proposals_are_bounded_and_include_guided_controls():
    state = np.asarray([100.0, 100.0, 160.0, 100.0, 0.0, 1.0, 0.0, 0.0])
    goal = np.asarray([100.0, 100.0, 220.0, 100.0, 0.0, 0.0, 0.0])
    candidates = proposal_sequences(
        np.random.default_rng(7), state, goal, 12, 8, 2, 0.2
    )
    assert candidates.shape == (12, 8, 2)
    assert np.all(candidates <= 1.0) and np.all(candidates >= -1.0)
    assert np.allclose(candidates[0], 0.0)


def test_task_cost_is_zero_at_goal_and_penalizes_agent_and_block():
    goal = np.asarray([100.0, 100.0, 220.0, 100.0, 0.0, 0.0, 0.0])
    goal_features = np.asarray([100.0, 100.0, 220.0, 100.0, 0.0, 1.0, 0.0, 0.0])
    assert task_state_cost(goal_features, goal) == 0.0
    moved_agent = goal_features.copy()
    moved_agent[0] += 30.0
    assert task_state_cost(moved_agent, goal) > 0.0
    moved_block = goal_features.copy()
    moved_block[2] += 30.0
    assert task_state_cost(moved_block, goal) > 0.0


def test_staged_proposals_are_bounded_and_contain_phase_changes():
    state = np.asarray([100.0, 100.0, 160.0, 100.0, 0.0, 1.0, 0.0, 0.0])
    goal = np.asarray([100.0, 100.0, 220.0, 100.0, 0.0, 0.0, 0.0])
    candidates = staged_proposal_sequences(
        np.random.default_rng(11), state, goal, 32, 8, 2, 0.15
    )
    assert candidates.shape == (32, 8, 2)
    assert np.isfinite(candidates).all()
    assert np.all(candidates <= 1.0) and np.all(candidates >= -1.0)
    # The structured bank must expose both a braking/approach action and a
    # later push action; otherwise it collapses back to constant proposals.
    assert np.any(np.linalg.norm(candidates[1:, 2:4] - candidates[1:, :2], axis=-1) > 1e-6)


def test_block_cost_ignores_agent_position_but_keeps_block_pose():
    goal = np.asarray([100.0, 100.0, 220.0, 100.0, 0.0, 0.0, 0.0])
    exact_block = np.asarray([0.0, 0.0, 220.0, 100.0, 0.0, 1.0, 0.0, 0.0])
    assert block_cost(exact_block, goal) == 0.0


def test_residual_adapter_preserves_shape_and_can_correct_base_prediction():
    rng = np.random.default_rng(4)
    predicted = rng.normal(size=(12, 4, 3)).astype(np.float32)
    actions = rng.normal(size=(12, 6, 2)).astype(np.float32)
    targets = predicted + 0.2 * actions[:, 2:6, :1]
    adapter = CalibratedResidualAdapter.fit(predicted, actions, targets, alpha=1e-6)
    corrected = adapter.apply(predicted, actions)
    assert corrected.shape == predicted.shape
    assert np.isfinite(corrected).all()
    assert np.mean((corrected - targets) ** 2) < np.mean((predicted - targets) ** 2)


def test_two_stage_objective_separates_block_from_agent_return():
    goal = np.asarray([100.0, 100.0, 220.0, 100.0, 0.0, 0.0, 0.0])
    block_reached_agent_far = np.asarray([420.0, 420.0, 220.0, 100.0, 0.0, 1.0, 0.0, 0.0])
    assert oracle_block_cost(block_reached_agent_far, goal) == 0.0
    assert joint_cost(block_reached_agent_far, goal) > 0.0
    assert np.isclose(angle_error(2 * np.pi - 0.05, 0.05), 0.1)
