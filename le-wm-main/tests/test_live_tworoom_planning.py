import numpy as np
import torch
import experiments.run_live_tworoom_planning as tw_room

from experiments.run_live_tworoom_planning import (
    _candidate_bank,
    LocalEffectRankedCostModel,
    RiskGatedRGBPolicy,
    VisionGeometryCostModel,
    _vision_door_candidate_bank,
    _vision_door_point,
    _vision_goal_candidate_bank,
    _effect_policy_action,
)


class _FakeEnv:
    num_envs = 1


class _FakeModelPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def set_env(self, env):
        self.env = env

    def get_action(self, info_dict, **kwargs):
        return self.action.copy()


class _SequenceModelPolicy:
    def __init__(self, actions):
        self.actions = [np.asarray(action, dtype=np.float32) for action in actions]
        self.index = 0

    def set_env(self, env):
        self.env = env

    def get_action(self, info_dict, **kwargs):
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return action.copy()


def _red_frame(x: int, y: int = 7) -> np.ndarray:
    frame = np.full((16, 16, 3), 255, dtype=np.uint8)
    frame[y:y + 2, x:x + 2] = [255, 0, 0]
    return frame


def test_tworoom_legacy_and_angular16_banks_are_deterministic_and_distinct() -> None:
    mean = np.zeros(10, dtype=np.float32)
    std = np.ones(10, dtype=np.float32)
    legacy = _candidate_bank(mean, std, horizon=5, action_block=5, variant="legacy")
    angular = _candidate_bank(mean, std, horizon=5, action_block=5, variant="angular16")
    assert legacy.shape == (17, 5, 10)
    assert angular.shape == (33, 5, 10)
    assert torch.isfinite(legacy).all() and torch.isfinite(angular).all()
    assert not torch.equal(legacy, angular[: len(legacy)])


def test_tworoom_candidate_bank_rejects_unknown_variant() -> None:
    with np.testing.assert_raises(ValueError):
        _candidate_bank(
            np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
            horizon=5, action_block=5, variant="unknown",
        )


def test_tworoom_vision_goal_bank_has_root_local_shape() -> None:
    image = torch.full((2, 3, 8, 8), 2.4)
    goal = image.clone()
    image[:, 0, 2:4, 1:3] = 2.4
    image[:, 1:, 2:4, 1:3] = -2.0
    goal[:, 0, 5:7, 5:7] = 2.4
    goal[:, 1:, 5:7, 5:7] = -2.0
    bank = _vision_goal_candidate_bank(
        image, goal, horizon=5, action_block=5,
        action_mean=np.zeros(10, dtype=np.float32),
        action_std=np.ones(10, dtype=np.float32),
    )
    assert bank.shape == (2, 17, 5, 10)
    assert torch.isfinite(bank).all()


def test_tworoom_vision_door_bank_has_two_stage_shape() -> None:
    image = torch.full((2, 3, 8, 8), 2.4)
    goal = image.clone()
    image[:, 0, 1:3, 1:3] = 2.4
    image[:, 1:, 1:3, 1:3] = -2.0
    goal[:, 0, 5:7, 5:7] = 2.4
    goal[:, 1:, 5:7, 5:7] = -2.0
    bank = _vision_door_candidate_bank(
        image, goal, horizon=5, action_block=5,
        action_mean=np.zeros(10, dtype=np.float32),
        action_std=np.ones(10, dtype=np.float32),
    )
    assert bank.shape == (2, 17, 5, 10)
    assert torch.isfinite(bank).all()


def test_tworoom_vision_door_point_ignores_wall_exterior() -> None:
    image = torch.full((1, 3, 8, 8), 2.4)
    image[:, :, :, 3:5] = -2.0
    image[:, :, 3:5, 3:5] = 2.4
    door, vertical = _vision_door_point(image)
    assert bool(vertical[0])
    torch.testing.assert_close(door[0], torch.tensor([4.0, 4.0]))


def test_tworoom_vision_geometry_cost_is_finite() -> None:
    image = torch.full((1, 1, 3, 8, 8), 2.4)
    goal = image.clone()
    image[:, :, 0, 2:4, 1:3] = 2.4
    image[:, :, 1:, 2:4, 1:3] = -2.0
    goal[:, :, 0, 5:7, 5:7] = 2.4
    goal[:, :, 1:, 5:7, 5:7] = -2.0
    info = {"pixels": image[:, None], "goal": goal[:, None]}
    model = VisionGeometryCostModel(
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32), 5,
    )
    costs = model.get_cost(info, torch.zeros(1, 4, 5, 10))
    assert costs.shape == (1, 4)
    assert torch.isfinite(costs).all()


def test_tworoom_rgb_gate_replaces_conflicting_action_without_state() -> None:
    current = np.full((16, 16, 3), 255, dtype=np.uint8)
    goal = current.copy()
    current[7:9, 2:4] = [255, 0, 0]
    goal[7:9, 12:14] = [255, 0, 0]
    policy = RiskGatedRGBPolicy(_FakeModelPolicy([[-1.0, 0.0]]))
    policy.set_env(_FakeEnv())
    action = policy.get_action({"pixels": current[None], "goal": goal[None]})
    assert action.shape == (1, 2)
    assert action[0, 0] > 0.0
    assert policy.fallback_count == 1


def test_tworoom_rgb_gate_keeps_agreeing_action() -> None:
    current = np.full((16, 16, 3), 255, dtype=np.uint8)
    goal = current.copy()
    current[7:9, 2:4] = [255, 0, 0]
    goal[7:9, 12:14] = [255, 0, 0]
    policy = RiskGatedRGBPolicy(_FakeModelPolicy([[1.0, 0.0]]))
    policy.set_env(_FakeEnv())
    action = policy.get_action({"pixels": current[None], "goal": goal[None]})
    np.testing.assert_allclose(action, [[1.0, 0.0]])
    assert policy.fallback_count == 0


def test_tworoom_phase_gate_fails_closed_during_door_phase(monkeypatch) -> None:
    current = np.full((16, 16, 3), 255, dtype=np.uint8)
    goal = current.copy()
    policy = RiskGatedRGBPolicy(
        _FakeModelPolicy([[1.0, 0.0]]), door_phase_reactive=True,
    )
    policy.set_env(_FakeEnv())
    monkeypatch.setattr(
        tw_room,
        "_rgb_reactive_action",
        lambda _current, _goal: (np.asarray([1.0, 0.0], dtype=np.float32), "door"),
    )
    action = policy.get_action({"pixels": current[None], "goal": goal[None]})
    np.testing.assert_allclose(action, [[1.0, 0.0]])
    assert policy.fallback_count == 1


def test_tworoom_door_phase_gate_cannot_be_overridden_by_effect_policy(monkeypatch) -> None:
    current = np.full((16, 16, 3), 255, dtype=np.uint8)
    goal = current.copy()
    policy = RiskGatedRGBPolicy(
        _FakeModelPolicy([[1.0, 0.0]]),
        door_phase_reactive=True,
        use_effect_policy=True,
    )
    policy.set_env(_FakeEnv())
    monkeypatch.setattr(
        tw_room,
        "_rgb_reactive_action",
        lambda _current, _goal: (np.asarray([1.0, 0.0], dtype=np.float32), "door"),
    )
    monkeypatch.setattr(
        policy,
        "_effect_map",
        lambda _index, _phase: (
            np.asarray([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32), 2
        ),
    )
    monkeypatch.setattr(
        policy,
        "_effect_map_quality",
        lambda _index, _phase: {"usable": True, "heldout_usable": True},
    )
    action = policy.get_action({"pixels": current[None], "goal": goal[None]})
    # The inverse local map would point in the negative x direction.  The
    # declared door-phase fail-closed rule must keep the RGB reactive action.
    np.testing.assert_allclose(action, [[1.0, 0.0]])
    assert policy.effect_policy_count == 0
    assert policy.effect_policy_phase_block_count == 1
    assert policy.fallback_count == 1


def test_tworoom_rgb_effect_map_rejects_nonprogressing_model_effect() -> None:
    goal = _red_frame(12)
    # The observed local chart has a deliberately reversed x effect: positive
    # x action produces negative marker displacement.  The final model action
    # points visually toward the goal, so only the effect-map gate can reject
    # it for predicting non-progress.
    frames = [
        _red_frame(10), _red_frame(9), _red_frame(9, 8), _red_frame(8, 8),
        _red_frame(8, 9), _red_frame(7, 9), _red_frame(7, 10),
    ]
    model = _SequenceModelPolicy([
        [[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 1.0]],
        [[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]],
    ])
    policy = RiskGatedRGBPolicy(model, use_effect_map=True)
    policy.set_env(_FakeEnv())
    for frame in frames[:-1]:
        policy.get_action({"pixels": frame[None], "goal": goal[None]})
    action = policy.get_action({"pixels": frames[-1][None], "goal": goal[None]})
    assert policy.effect_map_rows == [6]
    assert policy.effect_map_rank == [2]
    assert policy.effect_map_quality[0]["usable"] is True
    assert action[0, 0] > 0.0
    assert policy.fallback_count >= 1


def test_effect_policy_inverts_verified_local_map_and_fails_closed() -> None:
    action = _effect_policy_action(
        np.asarray([1.0, 0.0]),
        np.asarray([[-2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    assert action is not None
    assert action[0] < 0.0
    predicted = np.asarray(action, dtype=np.float64) @ np.asarray(
        [[-2.0, 0.0], [0.0, 1.0]], dtype=np.float64,
    )
    assert predicted[0] > 0.0
    assert _effect_policy_action(
        np.asarray([1.0, 0.0]), np.zeros((2, 2), dtype=np.float32),
    ) is None


def test_tworoom_effect_maps_are_root_local() -> None:
    policy = RiskGatedRGBPolicy(_FakeModelPolicy([[1.0, 0.0], [1.0, 0.0]]), use_effect_map=True)
    env = _FakeEnv()
    env.num_envs = 2
    policy.set_env(env)
    policy.effect_actions_by_env[0] = [
        np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
        np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
    ]
    policy.effect_deltas_by_env[0] = [
        np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
        np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
    ]
    policy.effect_actions_by_env[1] = [
        np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
        np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
    ]
    policy.effect_deltas_by_env[1] = [
        np.asarray([-1.0, 0.0]), np.asarray([0.0, 1.0]),
        np.asarray([-1.0, 0.0]), np.asarray([0.0, 1.0]),
    ]
    map0, rank0 = policy._effect_map(0)
    map1, rank1 = policy._effect_map(1)
    assert rank0 == 2 and rank1 == 2
    assert map0 is not None and map1 is not None
    assert map0[0, 0] > 0.0
    assert map1[0, 0] < 0.0


def test_tworoom_effect_maps_do_not_borrow_between_phases() -> None:
    policy = RiskGatedRGBPolicy(_FakeModelPolicy([[1.0, 0.0]]), use_effect_map=True)
    policy.set_env(_FakeEnv())
    for phase, sign in (("door", -1.0), ("goal", 1.0)):
        policy.effect_actions_by_env_phase[0][phase] = [
            np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
            np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
            np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
        ]
        policy.effect_deltas_by_env_phase[0][phase] = [
            np.asarray([sign, 0.0]), np.asarray([0.0, 1.0]),
            np.asarray([sign, 0.0]), np.asarray([0.0, 1.0]),
            np.asarray([sign, 0.0]), np.asarray([0.0, 1.0]),
        ]
    door_map, door_rank = policy._effect_map(0, "door")
    goal_map, goal_rank = policy._effect_map(0, "goal")
    assert door_rank == 2 and goal_rank == 2
    assert door_map is not None and goal_map is not None
    assert door_map[0, 0] < 0.0 < goal_map[0, 0]


def test_tworoom_affine_effect_map_removes_common_motion_from_slope() -> None:
    actions = [
        np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
        np.asarray([-1.0, 0.0]), np.asarray([0.0, -1.0]),
        np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
    ]
    slope = np.asarray([[2.0, 0.0], [0.0, 3.0]])
    intercept = np.asarray([5.0, -4.0])
    deltas = [action @ slope + intercept for action in actions]
    linear = RiskGatedRGBPolicy(_FakeModelPolicy([[1.0, 0.0]]), use_effect_map=True)
    affine = RiskGatedRGBPolicy(
        _FakeModelPolicy([[1.0, 0.0]]), use_effect_map=True,
        effect_map_intercept=True,
    )
    linear.set_env(_FakeEnv())
    affine.set_env(_FakeEnv())
    linear.effect_actions_by_env[0] = actions
    linear.effect_deltas_by_env[0] = deltas
    affine.effect_actions_by_env[0] = actions
    affine.effect_deltas_by_env[0] = deltas
    linear_map, _ = linear._effect_map(0)
    affine_map, _ = affine._effect_map(0)
    assert linear_map is not None and affine_map is not None
    assert not np.allclose(linear_map, slope)
    np.testing.assert_allclose(affine_map, slope, atol=1.0e-6)


def test_local_effect_ranker_records_when_it_changes_the_candidate_argmin() -> None:
    class Base:
        def get_cost(self, _info, _actions):
            return torch.tensor([[0.0, 0.1]], dtype=torch.float32)

    ranker = LocalEffectRankedCostModel(
        Base(), np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32), 1.0,
    )
    actions = torch.tensor([[[[1.0, 0.0]], [[-1.0, 0.0]]]])
    info = {
        "_local_effect_map": np.asarray([[[[-1.0, 0.0], [0.0, 1.0]],
                                           [[-1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32),
        "_local_effect_target": np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32),
        "_local_effect_valid": np.asarray([[True, True]]),
        "_candidate_bank_mask": np.asarray([[True, True]]),
    }
    costs = ranker.get_cost(info, actions)
    assert int(torch.argmin(costs, dim=1).item()) == 1
    diagnostics = ranker.runtime_diagnostics()
    assert diagnostics["local_effect_ranker_decisions"] == 1
    assert diagnostics["local_effect_ranker_argmin_changes"] == 1
