"""Explicit task versions and fail-closed joint-goal receipts (not a policy)."""
from __future__ import annotations

import numpy as np

from experiments.run_live_reacher_planning import ReacherGoalImageWrapper
from experiments.spectral_reachability.reacher_d2p_feasible_goal import ReacherFeasibleGoalImageWrapper, ReacherSoftLimitGoalImageWrapper, feasible_goal_delta

GOAL_CONTRACTS = ("legacy-periodic-replay", "joint-feasible-v1", "joint-feasible-v2-soft-source")


def goal_wrapper(contract):
    if contract not in GOAL_CONTRACTS:
        raise ValueError("explicit supported goal contract required")
    return {"joint-feasible-v1":ReacherFeasibleGoalImageWrapper,
        "joint-feasible-v2-soft-source":ReacherSoftLimitGoalImageWrapper,
        "legacy-periodic-replay":ReacherGoalImageWrapper}[contract]


def validate_goal_admission(contract, receipt):
    """Recompute the full-offset reflection; no outcomes or policy inputs."""
    goal_wrapper(contract)
    if contract == "legacy-periodic-replay":
        if receipt is not None:
            raise ValueError("legacy replay must not claim feasible-goal admission")
        return
    if not isinstance(receipt, dict) or receipt.get("planning_uses_privileged_state") is not False:
        raise ValueError("explicit environment-only feasible-goal receipt required")
    limited = np.asarray(receipt["limited_joints"], dtype=bool)
    target = np.asarray(receipt["feasible_target_qpos"], dtype=float)
    effective = np.asarray(receipt["effective_delta"], dtype=float)
    root = target - effective
    if root.shape != (2,) or limited.shape != (2,):
        raise ValueError("two-hinge receipt required")
    root[~limited] = (root[~limited] + np.pi) % (2 * np.pi) - np.pi
    _, expected = feasible_goal_delta(root, receipt["requested_delta"], limited, receipt["ranges_radians"],
        allow_soft_source=contract=="joint-feasible-v2-soft-source")
    # Modulo arithmetic can round reconstructed shoulder metadata. All
    # categorical choices and full-magnitude deltas must still match exactly.
    numeric = {"historical_target_qpos", "feasible_target_qpos"}
    if contract=="joint-feasible-v2-soft-source": numeric.add("source_limit_violation_radians")
    if set(receipt) != set(expected) or any(receipt[k] != expected[k] for k in expected if k not in numeric):
        raise ValueError("goal-admission policy or full-offset reflection changed")
    if any(not np.allclose(receipt[k], expected[k], rtol=0, atol=1e-14) for k in numeric):
        raise ValueError("goal admission target changed")


def capture_goal_admission(env, contract):
    receipt = getattr(env, "last_goal_admission", None)
    validate_goal_admission(contract, receipt)
    if contract != "legacy-periodic-replay":
        physics = env.unwrapped.env.physics
        model = physics.model
        if (not np.array_equal(model.jnt_limited, receipt["limited_joints"])
                or not np.array_equal(model.jnt_range, receipt["ranges_radians"])
                or not np.allclose(env.unwrapped.env.task.target_qpos, receipt["feasible_target_qpos"], rtol=0, atol=1e-14)):
            raise ValueError("actual simulator target/limits differ from admitted goal")
    return receipt
