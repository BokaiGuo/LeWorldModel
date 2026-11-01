"""Joint-limit-aware environment goal construction, never a deployment policy.

The historical wrapper wraps BOTH coordinates as periodic. The wrist is
instead a bounded hinge. Keep the requested displacement magnitude and
reflect its sign only when its proposed endpoint leaves the actual limits.
Invalid roots or offsets with neither endpoint feasible fail closed.
"""
from __future__ import annotations

import numpy as np

from experiments.run_live_reacher_planning import ReacherGoalImageWrapper


def feasible_goal_delta(root, delta, limited, ranges, *, allow_soft_source=False):
    root, delta, ranges = (np.asarray(v, dtype=np.float64) for v in (root, delta, ranges))
    limited = np.asarray(limited, dtype=bool)
    if (root.shape != (2,) or delta.shape != (2,) or limited.shape != (2,) or ranges.shape != (2, 2)
            or not np.isfinite(root).all() or not np.isfinite(delta).all() or not np.isfinite(ranges).all()):
        raise ValueError("finite two-hinge goal metadata required")
    effective, flipped = delta.copy(), []
    for j in range(2):
        if not limited[j]:
            continue
        low, high = ranges[j]
        if low >= high or (not allow_soft_source and not low <= root[j] <= high):
            raise ValueError("bounded source joint must start inside its physical range")
        if not low <= root[j] + delta[j] <= high:
            if not low <= root[j] - delta[j] <= high:
                raise ValueError("neither full-magnitude goal offset is physically feasible")
            effective[j] = -delta[j]
            flipped.append(j)
    historical = (root + delta + np.pi) % (2 * np.pi) - np.pi
    target = root + effective
    target[~limited] = (target[~limited] + np.pi) % (2 * np.pi) - np.pi
    receipt = {"policy": "actual_hinge_limits_full_offset_sign_reflection_v1",
        "limited_joints": limited.tolist(), "ranges_radians": ranges.tolist(), "requested_delta": delta.tolist(),
        "effective_delta": effective.tolist(), "flipped_joint_indices": flipped,
        "historical_target_qpos": historical.tolist(), "feasible_target_qpos": target.tolist(),
        "historical_target_outside_limits": [int(j) for j in range(2) if limited[j] and not ranges[j, 0] <= historical[j] <= ranges[j, 1]],
        "planning_uses_privileged_state": False,
        "scope": "Environment task generation metadata only; changes task, not retrospective P1 admission."}
    if allow_soft_source:
        receipt["policy"] = "actual_hinge_limits_full_offset_sign_reflection_soft_source_v2"
        receipt["source_limit_violation_radians"] = [float(max(ranges[j,0]-root[j],root[j]-ranges[j,1],0.)) if limited[j] else 0. for j in range(2)]
    return effective, receipt


class ReacherFeasibleGoalImageWrapper(ReacherGoalImageWrapper):
    """Explicit new task version; historical goal results remain immutable."""

    allow_soft_source = False

    def _render_target_and_restore(self, current_qpos, current_qvel):
        model = self._raw().env.physics.model
        if (int(model.nq) != 2 or not np.array_equal(np.asarray(model.jnt_qposadr), [0, 1])
                or not np.all(np.asarray(model.jnt_type) == 3)):
            raise ValueError("joint-aware goal wrapper requires exactly two scalar hinges")
        effective, receipt = feasible_goal_delta(current_qpos, self.target_delta, model.jnt_limited, model.jnt_range,
            allow_soft_source=self.allow_soft_source)
        configured = self.target_delta
        try:
            # Reuse the SAME render/restore transaction. Valid historical roots
            # retain the exact original delta, not a rounded goal-root inverse.
            self.target_delta = effective
            goal = super()._render_target_and_restore(current_qpos, current_qvel)
        finally:
            self.target_delta = configured
        actual = np.asarray(self._raw().env.task.target_qpos)
        if not np.allclose(actual, receipt["feasible_target_qpos"], atol=1e-14, rtol=0):
            raise RuntimeError("rendered target differs from admitted physical goal")
        self.last_goal_admission = receipt
        return goal
class ReacherSoftLimitGoalImageWrapper(ReacherFeasibleGoalImageWrapper):
    """Do not reject a real source state solely for soft-limit penetration.

    Never clip/alter the source; both candidate target endpoints still undergo
    exactly the nominal-range/full-displacement checks. Not historical v1.
    """
    allow_soft_source = True
