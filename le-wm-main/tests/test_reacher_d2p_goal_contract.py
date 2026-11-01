"""Goal-contract fixtures only, not neural or physical efficacy evidence."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability.reacher_d2p_feasible_goal import feasible_goal_delta
from experiments.spectral_reachability.reacher_d2p_goal_contract import goal_wrapper, validate_goal_admission


@pytest.mark.parametrize("wrist", [-2.6, -1.0, 0.0, 2.6])
def test_recompute_full_offset(wrist):
    _, receipt = feasible_goal_delta([2.9, wrist], [.85, -.65], [False, True], [[0, 0], [-2.7925, 2.7925]])
    validate_goal_admission("joint-feasible-v1", receipt)


def test_legacy_is_explicit_and_cannot_claim_admission():
    validate_goal_admission("legacy-periodic-replay", None)
    with pytest.raises(ValueError):
        validate_goal_admission("legacy-periodic-replay", {})
    with pytest.raises(ValueError):
        goal_wrapper(None)


@pytest.mark.parametrize("kind", ["missing", "privileged", "clip", "flip", "target", "nan"])
def test_tampered_or_missing_admission_rejected(kind):
    _, receipt = feasible_goal_delta([2.9, -2.6], [.85, -.65], [False, True], [[0, 0], [-2.7925, 2.7925]])
    receipt = copy.deepcopy(receipt)
    if kind == "missing":
        receipt = None
    elif kind == "privileged":
        receipt["planning_uses_privileged_state"] = True
    elif kind == "clip":
        receipt["effective_delta"][1] = .1
    elif kind == "flip":
        receipt["flipped_joint_indices"] = []
    elif kind == "target":
        receipt["historical_target_qpos"][1] += .1
    else:
        receipt["feasible_target_qpos"][1] = np.nan
    with pytest.raises(ValueError):
        validate_goal_admission("joint-feasible-v1", receipt)


def test_v2_soft_source_receipt_is_explicit_recomputed_and_not_v1():
    _,receipt = feasible_goal_delta([-.7,2.82945144],[.65,-.65],[False,True],[[0,0],[-2.7925,2.7925]],allow_soft_source=True)
    validate_goal_admission("joint-feasible-v2-soft-source",receipt)
    with pytest.raises(ValueError): validate_goal_admission("joint-feasible-v1",receipt)
    receipt["source_limit_violation_radians"][1] = 0.
    with pytest.raises(ValueError): validate_goal_admission("joint-feasible-v2-soft-source",receipt)
