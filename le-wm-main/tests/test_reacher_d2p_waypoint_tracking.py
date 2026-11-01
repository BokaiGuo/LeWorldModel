"""Data/frontier contracts only, not CPU neural evidence."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_waypoint_tracking import initial_plan, validate_stage


def choice():
    return {"native_decision": {"selected_program_index": 1, "same_bank_greedy_index": 2},
        "word_params": np.zeros((3, 4, 10)).tolist(), "cem_params": np.zeros((1, 6, 10)).tolist(),
        "word_records": [{"index": i, "planned_commands": np.full((1, 2, 10), i / 2).tolist()} for i in range(3)]}


@pytest.mark.parametrize("family,tail", [("native_dispatch", .5), ("greedy", 1), ("zero", 0), ("cem", 0)])
def test_only_initial_plan_is_used(family, tail):
    data = choice()
    data["native_commands"] = "observed suffix must never be read"
    value = initial_plan(data, family, device="cpu").numpy()
    assert value.shape == (1, 6, 10)
    assert (value[:, :4] == 0).all()
    assert (value[:, 4:] == tail).all()


@pytest.mark.parametrize("mutation", ["index", "box", "shape", "nan", "family"])
def test_initial_plan_rejects_invalid(mutation):
    data = choice()
    family = "native_dispatch"
    if mutation == "index":
        data["word_records"][1]["index"] = 0
    elif mutation == "box":
        data["word_records"][1]["planned_commands"][0][0][0] = 1.0001
    elif mutation == "shape":
        data["word_params"][1].pop()
    elif mutation == "nan":
        data["word_params"][1][0][0] = float("nan")
    else:
        family = "observed_suffix"
    with pytest.raises((ValueError, RuntimeError)):
        initial_plan(data, family, device="cpu")


def stage():
    return {"seed": 123, "family": "zero", "mode": "tracking", "packet": 3,
        "initial_selection_sha256": "initial", "parent_execution_sha256": "parent",
        "raw_steps_observed_before_selection": 10, "unexecuted_branch_observations_used": False,
        "raw_actions": np.zeros((5, 2)).tolist(), "target_vector": [0.] * 192}


def validate(value):
    validate_stage(value, seed=123, family="zero", mode="tracking", packet=3, initial_sha="initial", parent_sha="parent")


def test_stage_accepts_exact_causal_frontier():
    validate(stage())


@pytest.mark.parametrize("field,value", [("seed", 124), ("packet", 4), ("family", "cem"), ("mode", "replay"),
    ("initial_selection_sha256", "changed"), ("parent_execution_sha256", "changed"),
    ("raw_steps_observed_before_selection", 15), ("unexecuted_branch_observations_used", True),
    ("raw_actions", [[0, 0]] * 4), ("raw_actions", [[2, 0]] * 5),
    ("target_vector", [0] * 191), ("target_vector", [float("nan")] * 192)])
def test_stage_rejects_leaks_or_incomplete_values(field, value):
    receipt = copy.deepcopy(stage())
    receipt[field] = value
    with pytest.raises(ValueError):
        validate(receipt)
