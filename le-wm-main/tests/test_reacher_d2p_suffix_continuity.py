import copy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import identity_join, choose_model_incumbent


def fixture():
    rows = [{"seed": seed, "source_episode": seed + 10, "source_start_row": seed * 100,
        "word_results": {f"word_{i}": {} for i in range(17)}} for seed in (1, 2)]
    reference = {"schema": "reacher_d2p_model_native_action_words.v1", "status": "completed",
        "gates": {"valid": True}, "sealed_protocol": {"root_seeds": [1, 2]}, "rows": rows}
    selection = {"rows": [{**copy.deepcopy(row), "word_params": [0] * 17} for row in reversed(rows)]}
    return reference, selection


def test_root_identity_not_array_position():
    reference, selection = fixture()
    assert [(a["seed"], b["seed"]) for a, b in identity_join(reference, selection)] == [(1, 1), (2, 2)]


@pytest.mark.parametrize("kind", ["duplicate", "missing", "source", "word", "empty_gate", "numeric_gate"])
def test_fail_closed_reference(kind):
    reference, selection = fixture()
    if kind == "duplicate":
        selection["rows"][1] = copy.deepcopy(selection["rows"][0])
    elif kind == "missing":
        selection["rows"].pop()
    elif kind == "source":
        selection["rows"][0]["source_episode"] += 1
    elif kind == "word":
        reference["rows"][0]["word_results"].pop("word_16")
    elif kind == "empty_gate":
        reference["gates"] = {}
    else:
        reference["gates"] = {"valid": 1}
    with pytest.raises(ValueError):
        identity_join(reference, selection)


def test_incumbent_uses_only_model_cost_and_cold_tie_break():
    assert choose_model_incumbent({"cold": 2., "warm": 1.}) == "warm"
    assert choose_model_incumbent({"cold": 1., "warm": 2.}) == "cold"
    assert choose_model_incumbent({"cold": 1., "warm": 1.}) == "cold"


@pytest.mark.parametrize("costs", [{}, {"cold": 0., "warm": float("nan")}, {"cold": 0., "warm": 1., "physical": -2.}])
def test_incumbent_rejects_missing_nonfinite_or_outcome_arguments(costs):
    with pytest.raises(ValueError):
        choose_model_incumbent(costs)
