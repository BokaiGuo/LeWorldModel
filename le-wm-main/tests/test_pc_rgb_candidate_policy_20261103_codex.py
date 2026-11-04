import pytest

from experiments.spectral_reachability.pc_rgb_candidate_policy_20261103_codex import build_fixed_bank, select_candidates


def candidate(name, source, score, prefix=None):
    return {"name": name, "source": source, "prefix": prefix or [[0., 0.]]*5,
            "prefix_score": 1., "terminal_score": score, "suffix": [[0., 0.]]*10}


def test_selection_does_not_force_nonprogress():
    zero = candidate("zero", "fixed_action", 2.)
    progress = candidate("progress", "rgb_model", 1., [[0.5, 0.]]*5)
    progress["prefix_score"] = .5
    result = select_candidates([zero, progress], 1.)
    assert result["selected_name"] == "progress"
    assert result["abstained_reconfiguration"]


def test_privileged_and_outcome_fields_fail_closed():
    zero = candidate("zero", "fixed_action", 2.)
    with pytest.raises(ValueError):
        select_candidates([zero, candidate("direct", "privileged_physics", 0.)], 1.)
    leaked = candidate("random", "seeded_random", 1.)
    leaked["physical_distance"] = 0.
    with pytest.raises(ValueError):
        select_candidates([zero, leaked], 1.)


def test_bank_is_reproducible_and_contains_no_privileged_direct():
    bank = build_fixed_bank(20261103)
    assert bank == build_fixed_bank(20261103)
    assert len(bank) == 11
    assert [entry["name"] for entry in bank] == (
        ["zero"] + [f"direction_{i}" for i in range(8)] + ["random_0", "random_1"]
    )
    assert all(entry["source"] in {"fixed_action", "seeded_random"} for entry in bank)
    assert all(abs(value) <= 1.000001 for entry in bank for action in entry["prefix"] for value in action)
    with pytest.raises(ValueError):
        build_fixed_bank(-1)
