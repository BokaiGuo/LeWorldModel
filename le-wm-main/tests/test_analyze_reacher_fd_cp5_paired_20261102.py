import copy

import pytest

from experiments.spectral_reachability.analyze_reacher_fd_cp5_paired_20261102 import analyze


def events():
    output = []
    for seed in (20263160, 20263161):
        arms = []
        for arm, cp5, final, action in (
                ("selected", 2.0, 1.0, [1]), ("parameter_zero", 3.0, 1.5, [2])):
            arms.append({"arm": arm, "status": "complete", "suffix_seed": seed + 5000,
                         "result": {"physical": {"initial_qpos_distance": 3.0,
                                                  "after_first_chunk_qpos_distance": cp5,
                                                  "final_qpos_distance": final},
                                    "action_witness_raw": [[0]] * 5 + [action]}})
        output.append({"payload": {"kind": "root_returned", "seed": seed, "result": {"arms": arms}}})
    return output


def test_relative_change_and_suffix():
    row = analyze(events())["roots"][0]
    assert row["selected_minus_zero_final"] == -0.5
    assert row["suffix_change_difference"] == 0.5
    assert not row["same_executed_suffix_raw"]


@pytest.mark.parametrize("mutation", ["missing", "failed", "initial"])
def test_invalid_pairs_rejected(mutation):
    value = copy.deepcopy(events())
    if mutation == "missing":
        value.pop()
    elif mutation == "failed":
        value[0]["payload"]["result"]["arms"][0]["status"] = "error"
    else:
        value[0]["payload"]["result"]["arms"][0]["result"]["physical"]["initial_qpos_distance"] = 4
    with pytest.raises(ValueError):
        analyze(value)
