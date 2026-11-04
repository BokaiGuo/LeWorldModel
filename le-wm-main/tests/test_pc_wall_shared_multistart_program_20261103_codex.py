import numpy as np
import pytest

from experiments.spectral_reachability.pc_wall_shared_multistart_program_20261103_codex import (
    NAMES, make_bank, program, run, select,
)
from experiments.spectral_reachability.pc_wall_unicycle_atlas_cem_20261103 import HIGH, LOW, replay


def test_program_bounds_and_input_validation():
    for horizon in (12, 16):
        for values in ([0., 0., 0., 0.], [-1., 1., 1., -1.], [1., 1., -1., 1.]):
            actions = program(values, horizon)
            assert actions.shape == (horizon, 2)
            assert np.all(actions >= LOW) and np.all(actions <= HIGH)
    with pytest.raises(ValueError):
        program([0., 0., 0., 1.1], 12)


def test_selected_and_qualified_are_distinct_and_abstention():
    result = select([.33, .1, .7, .1, .4, .2], [.6, .7, .55, .8, .54, .5], .33)
    assert result["selected_index"] == 5
    assert result["qualified_reconfiguration_index"] == 4
    assert not result["selected_is_qualified"]
    assert not result["abstained_reconfiguration"]
    result = select([.33, .1, .7, .1, .4, .2], [.5, .6, .7, .8, .7, .8], .33)
    assert result["selected_index"] == 0
    assert result["qualified_reconfiguration_index"] is None
    assert result["abstained_reconfiguration"]


def test_candidate_binding_own_suffix_replay_and_counts():
    seed = 20261103
    bank, _, costs, root_cost, generation, _ = make_bank(seed)
    result = run(seed)
    assert result["generation_backend"] == generation
    assert result["root_cost"] == root_cost
    for index, item in enumerate(result["choices"]):
        assert item["name"] == NAMES[index]
        assert item["prefix_cost"] == costs[index]
        actions = np.asarray(item["replay"]["actions"])
        np.testing.assert_array_equal(actions[:4], bank[index])
        np.testing.assert_array_equal(actions[4:], item["suffix"])
        assert replay(actions) == item["replay"]
        assert item["backend"]["transitions"] == 2448
        assert item["backend"]["rollouts"] == 204
        assert len(item["trace"]) == 6
        assert all(len(batch["samples"]) == 34 for batch in item["trace"])
    for item in result["direct_controls"].values():
        assert replay(item["replay"]["actions"]) == item["replay"]
        assert item["backend"]["transitions"] == 14688
        assert item["backend"]["rollouts"] == 918
    expected = select(costs, [x["replay"]["terminal_error"] for x in result["choices"]], root_cost)
    assert result["decision"] == expected


def test_paired_base_noise_on_first_round():
    result = run(20261104)
    expected = result["choices"][0]["trace"][0]["samples"]
    for choice in result["choices"][1:]:
        np.testing.assert_array_equal(choice["trace"][0]["samples"], expected)
