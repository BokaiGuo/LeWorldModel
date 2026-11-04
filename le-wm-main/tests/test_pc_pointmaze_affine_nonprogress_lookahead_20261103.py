import json
from pathlib import Path

import numpy as np
import pytest

from experiments import pc_pointmaze_affine_nonprogress_lookahead_20261103 as lookahead
from experiments import pc_pointmaze_affine_nonprogress_oracle_20261103 as oracle
from experiments.pc_pointmaze_affine_nonprogress_lookahead_20261103 import choose
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_selector_uses_future_cost_and_requires_both_controls():
    first = np.array([1.0, 0.6, 1.1, 1.2, 1.3, 0.5, 0.4, 0.3, 0.2])
    terminal = np.full((9, 2), 1.0)
    terminal[0] = [.8, .7]
    terminal[8] = [.6, .5]
    terminal[2] = [.4, .35]
    terminal[3] = [.3, .2]
    selected, greedy, eligible = choose(first, 1.0, terminal)
    assert selected == 3
    assert greedy == 8
    assert eligible == [2, 3, 4]
    terminal[3] = [.5, .5]
    terminal[2] = [.5, .5]
    assert choose(first, 1.0, terminal)[0] is None  # greedy tie fails closed


def test_selector_ties_use_lowest_bank_index_and_no_forced_choice():
    first = np.ones(9)
    terminal = np.ones((9, 2))
    assert choose(first, 1.0, terminal) == (None, 0, list(range(1, 9)))
    terminal[2] = [.1, .2]
    terminal[3] = [.1, .2]
    assert choose(first, 1.0, terminal)[0] == 2
    assert choose(first, 2.0, terminal)[0] is None


@pytest.mark.parametrize("kind", ["badshape", "nonfinite", "nonfiniteinitial"])
def test_selector_rejects_invalid_data(kind):
    first, terminal, initial = np.ones(9), np.ones((9, 2)), 1.0
    if kind == "badshape":
        first = first[:-1]
    elif kind == "nonfinite":
        terminal[0, 0] = np.nan
    else:
        initial = np.inf
    with pytest.raises(ValueError):
        choose(first, initial, terminal)


def test_saved_selector_and_oracle_bindings_and_action_decoding():
    if not lookahead.OUTPUT.exists() or not oracle.OUTPUT.exists():
        pytest.skip("development results unavailable")
    selected = json.loads(lookahead.OUTPUT.read_text())
    physical = json.loads(oracle.OUTPUT.read_text())
    for module, report in ((lookahead, selected), (oracle, physical)):
        assert report["script_sha256"] == sha256_file(Path(module.__file__))
        assert report["protocol_sha256"] == sha256_file(module.PROTOCOL)
        assert report["arrays_sha256"] == sha256_file(module.ARRAYS)
    assert selected["model_before"] == selected["model_after"]
    assert physical["source_sha256"] == sha256_file(lookahead.OUTPUT)
    assert physical["trajectories"] == 36
    mean, std = np.asarray(selected["action_mean"], np.float32), np.asarray(selected["action_std"], np.float32)
    with np.load(lookahead.ARRAYS, allow_pickle=False) as a, \
            np.load(oracle.ARRAYS, allow_pickle=False) as b:
        for root_i, row in enumerate(selected["rows"]):
            root = row["root"]
            first, terminal = a[f"r{root}_first_scores"], a[f"r{root}_terminal_scores"]
            decision, greedy, eligible = choose(first, row["initial_affine_score"], terminal)
            assert (decision, greedy, eligible) == (row["selected"], row["greedy"], row["eligible"])
            tails = a[f"r{root}_lookahead_tails"]
            for candidate in range(9):
                index = 9 * root_i + candidate
                actual = physical["rows"][index]
                assert (actual["root"], actual["bank_index"]) == (root, candidate)
                expected_tail = np.clip(tails[candidate].reshape(5, 2) * std + mean, -1, 1).astype(np.float32)
                np.testing.assert_array_equal(b["sequences"][index, 5:], expected_tail)
                assert actual["model_terminal_score"] == pytest.approx(terminal[candidate].min())
                goal = oracle.GOAL[:2]
                assert actual["cp5_distance"] == pytest.approx(np.linalg.norm(b["cp5"][index, :2] - goal))
                assert actual["cp10_distance"] == pytest.approx(np.linalg.norm(b["cp10"][index, :2] - goal))
                assert actual["physical_nonprogress"] == (actual["cp5_distance"] >= actual["initial_distance"])
