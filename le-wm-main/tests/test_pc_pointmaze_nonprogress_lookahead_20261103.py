"""Saved lookahead selection and time-index bindings; no CUDA or physics rerun."""
import json

import numpy as np
import pytest

from experiments.pc_pointmaze_nonprogress_lookahead_20261103 import (
    ARRAYS, BANK, BANK_SHA, GOAL, OUTPUT, PROTOCOL, ROOTS, SOURCE, SOURCE_SHA,
    select_lookahead, sha256_file,
)


def test_selection_fails_closed_and_ties():
    costs = np.full((9, 2), 2.)
    assert select_lookahead(costs, [], 2)[0] is None
    assert select_lookahead(costs, [1], 2)[0] is None
    costs[1] = 1.
    costs[3] = 1.
    assert select_lookahead(costs, [1, 3], 2)[0] == 1
    with pytest.raises(ValueError):
        select_lookahead(costs, [0], 2)
    with pytest.raises(ValueError):
        select_lookahead(np.full((9, 2), np.nan), [1], 2)


def test_saved_lookahead():
    report, bank_report = json.loads(OUTPUT.read_text()), json.loads(BANK.read_text())
    assert report["bank_sha256"] == sha256_file(BANK) == BANK_SHA
    assert report["source_sha256"] == sha256_file(SOURCE) == SOURCE_SHA
    assert report["protocol_sha256"] == sha256_file(PROTOCOL)
    assert report["arrays_sha256"] == sha256_file(ARRAYS)
    assert report["model_before"] == report["model_after"]
    assert [r["root"] for r in report["rows"]] == list(ROOTS)
    with np.load(ARRAYS, allow_pickle=False) as data, np.load(BANK.with_suffix(".npz"), allow_pickle=False) as bank:
        k = 0
        for row, parent in zip(report["rows"], bank_report["rows"]):
            root = row["root"]
            assert root == parent["root"]
            costs = data[f"r{root}_terminal_costs"]
            first = data[f"r{root}_first_costs"]
            assert first.shape == (9, 50)
            expected_first = np.repeat(bank[f"r{root}_scores"][:, None], 50, axis=1)
            np.testing.assert_allclose(first, expected_first, rtol=2e-5, atol=2e-7)
            assert data[f"r{root}_optimized_normalized_tails"].shape == (9, 10)
            selected, tail, best = select_lookahead(costs, parent["eligible"], parent["greedy"])
            assert [b["arm"] for b in row["branches"]] == ["lookahead", "immediate", "greedy", "zero"]
            assert [b["bank_index"] for b in row["branches"]] == [selected, parent["selected"], parent["greedy"], 0]
            np.testing.assert_array_equal(row["best_tail"], tail)
            np.testing.assert_array_equal(row["best_cost"], best)
            assert row["two_packet_samples"] == 450
            assert row["predicted_packet_transitions"] == 900
            for branch in row["branches"]:
                if branch["bank_index"] is None:
                    continue
                distance = np.linalg.norm(data["endpoints_eval_only"][k, :2]-GOAL[:2])
                assert np.isclose(branch["cp5_distance"], distance)
                k += 1
        assert k == report["physical_prefixes"]
