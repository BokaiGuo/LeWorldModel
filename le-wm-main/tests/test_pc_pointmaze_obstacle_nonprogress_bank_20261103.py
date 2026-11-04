"""Saved selection and geometry checks; no inference or physical rerun."""
import json

import numpy as np
import pytest

from experiments.pc_pointmaze_obstacle_nonprogress_bank_20261103 import (
    ARRAYS, GOAL, OUTPUT, PROTOCOL, ROOTS, SOURCE, SOURCE_SHA,
    choose_nonprogress, sha256_file,
)


def test_nonprogress_selection_edge_cases():
    assert choose_nonprogress(np.arange(9), 20) == (None, [], 0)
    assert choose_nonprogress(np.ones(9), 1) == (1, list(range(1, 9)), 0)
    with pytest.raises(ValueError):
        choose_nonprogress(np.full(9, np.nan), 0)


def test_saved_nonprogress_bank():
    report = json.loads(OUTPUT.read_text())
    assert report["source_sha256"] == sha256_file(SOURCE) == SOURCE_SHA
    assert report["protocol_sha256"] == sha256_file(PROTOCOL)
    assert report["arrays_sha256"] == sha256_file(ARRAYS)
    assert report["physical_executions"] == 36
    assert report["model_before"] == report["model_after"]
    assert [row["root"] for row in report["rows"]] == list(ROOTS)
    with np.load(ARRAYS, allow_pickle=False) as data:
        for i, row in enumerate(report["rows"]):
            root = row["root"]
            bank = data[f"r{root}_bank"]
            jac = data[f"r{root}_jacobian"]
            singular = data[f"r{root}_singular"]
            scores = data[f"r{root}_scores"]
            assert bank.shape == (9, 2)
            np.testing.assert_array_equal(bank[0], np.zeros(2))
            for j in range(2):
                direction = bank[1+4*j] / .3
                np.testing.assert_allclose(np.linalg.norm(direction), 1, atol=1e-6)
                np.testing.assert_allclose(np.linalg.norm(jac @ direction), singular[j], rtol=1e-5)
                np.testing.assert_allclose(bank[1+4*j:5+4*j],
                                           np.array([.3, -.3, .6, -.6])[:, None]*direction, atol=1e-7)
            assert data[f"r{root}_rgb5"].shape == (9, 224, 224, 3)
            np.testing.assert_array_equal(scores, row["scores"])
            selected, eligible, greedy = choose_nonprogress(scores, row["current_cost"])
            assert (selected, eligible, greedy) == (row["selected"], row["eligible"], row["greedy"])
            for j in range(9):
                actual = np.linalg.norm(data["cp5_eval_only"][9*i+j, :2]-GOAL[:2])
                assert np.isclose(actual, row["distances_eval_only"][j])
