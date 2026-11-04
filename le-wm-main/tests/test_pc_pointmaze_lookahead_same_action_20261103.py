"""Saved same-action bindings and ranking checks; no physical/CUDA rerun."""
import json

import numpy as np
import pytest

from experiments.pc_pointmaze_lookahead_same_action_20261103 import (
    ARRAYS, BANK, BANK_SHA, GOAL, LOOKAHEAD, LOOKAHEAD_SHA, OUTPUT, PROTOCOL,
    ROOTS, SOURCE, SOURCE_SHA, pair_reversals, sha256_file,
)


def test_pair_reversals_ties_and_invalid():
    assert pair_reversals([0, 1, 1], [1, 0, 2]) == {"pairs": 3, "comparable": 2, "reversed": 1}
    with pytest.raises(ValueError):
        pair_reversals([np.nan], [0])


def test_saved_same_action():
    report = json.loads(OUTPUT.read_text())
    assert report["lookahead_sha256"] == sha256_file(LOOKAHEAD) == LOOKAHEAD_SHA
    assert report["bank_sha256"] == sha256_file(BANK) == BANK_SHA
    assert report["source_sha256"] == sha256_file(SOURCE) == SOURCE_SHA
    assert report["protocol_sha256"] == sha256_file(PROTOCOL)
    assert report["arrays_sha256"] == sha256_file(ARRAYS)
    assert report["model_before"] == report["model_after"]
    assert (report["physical_trajectories"], report["open_two_packet_predictions"],
            report["refresh_one_packet_predictions"], report["actual_rgb_encodings"]) == (36, 36, 36, 72)
    with np.load(ARRAYS, allow_pickle=False) as data, np.load(BANK.with_suffix(".npz"), allow_pickle=False) as bank:
        target = data["target_token"]
        assert [x["root"] for x in report["rows"]] == list(ROOTS)
        for i, row in enumerate(report["rows"]):
            root = row["root"]
            frames, actions = data[f"r{root}_frames"], data[f"r{root}_actions"]
            assert frames.shape == (9, 11, 224, 224, 3)
            assert actions.shape == (9, 10, 2) and np.isfinite(actions).all() and (np.abs(actions) <= 1).all()
            for j, b in enumerate(row["branches"]):
                assert b["index"] == j
                np.testing.assert_array_equal(actions[j, :5], np.repeat(bank[f"r{root}_bank"][j][None], 5, axis=0))
                np.testing.assert_array_equal(frames[j, 5], bank[f"r{root}_rgb5"][j])
                for field, token in (("open_cost", "open_tokens"), ("refresh_cost", "refresh_tokens"),
                                     ("actual_encoded_cost", "actual_tokens")):
                    expected = np.mean((data[f"r{root}_{token}"][j]-target)**2)
                    assert np.isclose(b[field], expected, rtol=2e-5, atol=2e-7)
                assert np.isclose(b["physical_distance"],
                                  np.linalg.norm(data["endpoints_eval_only"][9*i+j, :2]-GOAL[:2]))
            b = row["branches"]
            assert row["ranking"]["open_vs_actual"] == pair_reversals(
                [x["open_cost"] for x in b], [x["actual_encoded_cost"] for x in b])
            assert row["ranking"]["refresh_vs_actual"] == pair_reversals(
                [x["refresh_cost"] for x in b], [x["actual_encoded_cost"] for x in b])
            assert row["ranking"]["actual_vs_physical"] == pair_reversals(
                [x["actual_encoded_cost"] for x in b], [x["physical_distance"] for x in b])
