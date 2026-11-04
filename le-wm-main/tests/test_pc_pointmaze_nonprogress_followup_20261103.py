"""Saved followup evidence bindings; does not rerun inference or physics."""
import hashlib
import json

import numpy as np

from experiments.pc_pointmaze_nonprogress_followup_20261103 import (
    ARMS, ARRAYS, BANK, BANK_SHA, GOAL, OUTPUT, PROTOCOL, ROOTS, SOURCE, SOURCE_SHA, sha256_file,
)


def test_nonprogress_followup_saved():
    report, bank_report = json.loads(OUTPUT.read_text()), json.loads(BANK.read_text())
    assert report["bank_sha256"] == sha256_file(BANK) == BANK_SHA
    assert report["source_sha256"] == sha256_file(SOURCE) == SOURCE_SHA
    assert report["protocol_sha256"] == sha256_file(PROTOCOL)
    assert report["arrays_sha256"] == sha256_file(ARRAYS)
    assert report["model_before"] == report["model_after"]
    assert (report["feedback_trajectories"], report["rgb_replays"], report["cross_sequences"]) == (12, 12, 36)
    expected = [(r, p, s) for r in ROOTS for p in ARMS for s in ARMS]
    assert [(x["root"], x["prefix"], x["suffix"]) for x in report["cross_rows"]] == expected
    index = {key: i for i, key in enumerate(expected)}
    with np.load(ARRAYS, allow_pickle=False) as data, np.load(BANK.with_suffix(".npz"), allow_pickle=False) as bank:
        for i, (row, parent) in enumerate(zip(report["rows"], bank_report["rows"])):
            assert row["root"] == parent["root"] == ROOTS[i]
            assert [x["prefix"] for x in row["branches"]] == list(ARMS)
            for j, branch in enumerate(row["branches"]):
                root, arm = row["root"], branch["prefix"]
                idx = (parent["selected"], parent["greedy"], 0)[j]
                assert branch["bank_index"] == idx
                actions, frames = data[f"r{root}_{arm}_actions"], data[f"r{root}_{arm}_frames"]
                assert actions.shape == (25, 2) and frames.shape == (26, 224, 224, 3)
                assert np.isfinite(actions).all() and (np.abs(actions) <= 1).all()
                np.testing.assert_array_equal(actions[:5], np.repeat(bank[f"r{root}_bank"][idx][None], 5, axis=0))
                np.testing.assert_array_equal(frames[5], bank[f"r{root}_rgb5"][idx])
                assert len(branch["decisions"]) == 4
                for k, d in enumerate(branch["decisions"], 1):
                    assert d["input_rgb_sha256"] == hashlib.sha256(frames[5*k].tobytes()).hexdigest()
                    assert d["output_rgb_sha256"] == hashlib.sha256(frames[5*k+5].tobytes()).hexdigest()
                    assert (d["prediction_batches"], d["prediction_samples"]) == (4, 50)
                assert np.isclose(branch["cp5_distance"], np.linalg.norm(data["cp5_eval_only"][3*i+j, :2]-GOAL[:2]))
                assert np.isclose(branch["endpoint_distance"], np.linalg.norm(data["cp25_eval_only"][3*i+j, :2]-GOAL[:2]))
        for i, (root, prefix, suffix) in enumerate(expected):
            seq = data["cross_sequences"][i]
            np.testing.assert_array_equal(seq[:5], data[f"r{root}_{prefix}_actions"][:5])
            tail = np.zeros((20, 2)) if suffix == "zero" else data[f"r{root}_{suffix}_actions"][5:]
            np.testing.assert_array_equal(seq[5:], tail)
            end = data["cross_end_eval_only"][i, :2]
            zero = data["cross_end_eval_only"][index[root, prefix, "zero"], :2]
            row = report["cross_rows"][i]
            assert np.isclose(row["endpoint_distance"], np.linalg.norm(end-GOAL[:2]))
            np.testing.assert_allclose(row["displacement_minus_zero_suffix"], end-zero, atol=1e-12)
            assert np.isclose(row["distance_minus_zero_suffix"], np.linalg.norm(end-GOAL[:2])-np.linalg.norm(zero-GOAL[:2]))
