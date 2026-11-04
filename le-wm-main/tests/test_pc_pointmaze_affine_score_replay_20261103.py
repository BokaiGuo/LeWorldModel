import json
from pathlib import Path

import numpy as np
import pytest

from experiments import pc_pointmaze_affine_score_replay_20261103 as audit
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_saved_score_replay_and_all_candidate_orderings():
    if not audit.OUTPUT.exists():
        pytest.skip("saved CUDA replay not available")
    report = json.loads(audit.OUTPUT.read_text())
    assert report["script_sha256"] == sha256_file(Path(audit.__file__))
    assert report["protocol_sha256"] == sha256_file(audit.PROTOCOL)
    assert report["arrays_sha256"] == sha256_file(audit.ARRAYS)
    assert report["model_before"] == report["model_after"]
    assert report["prediction_batches"] == 112 and report["prediction_samples"] == 1792
    assert len(report["rows"]) == 16
    with np.load(audit.ARRAYS, allow_pickle=False) as saved:
        for row in report["rows"]:
            assert row["cp5_state_max_abs_error"] == 0.0
            source_path = audit.RANDOM if row["method"] == "random" else audit.SPECTRAL
            source = json.loads(source_path.read_text())
            prior = next(r for r in source["rows"] if r["root"] == row["root"])
            assert len(row["batches"]) == 7
            for entry in row["batches"]:
                phase, iteration = entry["phase"], entry["iteration"]
                prior_batch = prior[phase + "_trace"][iteration]
                actual = saved[f"r{row['root']}_{row['method']}_{phase}_{iteration}"]
                np.testing.assert_array_equal(actual, prior_batch["scores"])
                np.testing.assert_array_equal(np.argsort(actual, kind="stable")[:4],
                                              prior_batch["elite_indices"])
