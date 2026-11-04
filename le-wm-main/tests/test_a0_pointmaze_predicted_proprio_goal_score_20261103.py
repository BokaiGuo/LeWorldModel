"""Validate saved development scoring and its outcome bindings."""
import json

import numpy as np
import torch

from experiments.a0_pointmaze_predicted_proprio_goal_score_20261103 import (
    ARRAYS, ORACLE, OUTPUT, PROTOCOL, pairwise_ranking, score, sha256_file,
)


def test_score_reduces_proprio_per_candidate():
    class Model:
        def unroll(self, context, act_suffix):
            values = act_suffix[0, :, 0]
            visual = torch.zeros(len(values), 1, 16, 16, 384)
            proprio = torch.stack((values, 2 * values), -1)[:, None]
            return {"visual": [visual], "proprio": [proprio]}

    proposals = np.array([[[0., 0.]], [[1., 0.]]], dtype=np.float32)
    goal = {"visual": torch.zeros(1, 1, 16, 16, 384),
            "proprio": torch.tensor([[[1., 2.]]])}
    visual, proprio = score(Model(), None, goal, np.ones((16, 16), dtype=bool),
                            proposals, torch.zeros(2), torch.ones(2))
    np.testing.assert_array_equal(visual, [0., 0.])
    np.testing.assert_array_equal(proprio, [2.5, 0.])


def test_saved_scores_and_outcome_binding():
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert report["model_before"] == report["model_after"]
    assert report["prediction_samples"] == 624
    assert len(report["rows"]) == 12
    with np.load(ARRAYS, allow_pickle=False) as saved, \
            np.load(ORACLE.with_suffix(".npz"), allow_pickle=False) as bank:
        for row in report["rows"]:
            key = f"r{row['root']}_{row['method']}_{row['phase']}"
            physical = bank[f"{key}_distances"]
            assert row["physical_oracle_distance"] == np.min(physical)
            for name in ("visual", "proprio"):
                values = saved[f"{key}_{name}"]
                assert values.shape == (row["n"],)
                assert row[f"{name}_choice"] == np.argmin(values)
                assert row[f"{name}_physical_distance"] == physical[row[f"{name}_choice"]]
                assert row[f"{name}_vs_physical"] == pairwise_ranking(values, physical)
