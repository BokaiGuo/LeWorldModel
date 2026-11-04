import json
from pathlib import Path

import numpy as np
import torch

from experiments import a1_pointmaze_spectral_rank_decomposition_20261102 as run
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import DEFAULT_DATASET, sha256_file


def test_pair_margin_signs_and_ties():
    rows = run.pair_margins([0, 1, 2, 3, 4], [0, 2, 1, 3, 4], [0, 2, 1, 4, 3])
    assert len(rows) == 10
    pair = next(row for row in rows if (row["i"], row["j"]) == (1, 2))
    assert pair["predicted_encoded_reverse"]
    assert not pair["encoded_physical_reverse"]
    tied = run.pair_margins([0] * 5, [1] * 5, [2] * 5)
    assert all(row["predicted_encoded_tie"] and row["encoded_physical_tie"] for row in tied)


def test_saved_slots_and_recomputed_margins_bind_source():
    report = json.loads(run.OUTPUT.read_text())
    old = json.loads(run.SOURCE.read_text())
    assert report["source_sha256"] == sha256_file(run.SOURCE) == run.SOURCE_SHA
    assert report["protocol_sha256"] == sha256_file(run.PROTOCOL)
    assert report["code_sha256"] == sha256_file(Path(run.__file__))
    assert report["model_before"] == report["model_after"] == old["model_before"]
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    for row, source in zip(report["rows"], old["rows"]):
        assert row["root"] == source["root"]
        target = states[row["root"], 10, :2].astype(np.float64)
        for branch, earlier in zip(row["branches"], source["branches"]):
            assert branch["actions"] == earlier["candidate_actions"]
            assert branch["predicted_costs"] == earlier["model_costs"]
            assert branch["rgb_sha256"][branch["selected"]] == earlier["rgb5_sha256"]
            endpoints = np.asarray(branch["endpoints_eval_only"])
            np.testing.assert_allclose(np.linalg.norm(endpoints[:, :2] - target, axis=1),
                                       branch["physical_distances"], rtol=1e-12)
            assert branch["pairs"] == run.pair_margins(branch["predicted_costs"],
                                                       branch["actual_encoded_costs"],
                                                       branch["physical_distances"])
