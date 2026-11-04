import json

import numpy as np
import pytest

from experiments.a1_pointmaze_cp5_eight_tail_oracle_20261102 import (
    ARRAYS, HISTORY, HISTORY_SHA, OUTPUT, SOURCE, SOURCE_SHA, image_hash, reversals,
)
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_reversals_and_ties():
    assert reversals(range(8), range(7, -1, -1)) == {"reversed": 28, "ties": 0}
    assert reversals([1] * 8, range(8)) == {"reversed": 0, "ties": 28}
    with pytest.raises(ValueError):
        reversals(range(8), [0] * 7 + [np.nan])


def test_saved_candidates_recompute_metrics():
    if not OUTPUT.exists():
        pytest.skip("physical panel not yet complete")
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(SOURCE) == report["source_sha256"] == SOURCE_SHA
    assert sha256_file(HISTORY) == report["history_sha256"] == HISTORY_SHA
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert len(report["rows"]) == 4
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        for row in report["rows"]:
            root = row["root"]
            goal = arrays[f"r{root}_goal_latent"]
            for branch in row["branches"]:
                key = f"r{root}_{branch['method']}"
                latent, rgb, endpoint = (arrays[f"{key}_{name}"] for name in ("latent10", "rgb10", "endpoint"))
                assert rgb.shape == (8, 224, 224, 3)
                assert [image_hash(x) for x in rgb] == branch["rgb10_sha256"]
                np.testing.assert_allclose(((latent - goal) ** 2).mean(axis=1),
                                           branch["actual_latent_costs"], rtol=1e-6)
                distances = np.linalg.norm(endpoint[:, :2] - row["goal_position_eval_only"], axis=1)
                np.testing.assert_allclose(distances, branch["physical_distances"], atol=1e-12)
                assert np.argmin(distances) == branch["indices"]["physical_oracle"]
                assert np.argmin(branch["actual_latent_costs"]) == branch["indices"]["real_latent_oracle"]
                for name, scores in (("single", branch["single_costs"]),
                                     ("history", branch["history_costs"])):
                    assert reversals(scores, branch["actual_latent_costs"]) == branch[f"{name}_vs_actual"]
                assert reversals(branch["actual_latent_costs"], distances) == branch["actual_vs_physical"]
