import json
import sys

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from experiments.a1_pointmaze_cp5_spatial_objective_20261102 import (
    ARRAYS, OUTPUT, ROOT, SOURCE, SOURCE_SHA, costs,
)
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_spatial_proprio_cost_matches_official_l2_objective():
    sys.path.insert(0, str(ROOT / "external/jepa-wms"))
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

    torch.manual_seed(3)
    visual = torch.randn(1, 8, 1, 2, 2, 3)
    proprio = torch.randn(1, 8, 1, 3)
    goal_visual = torch.randn(1, 1, 1, 2, 2, 3)
    goal_proprio = torch.randn(1, 1, 1, 3)
    prediction = TensorDict({"visual": visual, "proprio": proprio}, batch_size=[1, 8])
    goal = TensorDict({"visual": goal_visual, "proprio": goal_proprio}, batch_size=[1, 1])
    pooled, spatial, prop = costs(prediction, goal)
    target = TensorDict({"visual": goal_visual[0], "proprio": goal_proprio[0]}, batch_size=[1])
    official = ReprTargetDistMPCObjective({}, target, alpha=0.1)(prediction, actions=None)
    torch.testing.assert_close(spatial + 0.1 * prop, official)
    assert torch.all(pooled <= spatial + 1e-6)


def test_saved_spatial_scores_recompute():
    if not OUTPUT.exists():
        pytest.skip("run not complete")
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(SOURCE) == SOURCE_SHA == report["source_sha256"]
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    with np.load(ARRAYS, allow_pickle=False) as data:
        for row in report["rows"]:
            root = row["root"]
            gv, gp = data[f"r{root}_goal_visual"], data[f"r{root}_goal_proprio"]
            for branch in row["branches"]:
                key = f"r{root}_{branch['method']}"
                visual, prop = data[f"{key}_pred_visual"], data[f"{key}_pred_proprio"]
                actual = data[f"{key}_actual_visual"]
                spatial = ((visual - gv) ** 2).mean(axis=(1, 2, 3, 4))
                prop_cost = ((prop - gp) ** 2).reshape(8, -1).mean(axis=1)
                actual_cost = ((actual - gv) ** 2).mean(axis=(1, 2, 3, 4))
                np.testing.assert_allclose(spatial, branch["spatial_costs"], rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(prop_cost, branch["proprio_costs"], rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(actual_cost, branch["actual_spatial_costs"], rtol=1e-5, atol=1e-7)
                assert np.argmin(spatial) == branch["indices"]["spatial"]
                assert np.argmin(spatial + .1 * prop_cost) == branch["indices"]["spatial_proprio"]
