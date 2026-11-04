import json
from unittest.mock import patch

import numpy as np

from experiments import pc_rgb_successor_exact_controls_dev_20261103_codex as controls
from experiments import pc_rgb_successor_exact_same_policy_dev_20261103_codex as prior
from experiments.pc_budget_allocation_20261103_codex import optimize as old_optimize
from experiments.spectral_reachability import pc_successor_floor_compiler_v1 as compiler
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize


def test_old_optimizer_parity_with_partial_batch():
    def score(chains):
        return np.sum((chains - .2) ** 2, axis=(1, 2))
    for count in (1, 12, 13, 37):
        a = old_optimize(np.zeros((5, 2)), 15, count, np.random.default_rng(21), score)
        b = optimize(np.zeros((5, 2)), 15, count, np.random.default_rng(21), score, scale_floor=1e-5)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]


def test_compiler_old_floor_matches_sealed_all_arms():
    original = json.loads(prior.PLANS.read_text())["rows"][0]
    old_controls = [r for r in json.loads(controls.PLANS.read_text())["rows"] if r["root"] == original["root"]]

    def original_floor(*args, **kwargs):
        return optimize(*args, **kwargs, scale_floor=1e-5)

    with patch.object(compiler, "optimize", original_floor):
        for arm in compiler.ARMS:
            meter = prior.Budget()
            out = compiler.decide(arm, original["root"], np.asarray(original["task"]["goal"]),
                                  lambda actions: meter.predict(original["task"], actions), meter)
            expected = original if arm == "successor" else next(x for x in old_controls if x["arm"] == arm)
            old = expected["result"] if arm == "successor" else expected
            assert meter.token_samples == expected["tokens"]
            assert out["first_bank"] == old.get("first_bank", [])
            np.testing.assert_array_equal(out["selected"]["chain"], old["selected"]["chain"])
            assert out["selected"]["terminal_score"] == old["selected"]["terminal_score"]
