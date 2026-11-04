import json

import numpy as np
import pytest

from experiments import pc_rgb_successor_exact_controls_dev_20261103_codex as control


def test_direct_cem_has_zero_svd_and_bounded_tokens():
    task = control.prior.panel.tasks()[0]
    row = control.decide(task, 0, "full_cem", [])
    assert row["svd"] == 0
    assert row["tokens"] == 480
    assert row["abstained"] is None
    assert row["comparison_chain"] == row["selected"]["chain"]


def test_saved_controls_follow_predeclared_policy():
    if not control.AUDIT.exists():
        pytest.skip("controls not yet available")
    plans = json.loads(control.PLANS.read_text())
    audit = json.loads(control.AUDIT.read_text())
    prior = json.loads(control.prior.PLANS.read_text())
    assert plans["dependencies"] == control.dependencies()
    assert audit["plans_sha256"] == control.prior.sha(control.PLANS)
    assert len(plans["rows"]) == len(audit["rows"]) == 96
    for row in plans["rows"]:
        arm, root = row["arm"], row["root"]
        assert sum(batch["tokens"] for batch in row["trace"]) == row["tokens"] <= 480
        if arm == "full_cem":
            assert row["svd"] == 0 and row["abstained"] is None
            continue
        assert row["svd"] == 1
        assert row["first_bank"] == prior["rows"][root]["result"]["first_bank"]
        original = row["precomparison"]
        comparison = np.asarray(original["chain"]).copy()
        comparison[:5] = 0.
        np.testing.assert_array_equal(comparison, row["comparison_chain"])
        abstain = row["comparison_score"] <= original["terminal_score"]
        assert row["abstained"] == abstain
        np.testing.assert_array_equal(row["selected"]["chain"], comparison if abstain else original["chain"])
        if arm == "same_bank_greedy":
            distances = np.linalg.norm(control.prior.exact.endpoints(
                row["task"], np.asarray([x["prefix"] for x in row["first_bank"]], dtype=np.float32))
                - row["task"]["goal"], axis=1)
            np.testing.assert_allclose(distances, row["selection_scores"], atol=1e-12)
            assert row["bank"][0]["prefix"] == row["first_bank"][int(np.argmin(distances))]["prefix"]
        if arm in ("random_bridge", "lookahead"):
            rng = np.random.default_rng(2026121200 + root)
            bank = []
            for item in row["first_bank"]:
                for suffix in (np.zeros((5, 2)), rng.uniform(-1, 1, (5, 2))):
                    bank.append(np.concatenate((item["prefix"], suffix)))
            if arm == "random_bridge":
                np.testing.assert_allclose([item["prefix"] for item in row["bank"]], bank)
            else:
                scores = np.linalg.norm(control.prior.exact.endpoints(row["task"],
                    np.asarray(bank, dtype=np.float32)) - row["task"]["goal"], axis=1)
                np.testing.assert_allclose(scores, row["selection_scores"], atol=1e-12)
                np.testing.assert_allclose(row["bank"][0]["prefix"], bank[int(np.argmin(scores))])
