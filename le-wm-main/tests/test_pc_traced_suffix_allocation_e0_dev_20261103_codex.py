import json

import numpy as np
import pytest

from experiments import pc_traced_suffix_allocation_e0_dev_20261103_codex as experiment
from experiments.pc_budget_allocation_20261103_codex import optimize


def test_persistent_cem_reproduces_uninterrupted_optimizer():
    prefix = np.full((5, 2), .2)
    goal = np.array([.2, .7])
    def score(actions):
        positions = experiment.exact.rollout(actions)[:, :2]
        return positions, np.linalg.norm(positions - goal, axis=1)
    state = experiment.CEMState(prefix, 123)
    for size in (12, 12, 12, 5):
        state.step(size, score)
    chain, cost, _ = optimize(prefix, 15, 41, np.random.default_rng(123), lambda x: score(x)[1])
    np.testing.assert_array_equal(state.best, chain)
    assert state.score == cost and state.count == 41


@pytest.mark.parametrize("arm", experiment.ARMS)
@pytest.mark.parametrize("budget", experiment.BUDGETS)
def test_budget_and_trace_on_exact_model(arm, budget):
    row = experiment.decide(0, np.array([.9, 0.]), budget, arm)
    assert row["counts"]["prediction_token_samples"] <= budget
    assert sum(x["tokens"] for x in row["batches"]) == row["counts"]["prediction_token_samples"]
    assert len(row["selected"]["chain"]) == 15
    if arm == "uniform":
        counts = [x["count"] for x in row["candidates"]]
        assert max(counts) - min(counts) <= 1
    if arm == "adaptive":
        assert [x["candidate"] for x in row["allocation"][:3]] == [0, 1, 2]
        assert all(x["samples"] == 12 for x in row["allocation"][:3])
    if arm == "greedy":
        assert len(row["candidates"]) == 1


def test_sealed_panel_allocation_counts_and_prior_policy():
    if not experiment.AUDIT.exists():
        pytest.skip("development panel not yet audited")
    data = json.loads(experiment.PLANS.read_text())
    audit = json.loads(experiment.AUDIT.read_text())
    assert audit["plans_sha256"] == experiment.exact.sha(experiment.PLANS)
    assert data["dependencies"] == experiment.hashes()
    assert len(data["rows"]) == len(audit["rows"]) == 96
    previous = json.loads(experiment.exact.PLANS.read_text())
    for row in data["rows"]:
        assert row["counts"]["prediction_token_samples"] == row["budget"]
        for candidate in row["candidates"]:
            index = next(i for i, item in enumerate(row["bank"]) if item["name"] == candidate["name"])
            assert candidate["count"] == sum(x["samples"] for x in row["allocation"]
                                               if x["candidate"] == index)
        if row["budget"] == 480 and row["arm"] in ("uniform", "greedy", "cem"):
            arm = "svd" if row["arm"] == "uniform" else row["arm"]
            old = previous["rows"][row["root"]]["methods"][arm]["decision"]["selected"]
            np.testing.assert_allclose(row["selected"]["chain"], old["chain"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(row["selected"]["score"], old["terminal_score"], rtol=0, atol=1e-12)
