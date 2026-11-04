"""Pure arithmetic and saved-artifact checks for fixed-pair goal-Q."""
import copy
import json

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import audit_reacher_fixed_pair_goal_q_20261102 as audit
from experiments.spectral_reachability import reacher_fixed_pair_goal_q_20261102 as run


def test_linear_solve_matches_original_q_abi():
    gram = np.array([[2., .1], [.1, .3]], dtype=np.float64)
    target = np.array([100.2, -.1], dtype=np.float32)
    values = np.array([[100.3, -.5], [100.4, .2]], dtype=np.float32)
    metric = run.GoalEncoderMetric(torch.tensor(gram))
    expected = run.q.q_costs(metric, torch.tensor(values), torch.tensor(target)).numpy()
    np.testing.assert_allclose(audit.score(gram, values, target), expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="invalid"):
        audit.score(np.array([[1., 2.], [0., 1.]]), values, target)


def test_full_saved_result_and_tampered_score():
    plan = json.loads((run.ROOT / "results/reacher_fixed_pair_goal_q_PLAN_20261102_v1.json").read_text())
    report = json.loads((run.ROOT / "results/reacher_fixed_pair_goal_q_RUN_20261102_v1.json").read_text())
    replay = json.loads(run.old.RECEIPT.read_text())["rows"]
    original = json.loads(run.old.PRED.read_text())["rows"]
    grams = {r["seed"]: r["goal_metric"]["gram"] for r in json.loads(run.preflight.Q_PATH.read_text())["rows"]}
    digest = "7e7de463479f53f0bf564f15350cdfe706266917e3190d73eb1537425c919ce9"
    assert sum(x["pairs"] for x in audit.validate(plan, report, replay, original, grams, digest)["roots"].values()) == 26
    bad = copy.deepcopy(report)
    bad["rows"][0]["predicted_q"][0] += .1
    with pytest.raises(ValueError, match="Q score mismatch"):
        audit.validate(plan, bad, replay, original, grams, digest)
