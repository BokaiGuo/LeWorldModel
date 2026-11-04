"""Translation edge semantics and saved-result integrity."""
import copy
import json

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import audit_reacher_goal_translation_subspace_20261102 as audit
from experiments.spectral_reachability import reacher_goal_translation_subspace_20261102 as run


def test_translation_extends_edges_without_wrapping():
    image = torch.arange(12, dtype=torch.float32).reshape(1, 1, 1, 3, 4).repeat(1, 1, 3, 1, 1)
    shifted = run.shift_goal(image, 1, -1)
    np.testing.assert_array_equal(shifted[0, 0, 0].numpy(),
                                  np.array([[1, 2, 3, 3], [1, 2, 3, 3], [5, 6, 7, 7]]))
    with pytest.raises(ValueError, match="outside"):
        run.shift_goal(image, 3, 0)


def test_saved_basis_and_score_tamper():
    plan = json.loads((run.ROOT / "results/reacher_goal_translation_subspace_PLAN_20261102_v1.json").read_text())
    report = json.loads((run.ROOT / "results/reacher_goal_translation_subspace_RUN_20261102_v1.json").read_text())
    original = json.loads(run.REPORT.read_text())["rows"]
    digest = "3b97a3d64a94526e2a58709c3292a527dd8b3032cfb4e3ea44d988c11fd7c584"
    assert sum(v["pairs"] for v in audit.validate(plan, report, original, digest)["roots"].values()) == 26
    changed = copy.deepcopy(report)
    changed["rows"][0]["model_score"][0] += 1.
    with pytest.raises(ValueError, match="score mismatch"):
        audit.validate(plan, changed, original, digest)
    changed = copy.deepcopy(report)
    changed["bases"][0]["basis"][0][0] += .1
    with pytest.raises(ValueError, match="invalid SVD basis"):
        audit.validate(plan, changed, original, digest)
