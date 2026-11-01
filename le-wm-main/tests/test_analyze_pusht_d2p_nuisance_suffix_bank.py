"""NumPy-only contracts and optional saved-real-report tamper regressions."""
import copy
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import analyze_pusht_d2p_nuisance_suffix_bank as audit

REPORT = Path(__file__).resolve().parents[1]/"results/pusht_d2p_nuisance_suffix_bank4x16_v1_20261026_cuda/report.json"


def test_projection_independent_goal_span():
    goals = np.zeros((5, 192))
    goals[1:, :4] = np.eye(4)
    q, rank, _ = audit.projection(goals)
    assert rank == 4
    np.testing.assert_allclose(q, np.diag([0]*4+[1]*188), atol=1e-12)
    np.testing.assert_allclose(q@q, q, atol=1e-12)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_projection_rejects_nonfinite(value):
    goals = np.zeros((5, 192))
    goals[0, 0] = value
    with pytest.raises(ValueError):
        audit.projection(goals)


def test_join_order_independence_and_duplicates():
    rows = [{"root_id": r, "proposal_index": p} for r in range(4) for p in range(16)]
    assert audit.joined(rows) == audit.joined(list(reversed(rows)))
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="join"):
        audit.joined(rows)


@pytest.fixture
def real():
    if not REPORT.exists():
        pytest.skip("saved CUDA report not installed; no inference launched")
    return audit.read(REPORT)


def test_real_full_audit(real):
    result = audit.analyze(REPORT)
    assert result["verified_pairs"] == 64
    assert result["verified_branches"] == 128
    assert result["new_inference"] == 0
    assert result["P1_admission"] is False
    assert len(result["rows"]) == 64


@pytest.mark.parametrize("mutation", ["gate", "compute", "duplicate", "branch", "stage", "baseline"])
def test_real_report_tampering_fails(real, monkeypatch, mutation):
    changed = copy.deepcopy(real)
    if mutation == "gate":
        changed["gates"]["cuda_only"] = 1
    elif mutation == "compute":
        changed["actual_predict_samples"] -= 1
    elif mutation == "duplicate":
        changed["rows"][-1] = changed["rows"][0]
    elif mutation == "branch":
        del changed["rows"][0]["branches"]["nuisance_gn4"]
    elif mutation == "stage":
        changed["rows"][0]["branches"]["nuisance_gn4"]["stage_sha256"] = "0"*64
    else:
        changed["rows"][0]["branches"]["identity_gn4"]["final_rgb_sha256"] = "0"*64
    original_read = audit.read
    monkeypatch.setattr(audit, "read", lambda path: changed if Path(path) == REPORT else original_read(path))
    with pytest.raises((ValueError, AssertionError)):
        audit.analyze(REPORT)


@pytest.mark.parametrize("mutation", ["cost", "line_search", "spectrum", "physical", "causal"])
def test_numerical_tampering_fails(real, mutation):
    branch = copy.deepcopy(real["rows"][0]["branches"]["identity_gn4"])
    plan = audit.read(REPORT.parent/"stages/root_0.plan.json")
    if mutation == "cost":
        branch["forecast"]["predicted_metric_cost"] += 1
    elif mutation == "line_search":
        branch["forecast"]["nonlinear"]["iterations"][0]["after"] += 1
    elif mutation == "spectrum":
        branch["forecast"]["linear"]["jacobian_frobenius_norm"] += 1
    elif mutation == "physical":
        branch["final_pose"]["pose_cost"] += 1
    else:
        branch["forecast"]["physical_labels_used"] = True
    with pytest.raises((ValueError, AssertionError)):
        audit.numerical(branch, np.asarray(plan["goal_vector"]), np.eye(192))
