"""Post-outcome development contract: one-packet vs two-packet cp5 choice."""
import json
from pathlib import Path
import numpy as np

from . import codex_p1_cp10_rgb_refresh_control_20261101_exclusive as base
from . import codex_p1_diag32_fixed_physical_20261101_exclusive as physical

ROOT = base.ROOT
NAME = "codex_p1_cp10_robust_feedback_20261101_exclusive"
OLD_PLAN = ROOT / "results/codex_p1_cp10_rgb_refresh_control_20261101_exclusive_PRESEAL_CODEX01.json"
OLD_SHA = "e15c1768d3cb32d02a42a78522331fa49beca4859534182a37a6583f37a94459"
ROLES = base.ROLES
POLICIES = ("nominal", "robust")
IDENTITY = base.IDENTITY
actions, choose, digest_actions, cp10_context = base.actions, base.choose, base.digest_actions, base.cp10_context


def select_cp5(policy, q1, q2):
    one = choose(q1)
    choose(q2)
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    return one


def select_cp10(policy, scores):
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (2, 5) or not np.isfinite(values).all() or policy not in POLICIES:
        raise ValueError("invalid variant scores or policy")
    if policy == "nominal":
        return choose(values[:, 0].tolist())
    return choose((values.mean(axis=1) + np.ptp(values, axis=1)).tolist())


def protocol():
    return {"cp5_rule": "both minimize one-packet MSE; hold wins ties",
            "cp10_rule": "nominal minimizes nominal MSE; robust minimizes mean of five MSE plus range; hold wins ties",
            "candidate_order": ["hold", "optimized"],
            "variants": "nominal, +/-0.05 both axes, +/-0.05 opposite axes, clipped raw [-1,1]",
            "neural_caps_per_complete_episode": {"actual_rgb_encode": 2, "action_encoder": 14, "predict": 14},
            "budget_note": "both arms compute 4 cp5 and 10 cp10 predictions; only original candidate dispatched",
            "feedback": "same timing and encoder; cp10 actual RGB and action history may differ by cp5 choice",
            "stop": "native stop/truncation or original RGB/canvas guard; no padding",
            "physical_future_used_for_selection": False, "outcome_observed_before_design": True,
            "cached_candidates_known_root_only": True, "prior_generation_cost_not_equalized": True,
            "independent_confirmation": False, "P1_admission": False}


def contract():
    require, sha = physical.require, physical.sha
    require(sha(OLD_PLAN) == OLD_SHA and json.loads(OLD_PLAN.read_bytes()) == base.contract(),
            "source plan/source drift")
    risk_path = ROOT / "results/codex_p1_cp10_risk_score_audit_20261101_exclusive_CODEX01.json"
    require(sha(risk_path) == "f26e3229e75499f22ecb238d4787531fc7d3c476fb84767ef49a99b013bf98f4",
            "risk audit drift")
    risk = json.loads(risk_path.read_bytes())
    require(all(sha(Path(v["path"])) == v["sha256"] for v in risk["files"].values()), "risk source drift")
    old = json.loads(OLD_PLAN.read_bytes())
    paths = [Path(__file__).resolve(),
             ROOT / "experiments/spectral_reachability/codex_p1_cp10_robust_feedback_runner_20261101_exclusive.py",
             ROOT / "experiments/spectral_reachability/codex_p1_cp10_robust_feedback_verify_20261101_exclusive.py",
             ROOT / "tests/test_codex_p1_cp10_robust_feedback_20261101_exclusive.py"]
    paths += [risk_path, ROOT / "experiments/spectral_reachability/codex_p1_cp10_risk_score_audit_20261101_exclusive.py",
              ROOT / "tests/test_codex_p1_cp10_risk_score_audit_20261101_exclusive.py"]
    return {"schema": NAME + ".contract.v1", "source_plan": str(OLD_PLAN), "source_plan_sha256": OLD_SHA,
            "roles": list(ROLES), "policies": list(POLICIES), "candidates": old["candidates"],
            "protocol": protocol(), "physical_caps_total": old["physical_caps_total"],
            "wall_seconds": old["wall_seconds"], "cuda_allocated_cap_bytes": old["cuda_allocated_cap_bytes"],
            "cuda_reserved_cap_bytes": old["cuda_reserved_cap_bytes"],
            "bindings": {str(p): sha(p) for p in paths},
            "independent_confirmation": False, "P1_admission": False}
