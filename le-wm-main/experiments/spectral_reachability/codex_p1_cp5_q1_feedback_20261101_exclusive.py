"""Post-outcome development contract: one-packet vs two-packet cp5 choice."""
import json
from pathlib import Path

from . import codex_p1_cp10_rgb_refresh_control_20261101_exclusive as base
from . import codex_p1_diag32_fixed_physical_20261101_exclusive as physical

ROOT = base.ROOT
NAME = "codex_p1_cp5_q1_feedback_20261101_exclusive"
OLD_PLAN = ROOT / "results/codex_p1_cp10_rgb_refresh_control_20261101_exclusive_PRESEAL_CODEX01.json"
OLD_SHA = "e15c1768d3cb32d02a42a78522331fa49beca4859534182a37a6583f37a94459"
ROLES = base.ROLES
POLICIES = ("q1_feedback", "q2_feedback")
IDENTITY = base.IDENTITY
actions, choose, digest_actions, cp10_context = base.actions, base.choose, base.digest_actions, base.cp10_context


def select_cp5(policy, q1, q2):
    # Validate both scores even though a policy uses only its assigned horizon.
    one, two = choose(q1), choose(q2)
    if policy == "q1_feedback":
        return one
    if policy == "q2_feedback":
        return two
    raise ValueError("unknown policy")


def select_cp10(scores):
    return choose(scores)


def protocol():
    return {"cp5_rule": "q1 arm minimizes one-packet MSE; q2 arm minimizes two-packet MSE; hold wins ties",
            "cp10_rule": "both minimize one-packet MSE over same candidate packet labels on own actual RGB; hold wins ties",
            "candidate_order": ["hold", "optimized"],
            "neural_caps_per_complete_episode": {"actual_rgb_encode": 2, "action_encoder": 6, "predict": 6},
            "budget_note": "q1 and q2 read from the same two-packet rollouts; both arms compute all scores",
            "feedback": "same timing and encoder; cp10 actual RGB and action history may differ by cp5 choice",
            "stop": "native stop/truncation or original RGB/canvas guard; no padding",
            "physical_future_used_for_selection": False, "outcome_observed_before_design": True,
            "cached_candidates_known_root_only": True, "prior_generation_cost_not_equalized": True,
            "independent_confirmation": False, "P1_admission": False}


def contract():
    require, sha = physical.require, physical.sha
    require(sha(OLD_PLAN) == OLD_SHA and json.loads(OLD_PLAN.read_bytes()) == base.contract(),
            "source plan/source drift")
    old = json.loads(OLD_PLAN.read_bytes())
    paths = [Path(__file__).resolve(),
             ROOT / "experiments/spectral_reachability/codex_p1_cp5_q1_feedback_runner_20261101_exclusive.py",
             ROOT / "experiments/spectral_reachability/codex_p1_cp5_q1_feedback_verify_20261101_exclusive.py",
             ROOT / "tests/test_codex_p1_cp5_q1_feedback_20261101_exclusive.py"]
    return {"schema": NAME + ".contract.v1", "source_plan": str(OLD_PLAN), "source_plan_sha256": OLD_SHA,
            "roles": list(ROLES), "policies": list(POLICIES), "candidates": old["candidates"],
            "protocol": protocol(), "physical_caps_total": old["physical_caps_total"],
            "wall_seconds": old["wall_seconds"], "cuda_allocated_cap_bytes": old["cuda_allocated_cap_bytes"],
            "cuda_reserved_cap_bytes": old["cuda_reserved_cap_bytes"],
            "bindings": {str(p): sha(p) for p in paths},
            "independent_confirmation": False, "P1_admission": False}
