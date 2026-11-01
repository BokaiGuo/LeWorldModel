"""Prospective P1 boundary-prefix contract and CPU-only input checks.

No outcome reader, physical executor, model invocation, or gate mutation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from . import codex_p1_rgb_task_score_20261030_v2 as rgb

NAME = "p1_boundary_feedback_prereg_20261102_exclusive"
ROLES = ("selected", "zero", "greedy", "random")
TAILS = ("common", "own_feedback")
RESOURCE_AXES = ("rgb_encode", "action_encoder", "predict", "predictor_samples",
                 "candidate_generation", "selection", "physical_steps", "warmup_steps")
HEX = re.compile(r"^[0-9a-f]{64}$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    require(type(value) is str and HEX.fullmatch(value) is not None, "SHA256 required")
    return value


def _actions(value, steps):
    raw = np.asarray(value)
    require(raw.dtype == np.dtype("float32") and raw.shape == (steps, 2)
            and np.isfinite(raw).all() and (np.abs(raw) <= 1).all(),
            "finite bounded float32 actions")
    return np.ascontiguousarray(raw, dtype="<f4")


def action_sha(value, steps):
    return hashlib.sha256(_actions(value, steps).tobytes()).hexdigest()


def screen_rgb(image):
    """Fixed renderer-specific crop screen, not a proof of visibility."""
    image = rgb._image(image)
    details = {}
    for name, color, single in (("pusher", rgb.BLUE, True), ("object", rgb.BLOCK, False)):
        _, stats = rgb._mask(image, color, 8, single=single)
        x0, y0, x1, y1 = stats["bounds_xyxy"]
        margin = min(x0, y0, 223 - x1, 223 - y1)
        require(margin >= 8, f"{name} crop margin")
        details[name] = {"margin_pixels": margin, **stats}
    return {"pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "markers": details, "occlusion_free_certified": False}


def contract(*, bindings, prefixes, selected_index, common_tail, d2p_plan):
    """Construct a non-authorizing template from already sealed pre-action inputs.

    SHA values are commitments supplied by the caller; this function does not
    authenticate the referenced files or certify selection provenance.
    """
    required = {"source", "task", "model", "environment", "bank", "selector",
                "planner", "goal_rgb", "initial_rgb", "action_history"}
    require(type(bindings) is dict and set(bindings) == required, "complete bindings")
    bindings = {key: _sha(bindings[key]) for key in sorted(required)}
    require(type(prefixes) is list and len(prefixes) == 17, "17-candidate bank")
    bank = [action_sha(item, 5) for item in prefixes]
    require(type(selected_index) is int and 0 <= selected_index < 17, "candidate index")
    role_indices = {"selected": selected_index, "zero": 0,
                    "task_greedy_direct": 4, "random": 7}
    role_indices["greedy"] = role_indices["task_greedy_direct"]
    require(not np.any(_actions(prefixes[0], 5)), "zero candidate")
    common_sha = validate_common_tail(common_tail, d2p_plan)
    branches = [{"role": role, "candidate_index": role_indices[role],
                 "prefix_sha256": bank[role_indices[role]], "tail_mode": tail,
                 "common_tail_sha256": common_sha if tail == "common" else None}
                for role in ROLES for tail in TAILS]
    return {
        "schema": NAME + ".v1", "status": "draft_not_authenticated_not_authorized",
        "bindings_sha256": bindings, "candidate_sha256": bank,
        "role_indices": role_indices,
        "greedy_definition": "task_greedy_direct: rgb_goal_support_direct at fixed index 4",
        "latent_greedy": "diagnostic only; not a physical arm",
        "freeze_blockers": ["input_source_and_plan_hash_authentication",
                            "candidate_roster_and_selected_identity_authentication",
                            "feedback_runner_and_accounting_validation",
                            "optimizer_seed_and_initialization_freeze",
                            "separate_execution_authorization"],
        "duplicate_roles_retained": True, "branches": branches,
        "common_tail": {"source": "root D2P last five actions followed by five zeros",
                        "sha256": common_sha, "steps": 10,
                        "fixed_before_any_physical_action": True},
        "feedback": {"checkpoint": 5, "actual_rgb_required": True,
                     "actual_five_actions_required": True,
                     "same_prefix_replay_must_match": True,
                     "missing_or_invalid": "branch_failure_no_open_loop_fallback"},
        "visibility": {"shape": [224, 224, 3], "dtype": "uint8",
                       "colors": {"pusher": list(rgb.BLUE), "object": list(rgb.BLOCK)},
                       "minimum_pixels": 8, "margin_pixels": 8,
                       "pusher_single_component": True,
                       "scope": "initial_and_goal_before_dispatch_then_every_returned_frame",
                       "screen_pass_does_not_certify_unoccluded": True,
                       "failure": "stop_branch_retain_allocation_no_imputation"},
        "own_tail_cap_per_branch": {
            "rgb_encode": 1, "iterations": 5, "samples_per_iteration": 64,
            "elites": 8, "variants": 5, "packets": 2,
            "action_encoder": 10, "predict": 10, "predictor_samples": 3200,
            "wall_seconds": 1800,
        },
        "cost": {"resource_axes": list(RESOURCE_AXES),
                 "warmup_cap_per_branch": 10, "control_cap_per_branch": 15,
                 "maximum_branches": 8, "maximum_warmup_steps": 80,
                 "maximum_control_steps": 120,
                 "same_cap_is_not_equal_actual_or_total_compute": True,
                 "root_selector_and_shared_work_separately_charged": True,
                 "failed_attempts_not_refunded": True, "dummy_padding": False},
        "stopping": {
            "success_before_cp5": "performance_success_feedback_and_mechanism_censored",
            "success_at_or_after_cp5": "stop_keep_success_and_actual_step",
            "termination_without_success": "failure",
            "truncation_without_success": "failure",
            "crop_or_invalid_rgb": "stop_branch_method_failure",
            "identity_mismatch": "technical_invalid_no_root_substitution",
            "post_dispatch_failure": "retain_no_primary_rerun",
            "missing_metrics": "missing_never_imputed",
        },
        "analysis": {
            "common_tail_estimand": "prefix-conditioned response to identical actions",
            "own_tail_estimand": "performance after per-prefix real-feedback planning",
            "contrasts": ["selected-minus-zero", "selected-minus-greedy",
                          "selected-minus-random"],
            "metrics": ["native_success", "object_xy_error", "wrapped_angle_error",
                        "object_pose_cost_xy_plus_20_angle", "rgb_task_score",
                        "model_terminal_latent_mse", "actual_stop_step"],
            "same_root_descriptive_only": True,
        },
        "execution_authorized": False, "independent_confirmation": False,
        "P1_admission": False, "gate_modified": False,
    }


def validate_common_tail(common_tail, d2p):
    tail = _actions(common_tail, 10)
    plan = _actions(d2p, 10)
    require(np.array_equal(tail[:5], plan[5:]) and not np.any(tail[5:]),
            "common tail must be D2P tail plus zero")
    return action_sha(tail, 10)


def write_new(path, value):
    path = Path(path)
    require(path.name.startswith(NAME + "_PREP_") and path.suffix == ".json"
            and not os.path.lexists(path), "exclusive protocol output")
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()
