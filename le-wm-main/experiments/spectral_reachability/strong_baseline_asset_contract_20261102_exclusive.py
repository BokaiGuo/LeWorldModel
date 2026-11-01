"""Metadata-only contract for reported strong-baseline assets.

This module never reads experiment outcomes and never authorizes execution.
It keeps paper-reported values separate from local, protocol-matched results.
"""

from copy import deepcopy


NAME = "strong_baseline_asset_contract_20261102_exclusive"

_ASSETS = {
    "GRASP": {
        "paper_version": "2602.00475v1",
        "source_locator": "Tables 1, 3, 4; PDF pp. 7-8",
        "reported_cells": 112,
        "metric_unit": "success_percent_and_time_seconds",
        "outcome_domain": "learned_world_model_only",
        "paper_value_usage": "context_only",
        "local_comparison_status": "blocked_protocol_mismatch_or_unknown",
        "required_local_bindings": [
            "checkpoint_and_world_model_identity",
            "task_goal_distribution_and_root_manifest",
            "observation_and_action_contract",
            "feedback_horizon_success_and_timing_denominator",
            "complete_predict_encoder_backward_and_wall_resource_vector",
        ],
        "boundary": "reported success is under the learned world model, not local physical success",
    },
    "Hi-LeWM": {
        "paper_version": "2607.12547v2",
        "source_locator": "Tables 3, 4, 5; PDF pp. 13-14",
        "reported_cells": 106,
        "metric_unit": "success_percent_with_reported_sd",
        "outcome_domain": "PushT_or_Cube_goal_conditioned_control",
        "paper_value_usage": "context_only",
        "local_comparison_status": "blocked_protocol_mismatch_or_unknown",
        "required_local_bindings": [
            "hierarchical_goal_offset_and_schedule",
            "low_level_and_high_level_checkpoint_lineage",
            "macro_action_encoder_and_bank_training_cost",
            "observation_action_feedback_and_execution_contract",
            "per_level_cem_budget_and_evaluation_split",
        ],
        "boundary": "goal offset is environment steps; it is not a local prediction horizon",
    },
    "LeWM": {
        "paper_version": "2603.19312v3",
        "source_locator": "Table 10; PDF p. 26",
        "reported_cells": 8,
        "metric_unit": "success_rate_percent_with_unspecified_pm",
        "outcome_domain": "PushT_goal_conditioned_control",
        "paper_value_usage": "context_only",
        "local_comparison_status": "blocked_protocol_mismatch_or_unknown",
        "required_local_bindings": [
            "versioned_checkpoint_and_task_identity",
            "solver_specific_iterations_and_stopping",
            "action_block_repeat_and_execution_budget",
            "feedback_and_replanning_schedule",
            "uncertainty_definition_and_independent_unit",
        ],
        "boundary": "CEM budget must not be borrowed for the gradient solver rows",
    },
    "PlanWAM": {
        "paper_version": "2610.11382v1",
        "source_locator": "Tables 1, 2, 3; versioned HTML locators",
        "reported_cells": 12,
        "metric_unit": "domain_specific_scores",
        "outcome_domain": "autonomous_driving",
        "paper_value_usage": "context_only",
        "local_comparison_status": "blocked_protocol_mismatch_or_unknown",
        "required_local_bindings": [
            "NAVSIM_or_HUGSIM_task_and_split",
            "multi_camera_history_and_ego_observation_contract",
            "trajectory_waypoint_and_candidate_contract",
            "metric_definition_and_scene_weighting",
            "training_lora_scorer_and_inference_cost",
        ],
        "boundary": "driving scores are not manipulation success rates and cannot rank against PushT",
    },
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def build_catalog():
    assets = deepcopy(_ASSETS)
    return {
        "schema": NAME + ".v1",
        "assets": assets,
        "cross_paper_ranking_allowed": False,
        "local_reproduction": False,
        "scientific_admission": False,
        "neural_calls": 0,
        "physics_calls": 0,
        "network_calls": 0,
        "status": "reported_only_protocol_boundary",
    }


def validate_catalog(value):
    _require(isinstance(value, dict), "catalog must be an object")
    _require(value.get("schema") == NAME + ".v1", "catalog schema")
    _require(value.get("cross_paper_ranking_allowed") is False, "ranking is forbidden")
    _require(value.get("local_reproduction") is False, "local reproduction must be false")
    _require(value.get("scientific_admission") is False, "admission must be false")
    _require(value.get("neural_calls") == 0 and value.get("physics_calls") == 0,
             "runtime calls must be zero")
    _require(set(value.get("assets", {})) == set(_ASSETS), "asset set")
    for name, asset in value["assets"].items():
        _require(asset["evidence_kind"] == "paper_reported", f"{name}: evidence kind")
        _require(asset["paper_value_usage"] == "context_only", f"{name}: context boundary")
        _require(asset["local_comparison_status"] == "blocked_protocol_mismatch_or_unknown",
                 f"{name}: comparison boundary")
        _require(asset["required_local_bindings"], f"{name}: missing bindings")
    return True


def validate_author_value(value):
    _require(isinstance(value, dict), "author value must be an object")
    _require("success" not in value and "outcome" not in value, "outcome fields forbidden")
    _require(value.get("evidence_kind") == "paper_reported", "author evidence kind")
    _require(value.get("method") in _ASSETS, "unknown baseline")
    return True


def comparison_status(asset, local_bindings):
    """Return only conservative metadata status; never computes a ranking."""
    if asset.get("paper_value_usage") != "context_only":
        raise ValueError("reported value cannot be promoted")
    if not local_bindings or local_bindings.get("checkpoint") in (None, "unknown"):
        return "unknown"
    # A collection of caller-supplied labels cannot authenticate provenance,
    # equivalence, or resource receipts. No route in this contract grants parity.
    return "blocked"


# Make the evidence kind explicit only in the generated catalog, keeping the
# source table above concise and immutable.
for _asset in _ASSETS.values():
    _asset["evidence_kind"] = "paper_reported"
