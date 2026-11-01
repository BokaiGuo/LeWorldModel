"""Freeze the dev303 selector-v3 development holdout without opening outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NAME = "p1_dev303_prospective_sidecar_20261101_codex"
OUTPUT = ROOT / "results" / f"{NAME}_PREP_MAIN01.json"
DEV303_SEED = 301000003
SELECTOR_SOURCE_NAMES = (
    "experiments/spectral_reachability/p1_dev302_mixed_prefix_bank_20261101_codex.py",
    "experiments/spectral_reachability/p1_dev302_mixed_prefix_bank_v2_20261101_codex.py",
    "experiments/spectral_reachability/p1_dev302_mixed_prefix_bank_v3_20261101_codex.py",
    "experiments/spectral_reachability/p1_dev302_mixed_bank_cuda_20261101_codex.py",
    "experiments/spectral_reachability/p1_dev302_mixed_bank_cuda_v2_20261101_codex.py",
    "experiments/spectral_reachability/p1_dev302_mixed_bank_cuda_v3_20261101_codex.py",
    "experiments/spectral_reachability/p1_unified_cp5_feedback_cem_20261101_codex.py",
    "experiments/spectral_reachability/codex_p1_rgb_task_score_20261030_v2.py",
)
SOURCE_HASHES = {
    SELECTOR_SOURCE_NAMES[0]: "c3784aa9306007e1394381567e5b3bbfaf049747ed9e1678f171bd6d9142f4c0",
    SELECTOR_SOURCE_NAMES[1]: "3309c45a56e9b6ac38cb6b5903d0781c0a8aa4aec9cf0e1122fe9ee7f4d6b068",
    SELECTOR_SOURCE_NAMES[2]: "bb08f452bb3d4664fa3ea7cba5617ecc1f98b4dabd52fcede27251a91ca0a4a8",
    SELECTOR_SOURCE_NAMES[3]: "69d49d09f000936c99a95adb3ae16ef57e5bce3151d011fa97b5e4ce8366cb0e",
    SELECTOR_SOURCE_NAMES[4]: "34037067815479651cdbd76d398a7f490a874266351f6e266d88fafd7718081a",
    SELECTOR_SOURCE_NAMES[5]: "5e8b3b2562096625da05208293e7aa3e31088b7be382919fec6f0b7e5577e779",
    SELECTOR_SOURCE_NAMES[6]: "f0510356e73d01f020f302e70bd0001a88e46f2e2f326b41d7d98145a86c8b1f",
    SELECTOR_SOURCE_NAMES[7]: "5641c8c0d68be8bca28a2067c5c62f15e9cd04f8e08181f79c4a4e595bede1d7",
}
DEFAULT_REFS = {
    "dev301_queue": {
        "path": ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV01.json",
        "sha256": "6756ad9f3ccf058baa422520d3058dbbdc10e7ac86ff1091b53b719f5a1fe847",
    },
    "dev301_ledger": {
        "path": ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV01.json.native_reset/queue_ledger.jsonl",
        "sha256": "4b3139323eb6d5d20675fb11760b39062822b6a573bccc1e270995e50cffe4d3",
    },
    "dev301_receipt": {
        "path": ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV01.json.native_reset"
                       / "dev_reused_0000.warmup/source_receipt.json",
        "sha256": "b8db7197bb647ef404394947a285d1232f4b2d5b55618d0a79de4392d5755d30",
    },
    "dev302_queue": {
        "path": ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV02.json",
        "sha256": "467975c9c23046e430aae0163fef0450cffc03c27fdf32f8294c616724d0067d",
    },
    "dev302_ledger": {
        "path": ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV02.json.native_reset/queue_ledger.jsonl",
        "sha256": "e7b43bf9e3a28fda61d875ffb31c8272a6791e905f1f7174e7f73707762c7bef",
    },
    "dev302_receipt": {
        "path": ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV02.json.native_reset"
                       / "dev_holdout02_0000.warmup/source_receipt.json",
        "sha256": "341f064c7897f3ca7839186c385e8a4e9811f4d7f3b7ef0a3b680280127051ab",
    },
}
for _name, _sha in SOURCE_HASHES.items():
    DEFAULT_REFS[_name] = {"path": ROOT / _name, "sha256": _sha}
DEFAULT_REFS["forbidden_paths"] = [
    ROOT / "results/p1_dev303_selector_v3_20261101_codex_RUN_MAIN01",
    ROOT / "results/p1_dev303_physical_20261101_codex_RUN_MAIN01",
    ROOT / "results/p1_dev303_outcome_20261101_codex_MAIN01.json",
    ROOT / "results/task_p1_dev303_goal_blind_20261101_codex.json.gz",
    ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV03.json",
]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def read_fixed(ref, *, json_lines=False):
    path = Path(ref["path"])
    require(path.is_file() and not path.is_symlink(), f"fixed regular input: {path}")
    raw = path.read_bytes()
    require(digest(raw) == ref["sha256"], f"fixed SHA: {path}")
    if json_lines:
        require(not raw or raw.endswith(b"\n"), "complete JSONL")
        return [strict_json(line) for line in raw.splitlines()]
    return strict_json(raw)


def source_check(refs):
    expected = {
        "dev301_queue", "dev301_ledger", "dev301_receipt",
        "dev302_queue", "dev302_ledger", "dev302_receipt",
        *SELECTOR_SOURCE_NAMES, "forbidden_paths",
    }
    require(set(refs) == expected, "exact prospective source set")
    require(all(not Path(path).exists() and not Path(path).is_symlink()
                for path in refs["forbidden_paths"]), "dev303 outcome/source namespace already exists")
    for name in SELECTOR_SOURCE_NAMES:
        path = Path(refs[name]["path"])
        require(path.is_file() and not path.is_symlink() and
                digest(path.read_bytes()) == refs[name]["sha256"], f"selector source {name}")
    queues = [read_fixed(refs[name]) for name in ("dev301_queue", "dev302_queue")]
    ledgers = [read_fixed(refs[name], json_lines=True)
               for name in ("dev301_ledger", "dev302_ledger")]
    receipts = [read_fixed(refs[name]) for name in ("dev301_receipt", "dev302_receipt")]
    for queue in queues:
        require(queue["schema"] == "p1_future_root_queue_20261101_codex.frozen.v1" and
                queue["P1_admission"] is False and queue["execution_authorized"] is False,
                "development queue boundary")
    prior = [slot for slot in queues[0]["spec"]["slots"]
             if DEV303_SEED in slot["reset_seeds"]]
    require(prior == [{"slot_id": "not_a_confirmation", "role": "confirmation",
                       "reset_seeds": [DEV303_SEED]}],
            "dev303 seed must match the predeclared, unconsumed DEV01 slot")
    attempted = [event for ledger in ledgers for event in ledger
                 if event.get("seed") == DEV303_SEED]
    require(not attempted, "dev303 seed already attempted in bounded development ledgers")
    development_attempts = sorted({event["seed"] for ledger in ledgers for event in ledger
                                   if event.get("kind") == "reset_attempt"})
    require(development_attempts == [301000001, 301000002], "bounded development roster")
    for receipt in receipts:
        require(receipt["schema"] == "p1_queue_goal_history_20261101_codex.source.v1" and
                receipt["P1_admission"] is receipt["independent_confirmation"] is False and
                receipt["training_membership"] == receipt["historical_coverage"] == "unknown",
                "development source receipt boundary")
    return {
        "seed": DEV303_SEED,
        "previously_declared_in_dev301_queue": True,
        "prior_declared_role": "confirmation",
        "prior_declared_slot": "not_a_confirmation",
        "attempted_in_dev301_or_dev302_ledgers": False,
        "bounded_attempted_seeds": development_attempts,
        "development_history_nonparticipation": "verified_within_bounded_ledgers",
        "developer_blindness": "not_established",
        "global_historical_freshness": "not_established",
        "training_membership": "unknown",
        "source_materialized": False,
        "source_requirement": (
            "before any selector/CUDA/physics, create and seal a development-authorized "
            "DEV03 reset receipt for exactly seed 301000003; the old confirmation-role "
            "slot cannot be silently repurposed"
        ),
    }


def build(refs=DEFAULT_REFS):
    root = source_check(refs)
    return {
        "schema": NAME + ".protocol.v1",
        "status": "dev303_prospective_protocol_frozen_waiting_for_source",
        "frozen_after": "selector_v3_dev302_development",
        "root_source": root,
        "candidate_library": {
            "labels": [
                "zero", "spectral", "gn", "rgb_nearest_contact",
                "rgb_goal_support_direct", "rgb_goal_support_top_route",
                "rgb_goal_support_bottom_route", "random_control",
            ],
            "order_fixed": True,
            "prefix_steps": 5,
            "spectral_and_gn": "sealed from the dev303 root before physical execution",
            "rgb_candidates": "deterministic from dev303 current RGB and goal RGB",
            "random_seed": 3021101,
            "candidate_addition_or_removal_after_outcome": "prohibited",
        },
        "contact_geometry": {
            "input_permission": "RGB-only",
            "simulator_state_used": False,
            "pusher_rgb": [65, 105, 225],
            "object_rgb": [119, 136, 153],
            "minimum_component_pixels": 8,
            "gain": 2.0,
            "pusher_radius_pixels": 6.5,
            "clearance_pixels": 10.0,
            "pixels_per_cumulative_raw": 17.0,
            "contact_center_to_object_pixels": 8.0,
            "segmentation_or_geometry_failure": "abstain_and_record_failure",
        },
        "selector": {
            "version": "v3",
            "contact_filter": "contact_feasible_prior must be true",
            "tail_seed_order": ["old_d2p_plus_zero", "old_gn_plus_zero", "zero"],
            "score": "mean five terminal latent goal MSEs plus max-minus-min range",
            "selection": "global minimum robust score; stable iteration then candidate tie order",
            "same_bank_greedy": "minimum predicted cp5 latent goal MSE; stable candidate order",
            "post_outcome_selector_change": "requires v4 and a different untouched holdout",
        },
        "feedback": {
            "count": 1,
            "boundary": "after the five-step prefix at cp5",
            "input": "actual RGB plus actual five-action history",
            "applies_to_arms": [
                "selected_v3", "same_bank_greedy", "zero_prefix", "random_control"
            ],
            "missing_feedback": "arm failure; no open-loop substitution",
        },
        "budget": {
            "root_selector": {
                "prefix_candidates": 8,
                "tail_iterations_per_prefix": 5,
                "tail_samples": 64,
                "tail_elites": 8,
                "robust_variants": 5,
                "tail_packets": 2,
                "action_encoder_calls": 88,
                "predict_calls": 88,
                "predict_samples": 25608,
                "wall_seconds": 1800,
            },
            "feedback_replan_per_arm": {
                "rgb_encode_calls": 1,
                "iterations": 5,
                "samples": 64,
                "elites": 8,
                "variants": 5,
                "packets": 2,
                "action_encoder_calls": 10,
                "predict_calls": 10,
                "predict_samples": 3200,
                "wall_seconds": 1800,
            },
            "physical": {
                "arms": 4,
                "warmup_zero_steps_per_arm": 10,
                "prefix_steps_per_arm": 5,
                "tail_steps_per_arm": 10,
                "maximum_control_steps": 60,
                "maximum_warmup_steps": 40,
                "no_dummy_compute": True,
                "actual_costs_reported_by_type": True,
            },
        },
        "metrics": {
            "primary": "native_success_by_fixed_step_15",
            "mechanism_primary": (
                "selected_v3 has no cp5 object-pose-cost advantage over its initial state, "
                "then improves terminal object-pose cost or reaches native success"
            ),
            "object_pose_cost": "object_xy_error + 20 * wrapped_object_angle_error",
            "secondary": [
                "terminal_object_xy_error",
                "terminal_wrapped_object_angle_error",
                "terminal_object_pose_cost",
                "object_only_success",
                "official_native_success",
                "RGB_task_score",
                "model_predicted_terminal_latent_goal_MSE",
            ],
            "comparisons": [
                "selected_v3_minus_same_bank_greedy",
                "selected_v3_minus_zero_prefix",
                "selected_v3_minus_random_control",
            ],
            "single_root_inference": "descriptive development evidence only; no significance claim",
        },
        "failure_and_censoring": {
            "root_replacement_after_outcome": "prohibited",
            "additional_root_after_outcome": "prohibited",
            "native_success_before_cp5": (
                "retain performance success; mark mechanism and feedback-replan metrics censored"
            ),
            "native_success_after_cp5_before_step15": (
                "retain success and stop; terminal-at-success metrics reported with stop step"
            ),
            "termination_or_truncation_without_success": "failure with exact stop step",
            "selector_abstention": "method failure; do not choose another root",
            "reset_or_source_identity_mismatch": "technical invalidation; no root substitution",
            "failure_before_any_action_dispatch": "same-root rerun allowed only with new run id",
            "failure_after_action_dispatch": "no rerun for primary analysis; retain as technical failure",
            "missing_or_corrupt_metric": "metric missing, never imputed",
        },
        "root_selection": {
            "rule": "exactly seed 301000003; no result-conditioned replacement",
            "root_count": 1,
            "development_holdout": True,
            "independent_confirmation": False,
        },
        "bindings": {
            name: refs[name]["sha256"]
            for name in (
                "dev301_queue", "dev301_ledger", "dev301_receipt",
                "dev302_queue", "dev302_ledger", "dev302_receipt",
                *SELECTOR_SOURCE_NAMES,
            )
        },
        "outcome_opened": False,
        "cuda_initialized": False,
        "physics_calls": 0,
        "neural_calls": 0,
        "execution_authorized": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }


def write_new(path, value):
    path = Path(path)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        require(os.write(fd, payload) == len(payload), "complete write")
        os.fsync(fd)
    finally:
        os.close(fd)
    return digest(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    require(args.output.name == OUTPUT.name, "exclusive prospective output name")
    report = build()
    sha256 = write_new(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "sha256": sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
