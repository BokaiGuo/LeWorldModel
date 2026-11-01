"""V2 full-CEM host wrapper with an explicit, non-legacy protocol.

No model/physics execution on import or preparation. The existing v1
execution machinery is reused without modifying its source or saved outputs.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from . import codex_p1_full_cem_host_20261101_exclusive as prior

NAME = "codex_p1_full_cem_host_v2_20261101_exclusive"
AUTH = "MAIN_AUTHORIZES_P1_FULL_CEM_V2_HOST_CUDA_AND_PHYSICS"
HERE = Path(__file__).resolve()
DOC = HERE.parents[3] / "docs/D2P_P1_FULL_CEM_HOST_V2_20261101_EXCLUSIVE.md"
TEST = HERE.parents[2] / "tests/test_codex_p1_full_cem_host_v2_20261101_exclusive.py"
BOUNDARIES = (0, 5, 10, 15)

# Only fields with unchanged, applicable meaning are carried forward.
PRESERVED = (
    "root_slot", "boundaries", "feedback_horizons", "packet_steps", "terminal",
    "goal", "initial_native_success", "feedback", "physical_caps", "visual_caps",
    "cuda_allocated_bytes", "cuda_reserved_bytes", "disk_floor_bytes",
    "final_reserve_bytes", "output_bytes", "wall_seconds", "E3_admission",
    "P1_admission", "independent_confirmation",
)


def make_protocol(legacy, root):
    prior.spec.require(type(root) is int and root in (0, 1), "root")
    prior.spec.require(legacy.get("root_slot") == root, "root/protocol mismatch")
    prior.spec.require(legacy.get("boundaries") == list(BOUNDARIES)
                       and legacy.get("feedback_horizons") == list(prior.cem.HORIZONS),
                       "controller boundary/horizon drift")
    costs = {str(b): prior.node_cost(b) for b in BOUNDARIES}
    prior.spec.require(legacy.get("node_costs") == costs, "node cost drift")
    prior.spec.require(all(key in legacy for key in PRESERVED), "missing resource/identity field")
    value = {key: deepcopy(legacy[key]) for key in PRESERVED}
    value.update(
        schema=f"{NAME}_root{root}.protocol.v2", arms=["full_cem"],
        node_costs=deepcopy(costs), node_caps_by_boundary=deepcopy(costs),
        module_hooks_gate_each_node=False, module_hook_coverage_verified=False,
        actual_usage_verified=False, full_cost_certified=False, fairness_proven=False,
        independent_backward_measurement="not_applicable_no_cem_backward_calls",
        controller_seed=prior.SEED,
        cem={
            "iterations": prior.cem.ITERATIONS, "samples_per_iteration": prior.cem.SAMPLES,
            "elites": prior.cem.ELITES, "candidate_scores_per_node": prior.cem.ITERATIONS * prior.cem.SAMPLES,
            "winner_replays_per_node": 1, "initialization": "zero_center_float32_std_0.5",
            "sampling": "PCG64_normal_clip_raw_minus1_plus1_first_candidate_center",
            "elite_update": "float32_mean_std_clip_0.05_1.0",
            "winner": "global_min_score_then_lowest_index",
            "gradient_suffix": False, "extra_winner_updates": 0,
        },
        node_ledger={
            "driver_level": "node", "account_arm_label": "native",
            "account_arm_label_meaning": "legacy_adapter_slot_not_native_baseline",
            "contract_arm_slots": "four_legacy_schema_slots_not_four_executed_arms",
            "status": "closed_unverified", "actual_usage_verified": False,
            "coverage_verified": False, "shared_upstream_cost": "unknown_unallocated",
        },
        timing="runner_wall_time_including_planner_and_audit_not_equal_compute",
    )
    return value


def check_prepare_name(path, root):
    prior.spec.require(type(root) is int and root in (0, 1), "root")
    p = Path(path)
    prior.spec.require(p.name.startswith(f"{NAME}_root{root}_PREP_")
                       and p.name.endswith(".json"), "v2 root-specific prepare required")
    return p


def check_output_name(path, root):
    p = Path(path)
    prior.spec.require(p.name.startswith(f"{NAME}_root{root}_RUN_"),
                       "v2 root-specific run output required")
    return p


@contextmanager
def configured(root):
    # v1's report schema uses its module NAME; patch only in this process,
    # never on disk, and restore it even on exception.
    with patch.object(prior, "NAME", NAME):
        with prior.configured(root) as runner:
            legacy_protocol, legacy_build = runner.protocol, runner.build

            def protocol():
                return make_protocol(legacy_protocol(), root)

            def build():
                value = legacy_build()
                value["schema"] = f"{NAME}_root{root}.prepare.v2"
                value["protocol"] = protocol()
                for path in (HERE, DOC, TEST):
                    value["sources"][str(path)] = runner.sha(path)
                return value

            with patch.object(runner, "NAME", f"{NAME}_root{root}"), \
                 patch.object(runner, "AUTH", AUTH), \
                 patch.object(runner, "protocol", protocol), \
                 patch.object(runner, "build", build):
                yield runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=int, choices=(0, 1))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-prepare")
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    p = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--authorize-host-cuda-and-physics", action="store_true", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        check_prepare_name(args.output, args.root)
    elif args.command == "run":
        check_prepare_name(args.prepared, args.root)
        check_output_name(args.output, args.root)
        prior.spec.require(args.authorization == AUTH, "explicit v2 authorization")
    with configured(args.root) as runner:
        if args.command == "dry-prepare":
            value = runner.build()
            result = {"status": "dry_prepared_no_model_or_physics", "digest": runner.digest(value),
                      "root": args.root, "protocol": value["protocol"]}
        elif args.command == "prepare":
            result = runner.prepare(args.output)
        else:
            with prior.gpu.locking.gpu_lock(prior.gpu.LOCK):
                result = runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
