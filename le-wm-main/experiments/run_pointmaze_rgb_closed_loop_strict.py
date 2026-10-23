"""Run the minimum reviewer-required strict PointMaze RGB closed-loop panel.

The six arms share roots, a 32-action root-fixed RGB candidate bank, one-block
execution semantics, an execution cap, renderer source, and frozen checkpoint.
Only the two oracle controls receive simulator state for action selection.
The effect adapter is already frozen; its original train roots and HTSC
calibration roots are audited from its checkpoint before this runner launches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
STRICT_METHODS = (
    "visual_dynamics",
    "effect_adapter_rgb",
    "effect_adapter_rgb_htsc",
    "effect_adapter_rgb_htsc_consistency",
    "ground_truth_candidate_oracle",
    "perfect_dynamics_mpc",
)
DEPLOYABLE_METHODS = frozenset(STRICT_METHODS[:4])
PRIVILEGED_METHODS = frozenset(STRICT_METHODS[4:])


def parse_roots(value: str, label: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be a non-empty duplicate-free root list")
    return values


def _root_from_record(record: dict[str, Any]) -> int:
    instance = str(record.get("instance_id", ""))
    match = re.search(r"episode_(\d+)$", instance)
    if match is None:
        raise ValueError(f"cannot recover PointMaze root from instance_id={instance!r}")
    return int(match.group(1))


def effect_policy_provenance(path: Path) -> dict[str, Any]:
    """Recover only the training/calibration root identities needed for audit."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    protocol = dict(payload.get("protocol", {}))
    train_path = Path(protocol["training_candidate_dataset"]).resolve()
    calibration_path = Path(protocol["calibration_candidate_dataset"]).resolve()
    train_records = json.loads(train_path.read_text(encoding="utf-8"))["records"]
    calibration_records = json.loads(calibration_path.read_text(encoding="utf-8"))["records"]
    adapter_train = sorted({_root_from_record(row) for row in train_records if row.get("split") == "train"})
    htsc_calibration = sorted({_root_from_record(row) for row in calibration_records if row.get("split") == "calibration"})
    if not adapter_train or not htsc_calibration:
        raise ValueError("effect policy is missing declared train or HTSC calibration roots")
    if set(adapter_train) & set(htsc_calibration):
        raise ValueError("effect adapter train roots overlap its HTSC calibration roots")
    return {
        "checkpoint": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "candidate_family": payload.get("candidate_family"),
        "action_steps": int(payload.get("action_steps", 0)),
        "adapter_train_roots": adapter_train,
        "htsc_calibration_roots": htsc_calibration,
        "source_training_candidate_dataset": str(train_path),
        "source_calibration_candidate_dataset": str(calibration_path),
    }


def validate_root_partition(
    provenance: dict[str, Any],
    visual_calibration: list[int],
    full_calibration: list[int],
    test_roots: list[int],
) -> None:
    """Reject accidental reuse before any held-out closed-loop outcome is read."""

    groups = {
        "adapter_train": set(provenance["adapter_train_roots"]),
        "htsc_calibration": set(provenance["htsc_calibration_roots"]),
        "visual_calibration": set(visual_calibration),
        "full_method_calibration": set(full_calibration),
        "held_out_test": set(test_roots),
    }
    names = list(groups)
    overlaps = {
        f"{left}__{right}": sorted(groups[left] & groups[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
        if groups[left] & groups[right]
    }
    if overlaps:
        raise ValueError(f"strict PointMaze split overlap: {overlaps}")


def _bootstrap_interval(values: np.ndarray, *, seed: int = 20261021) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(4000)
    ])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def validate_strict_report(
    report: dict[str, Any],
    *,
    expected_test_roots: list[int],
    max_blocks: int,
    expected_evaluator_goal: np.ndarray | None = None,
) -> dict[str, Any]:
    """Verify equal roots/action contract and hard deployment separation."""

    protocol = report["protocol"]
    if tuple(protocol["methods"]) != STRICT_METHODS:
        raise ValueError("strict report methods do not match the predeclared six-arm panel")
    if protocol["test_episodes"] != expected_test_roots:
        raise ValueError("strict report changed the declared held-out roots")
    action_bank = protocol["action_bank"]
    if not action_bank.get("strict_shared_root_candidate_bank", False):
        raise ValueError("strict comparison did not pin a root-shared candidate bank")
    if int(action_bank["max_execution_blocks"]) != int(max_blocks):
        raise ValueError("strict report changed the shared execution budget")
    if set(protocol["privileged_methods"]) != PRIVILEGED_METHODS:
        raise ValueError("oracle controls are not separately marked as privileged")

    rows = report["rows"]
    per_root: dict[int, list[dict[str, Any]]] = {root: [] for root in expected_test_roots}
    for row in rows:
        root = int(row["episode_id"])
        if root not in per_root:
            raise ValueError(f"unexpected root in strict report: {root}")
        if int(row["blocks_executed"]) > int(max_blocks):
            raise ValueError(f"arm exceeded shared budget on root {root}")
        contract = row["policy_contract"]
        method = row["method"]
        planner_goal = np.asarray(contract.get("planner_goal_qpos"), dtype=np.float64)
        evaluator_goal = np.asarray(contract.get("evaluator_goal_qpos"), dtype=np.float64)
        if contract.get("target_chart") != "particle_relative_qpos_v1":
            raise ValueError(f"row {method}/{root} lacks the declared qpos target chart")
        if planner_goal.shape != (2,) or evaluator_goal.shape != (2,) or not np.isfinite(planner_goal).all() or not np.isfinite(evaluator_goal).all():
            raise ValueError(f"row {method}/{root} lacks finite planner/evaluator goal coordinates")
        if expected_evaluator_goal is not None and not np.allclose(
            evaluator_goal, np.asarray(expected_evaluator_goal, dtype=np.float64), atol=1.0e-12, rtol=0.0,
        ):
            raise ValueError(f"row {method}/{root} changed the declared robust evaluator target")
        if not np.isclose(
            float(contract.get("planner_evaluator_goal_gap_qpos", float("nan"))),
            float(np.linalg.norm(planner_goal - evaluator_goal)), atol=1.0e-12,
        ):
            raise ValueError(f"row {method}/{root} has an inconsistent planner/evaluator goal gap")
        if contract.get("evaluation_objective_id") != "terminal_global_goal_distance_v1":
            raise ValueError(f"row {method}/{root} changed the terminal evaluation objective")
        if contract.get("replanning_semantics") != "receding_horizon_one_block_reobserve_v1":
            raise ValueError(f"row {method}/{root} changed the re-planning semantics")
        if method in DEPLOYABLE_METHODS:
            if contract["simulator_state_in_policy_input"]:
                raise ValueError(f"deployable arm {method} received simulator state")
        elif method in PRIVILEGED_METHODS:
            if not contract["simulator_state_in_policy_input"]:
                raise ValueError(f"privileged control {method} is missing its state-use label")
        else:
            raise ValueError(f"unexpected method row {method}")
        if not contract["strict_shared_root_candidate_bank"]:
            raise ValueError(f"row {method}/{root} did not use the strict candidate contract")
        per_root[root].append(row)

    for root, root_rows in per_root.items():
        if {row["method"] for row in root_rows} != set(STRICT_METHODS):
            raise ValueError(f"root {root} is missing an arm")
        digests = {row["policy_contract"]["candidate_bank_digest"] for row in root_rows}
        banks = {
            json.dumps(row["policy_contract"]["candidate_actions"], separators=(",", ":"))
            for row in root_rows
        }
        if len(digests) != 1 or None in digests or len(banks) != 1:
            raise ValueError(f"root {root} did not share an exact candidate action bank")
        for row in root_rows:
            for trace in row.get("trace", []):
                if trace.get("candidate_bank_digest") not in {None, next(iter(digests))}:
                    raise ValueError(f"root {root} has a decision outside its pinned candidate bank")
                selected_action = trace.get("selected_action")
                if selected_action is not None and not any(
                    np.allclose(np.asarray(selected_action, dtype=np.float64), candidate, atol=1.0e-12)
                    for candidate in np.asarray(row["policy_contract"]["candidate_actions"], dtype=np.float64)
                ):
                    raise ValueError(f"root {root} executed an action outside its pinned candidate bank")
                if row["method"] == "ground_truth_candidate_oracle":
                    gate = trace.get("action_transfer_gate", {})
                    if gate.get("selection_objective_id") != "one_step_active_waypoint_distance_v1":
                        raise ValueError("one-step privileged control lacks its local-ranking objective")
                    if gate.get("search_completeness") != "exact_one_step_fixed_bank_only":
                        raise ValueError("one-step privileged control overstates search completeness")
                if row["method"] == "perfect_dynamics_mpc":
                    gate = trace.get("action_transfer_gate", {})
                    if gate.get("selection_objective_id") != "bounded_beam_active_waypoint_distance_v1":
                        raise ValueError("bounded MPC control lacks its local-ranking objective")
                    if gate.get("search_completeness") != "bounded_beam_not_terminal_complete":
                        raise ValueError("bounded MPC control overstates search completeness")
    return per_root


def paired_summary(per_root: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    reference = "visual_dynamics"
    output: dict[str, Any] = {}
    for method in STRICT_METHODS:
        rows = {row["method"]: row for root_rows in per_root.values() for row in root_rows}
        candidate = [
            float(rows[method]["final_distance"]) - float(rows[reference]["final_distance"])
            for root_rows in per_root.values()
            for rows in [{row["method"]: row for row in root_rows}]
        ]
        success = [
            float(rows[method]["success"]) - float(rows[reference]["success"])
            for root_rows in per_root.values()
            for rows in [{row["method"]: row for row in root_rows}]
        ]
        distances = np.asarray(candidate, dtype=np.float64)
        successes = np.asarray(success, dtype=np.float64)
        output[method] = {
            "paired_final_distance_delta_vs_visual_dynamics": float(distances.mean()),
            "paired_final_distance_delta_95pct_bootstrap": _bootstrap_interval(distances),
            "paired_success_delta_vs_visual_dynamics": float(successes.mean()),
            "paired_success_delta_95pct_bootstrap": _bootstrap_interval(successes, seed=20261022),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/point_maze_tail1800_1899")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar")
    parser.add_argument("--dinov2-source", type=Path, default=ROOT / "external/local_dinov2")
    parser.add_argument("--effect-policy-checkpoint", type=Path, default=ROOT / "results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_live_rgb_latent.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-calibration-roots", default="1868,1869,1870,1871,1872,1873,1874,1875,1876,1877")
    parser.add_argument("--full-calibration-roots", default="1878,1879,1880,1881,1882,1883,1884,1885,1886,1887")
    parser.add_argument("--test-roots", default="1888,1889,1890,1891,1892,1893,1894,1895,1896,1897,1898,1899")
    parser.add_argument(
        "--test-visual-source", choices=("stored_episode", "dataset_matched_renderer"),
        default="stored_episode",
        help="held RGB source; renderer mode enables root-disjoint states without stored test frames",
    )
    parser.add_argument("--max-blocks", type=int, default=24)
    parser.add_argument("--oracle-depth", type=int, default=2)
    parser.add_argument("--oracle-beam-width", type=int, default=8)
    parser.add_argument(
        "--goal-qpos", type=float, nargs=2, default=[0.8, 0.8], metavar=("X", "Y"),
        help="robust free-space terminal target; default is the legacy maze G-cell center",
    )
    parser.add_argument(
        "--allow-wall-sensitive-legacy-goal", action="store_true",
        help="permit the historical wall-sensitive [1.2, 1.2] target only for reproduction",
    )
    args = parser.parse_args()

    visual_calibration = parse_roots(args.visual_calibration_roots, "visual calibration roots")
    full_calibration = parse_roots(args.full_calibration_roots, "full-method calibration roots")
    test_roots = parse_roots(args.test_roots, "test roots")
    from experiments.audit_pointmaze_goal_geometry import audit_goal_geometry

    goal_qpos = np.asarray(args.goal_qpos, dtype=np.float64).reshape(-1)
    goal_geometry = audit_goal_geometry(goal_qpos, success_radius=0.25, robust_clearance=0.05)
    if not goal_geometry["target_has_robust_clearance"] and not args.allow_wall_sensitive_legacy_goal:
        raise ValueError(
            "strict PointMaze requires a declared robust free-space target; "
            "pass --allow-wall-sensitive-legacy-goal only to reproduce the legacy metric"
        )
    provenance = effect_policy_provenance(args.effect_policy_checkpoint.resolve())
    if provenance["action_steps"] != 5 or provenance["candidate_family"] != "runtime_one_block_pulse_v1":
        raise ValueError("strict live comparison requires a one-block runtime-matched effect policy")
    validate_root_partition(provenance, visual_calibration, full_calibration, test_roots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    live_report = args.output_dir / "live_report.json"
    command = [
        sys.executable, "-m", "experiments.run_live_pointmaze_planning",
        "--dataset-root", str(args.dataset_root.resolve()),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--dinov2-source", str(args.dinov2_source.resolve()),
        "--effect-policy-checkpoint", str(args.effect_policy_checkpoint.resolve()),
        "--output", str(live_report.resolve()),
        "--calibration-episodes", ",".join(map(str, visual_calibration)),
        "--effect-consistency-calibration-episodes", ",".join(map(str, full_calibration)),
        "--test-episodes", ",".join(map(str, test_roots)),
        "--test-visual-source", args.test_visual_source,
        "--max-blocks", str(args.max_blocks),
        "--oracle-depth", str(args.oracle_depth),
        "--oracle-beam-width", str(args.oracle_beam_width),
        "--goal-qpos", *(str(value) for value in goal_qpos.tolist()),
        "--strict-shared-root-candidate-bank",
        "--methods", *STRICT_METHODS,
    ]
    (args.output_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    subprocess.run(command, cwd=ROOT, check=True)
    report = json.loads(live_report.read_text(encoding="utf-8"))
    per_root = validate_strict_report(
        report,
        expected_test_roots=test_roots,
        max_blocks=args.max_blocks,
        expected_evaluator_goal=goal_qpos,
    )
    summary = {
        "schema": "pointmaze_rgb_closed_loop_strict.v2",
        "status": "completed",
        "methods": {
            "visual_dynamics": "frozen_visual_dynamics",
            "effect_adapter_rgb": "effect_adapter_only",
            "effect_adapter_rgb_htsc": "HTSC_only",
            "effect_adapter_rgb_htsc_consistency": "full_effect_adapter_HTSC_visual_consistency",
            "ground_truth_candidate_oracle": "privileged_one_block_candidate_oracle",
            "perfect_dynamics_mpc": "privileged_bounded_exact_dynamics_beam_control",
        },
        "split_provenance": {
            **provenance,
            "visual_calibration_roots": visual_calibration,
            "full_method_calibration_roots": full_calibration,
            "held_out_test_roots": test_roots,
            "all_named_splits_pairwise_disjoint": True,
        },
        "shared_contract": {
            "same_roots": True,
            "same_exact_root_candidate_actions": True,
            "same_candidate_count": 32,
            "same_one_block_action_duration": 5,
            "same_max_execution_blocks": int(args.max_blocks),
            "test_visual_source": args.test_visual_source,
            "deployable_arms_rgb_only": sorted(DEPLOYABLE_METHODS),
            "privileged_controls_separately_marked": sorted(PRIVILEGED_METHODS),
            "evaluator_goal_qpos": goal_qpos.tolist(),
            "goal_geometry": goal_geometry,
            "wall_sensitive_legacy_goal_reproduction": bool(args.allow_wall_sensitive_legacy_goal),
        },
        "aggregates": report["aggregates"],
        "paired_vs_frozen_visual_dynamics": paired_summary(per_root),
        "live_report": str(live_report.resolve()),
        "claim_boundary": (
            "A same-legacy-dynamics RGB-only closed-loop ablation with fixed per-root action banks. "
            "Only the four deployable arms are eligible for RGB planning claims; privileged arms are "
            "local-ranking/bounded-search controls rather than terminal-task ceilings. "
            "The HTSC bound remains a marginal fixed-bank calibration and this one-environment panel "
            "does not establish universal transfer, physical causality, or a universal world model."
        ),
    }
    output = args.output_dir / "strict_comparison.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregates": report["aggregates"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
