"""Train and strictly evaluate the PointMaze support-expanded HTSC repair.

The live planner and all pre-existing effect-policy code remain immutable.
This wrapper builds a policy from train/effect-calibration local prefixes,
freezes its support gate, then invokes the existing strict live-panel runner
twice on exactly the same roots and physical budget: the original policy is a
support control, and the expanded policy supplies adapter-only, HTSC fallback,
and prior full-arm comparisons.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.build_pointmaze_support_expanded_effect_policy import (
    DEFAULT_EFFECT_CALIBRATION,
    DEFAULT_TRAIN,
    build_support_expanded_policy,
    parse_roots,
    validate_collection_partition,
)
from experiments.run_live_pointmaze_planning import GOAL_STATE


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "effect_adapter_rgb"
HTSC = "effect_adapter_rgb_htsc"
FULL = "effect_adapter_rgb_htsc_consistency"


def bootstrap_ci(values: np.ndarray, seed: int = 20261021, draws: int = 10_000) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(data, size=len(data), replace=True).mean() for _ in range(draws)])
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def _rows(report: dict[str, Any], method: str) -> dict[int, dict[str, Any]]:
    values = {int(row["episode_id"]): row for row in report["rows"] if row["method"] == method}
    if not values:
        raise ValueError(f"live report contains no {method} rows")
    return values


def support_release_diagnostics(report: dict[str, Any], method: str) -> dict[str, Any]:
    """Compute held-out gate coverage, distance shift, and post-action false releases."""

    decisions = [
        (row, index, trace, trace.get("action_transfer_gate", {}))
        for row in _rows(report, method).values()
        for index, trace in enumerate(row.get("trace", []))
        if trace.get("action_transfer_gate", {}).get("effect_selected_index") is not None
    ]
    if not decisions:
        return {"effect_decisions": 0, "release_rate": None, "joint_support_rate": None}
    state_distance = np.asarray([float(gate["state_support_distance"]) for _, _, _, gate in decisions])
    state_threshold = np.asarray([float(gate["state_support_threshold"]) for _, _, _, gate in decisions])
    action_distance = np.asarray([float(gate["action_support_distance"]) for _, _, _, gate in decisions])
    action_threshold = np.asarray([float(gate["action_support_threshold"]) for _, _, _, gate in decisions])
    released = np.asarray([bool(gate.get("effect_policy_released", False)) for _, _, _, gate in decisions])
    joint = np.asarray([
        bool(gate.get("state_support_passed", False)) and bool(gate.get("action_support_passed", False))
        for _, _, _, gate in decisions
    ])
    false_releases = 0
    released_count = 0
    for row, index, trace, gate in decisions:
        if not bool(gate.get("effect_policy_released", False)):
            continue
        released_count += 1
        previous = (
            np.asarray(row["planning_start_state_for_evaluation"], dtype=np.float64)[:2]
            if index == 0 else np.asarray(row["trace"][index - 1]["runtime_state_for_evaluation"], dtype=np.float64)[:2]
        )
        realized = np.asarray(trace["runtime_state_for_evaluation"], dtype=np.float64)[:2]
        predicted = np.asarray(gate.get("predicted_terminal", previous), dtype=np.float64)[:2]
        predicted_progress = float(np.linalg.norm(previous - GOAL_STATE) - np.linalg.norm(predicted - GOAL_STATE))
        realized_progress = float(np.linalg.norm(previous - GOAL_STATE) - np.linalg.norm(realized - GOAL_STATE))
        if predicted_progress > 0.0 and realized_progress <= 0.0:
            false_releases += 1
    def summary(values: np.ndarray) -> dict[str, float]:
        return {"mean": float(values.mean()), "p50": float(np.quantile(values, .5)), "p90": float(np.quantile(values, .9)), "max": float(values.max())}
    return {
        "effect_decisions": int(len(decisions)),
        "release_count": int(released.sum()),
        "release_rate": float(released.mean()),
        "joint_support_count": int(joint.sum()),
        "joint_support_rate": float(joint.mean()),
        "state_support_rate": float(np.mean(state_distance <= state_threshold)),
        "action_support_rate": float(np.mean(action_distance <= action_threshold)),
        "false_release_count": int(false_releases),
        "false_release_rate": None if not released_count else float(false_releases / released_count),
        "state_distance": summary(state_distance),
        "action_distance": summary(action_distance),
        "state_distance_over_threshold": summary(state_distance / np.maximum(state_threshold, 1.0e-12)),
        "action_distance_over_threshold": summary(action_distance / np.maximum(action_threshold, 1.0e-12)),
        "false_release_definition": "released predicted-goal-progress action whose realized next-block goal progress is non-positive",
    }


def paired_utility(report: dict[str, Any], left: str, right: str, seed: int) -> dict[str, Any]:
    left_rows, right_rows = _rows(report, left), _rows(report, right)
    if set(left_rows) != set(right_rows):
        raise ValueError("paired utility requires the same held-out roots")
    # Positive means lower terminal distance for left.
    gain = np.asarray([
        float(right_rows[root]["final_distance"]) - float(left_rows[root]["final_distance"])
        for root in sorted(left_rows)
    ])
    success = np.asarray([
        float(left_rows[root]["success"]) - float(right_rows[root]["success"])
        for root in sorted(left_rows)
    ])
    interval = bootstrap_ci(gain, seed)
    return {
        "left": left, "right": right, "n_roots": int(len(gain)),
        "paired_terminal_distance_gain_right_minus_left": float(gain.mean()),
        "paired_terminal_distance_gain_95pct_bootstrap": interval,
        "paired_success_delta_left_minus_right": float(success.mean()),
        "paired_success_delta_95pct_bootstrap": bootstrap_ci(success, seed + 1),
        "utility_ci_positive": bool(interval[0] > 0.0),
    }


def recovery_decision(reference: dict[str, Any], repaired: dict[str, Any], adapter_vs_htsc: dict[str, Any], adapter_vs_full: dict[str, Any]) -> dict[str, Any]:
    # "Action support" here means the joint state-and-candidate-action gate
    # used by HTSC, not merely a scalar raw-action norm.
    support_recovered = bool(
        repaired["joint_support_rate"] > reference["joint_support_rate"]
        and repaired["release_rate"] > reference["release_rate"]
        and repaired["release_count"] > 0
    )
    utility_positive = bool(adapter_vs_htsc["utility_ci_positive"] and adapter_vs_full["utility_ci_positive"])
    success = bool(support_recovered and utility_positive)
    return {
        "held_test_joint_action_support_recovered": support_recovered,
        "held_test_paired_utility_ci_positive": utility_positive,
        "claimed_success": success,
        "status": "positive_repair" if success else "negative_or_inconclusive_repair",
        "rule": (
            "A support-expanded repair is positive only if held-test joint support/release improve over the frozen "
            "reference and the adapter-only paired terminal-distance CI is strictly positive against both HTSC fallback and prior full arm."
        ),
    }


def _run_strict(*, output_dir: Path, policy: Path, dataset_root: Path, checkpoint: Path, dinov2_source: Path, visual_roots: list[int], full_roots: list[int], test_roots: list[int], max_blocks: int) -> None:
    command = [
        sys.executable, "-m", "experiments.run_pointmaze_rgb_closed_loop_strict",
        "--dataset-root", str(dataset_root), "--checkpoint", str(checkpoint),
        "--dinov2-source", str(dinov2_source), "--effect-policy-checkpoint", str(policy),
        "--output-dir", str(output_dir), "--visual-calibration-roots", ",".join(map(str, visual_roots)),
        "--full-calibration-roots", ",".join(map(str, full_roots)), "--test-roots", ",".join(map(str, test_roots)),
        "--max-blocks", str(max_blocks),
    ]
    (output_dir / "command.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/point_maze_tail1800_1899")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar")
    parser.add_argument("--dinov2-source", type=Path, default=ROOT / "external/local_dinov2")
    parser.add_argument("--base-effect-policy", type=Path, default=ROOT / "results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_live_rgb_latent.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-roots", default=",".join(map(str, DEFAULT_TRAIN)))
    parser.add_argument("--effect-calibration-roots", default=",".join(map(str, DEFAULT_EFFECT_CALIBRATION)))
    parser.add_argument("--visual-calibration-roots", default="1868,1869,1870,1871,1872,1873,1874,1875,1876,1877")
    parser.add_argument("--full-calibration-roots", default="1878,1879,1880,1881,1882,1883,1884,1885,1886,1887")
    parser.add_argument("--test-roots", default="1888,1889,1890,1891,1892,1893,1894,1895,1896,1897,1898,1899")
    parser.add_argument("--prefix-count", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--max-blocks", type=int, default=24)
    args = parser.parse_args()
    train = parse_roots(args.train_roots, "train roots")
    effect_calibration = parse_roots(args.effect_calibration_roots, "effect-calibration roots")
    visual = parse_roots(args.visual_calibration_roots, "visual-calibration roots")
    full = parse_roots(args.full_calibration_roots, "full-calibration roots")
    test = parse_roots(args.test_roots, "held-out test roots")
    validate_collection_partition(train, effect_calibration)
    named = {"train": set(train), "effect": set(effect_calibration), "visual": set(visual), "full": set(full), "test": set(test)}
    overlap = {f"{a}__{b}": sorted(named[a] & named[b]) for a in named for b in named if a < b and named[a] & named[b]}
    if overlap:
        raise ValueError(f"support-expanded protocol root overlap: {overlap}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight = args.output_dir / "gpu_preflight.json"
    subprocess.run([sys.executable, "-m", "experiments.gpu_preflight", "--device", "cuda", "--output", str(preflight)], cwd=ROOT, check=True)
    build = build_support_expanded_policy(
        dataset_root=args.dataset_root.resolve(), checkpoint=args.checkpoint.resolve(), dinov2_source=args.dinov2_source.resolve(),
        base_effect_policy=args.base_effect_policy.resolve(), output_dir=args.output_dir / "build",
        train_roots=train, effect_calibration_roots=effect_calibration, prefix_count=args.prefix_count, epochs=args.epochs,
    )
    reference_dir, repaired_dir = args.output_dir / "reference_policy", args.output_dir / "support_expanded_policy"
    _run_strict(output_dir=reference_dir, policy=args.base_effect_policy.resolve(), dataset_root=args.dataset_root.resolve(), checkpoint=args.checkpoint.resolve(), dinov2_source=args.dinov2_source.resolve(), visual_roots=visual, full_roots=full, test_roots=test, max_blocks=args.max_blocks)
    _run_strict(output_dir=repaired_dir, policy=Path(build["policy"]), dataset_root=args.dataset_root.resolve(), checkpoint=args.checkpoint.resolve(), dinov2_source=args.dinov2_source.resolve(), visual_roots=visual, full_roots=full, test_roots=test, max_blocks=args.max_blocks)
    reference_live = json.loads((reference_dir / "live_report.json").read_text(encoding="utf-8"))
    repaired_live = json.loads((repaired_dir / "live_report.json").read_text(encoding="utf-8"))
    reference_diag = support_release_diagnostics(reference_live, HTSC)
    repaired_diag = support_release_diagnostics(repaired_live, HTSC)
    adapter_vs_htsc = paired_utility(repaired_live, REFERENCE, HTSC, 7301)
    adapter_vs_full = paired_utility(repaired_live, REFERENCE, FULL, 7302)
    decision = recovery_decision(reference_diag, repaired_diag, adapter_vs_htsc, adapter_vs_full)
    report = {
        "schema": "pointmaze.support_expanded_htsc_repair.v1", "status": "completed",
        "gpu_preflight": str(preflight.resolve()), "build": build,
        "evaluation_contract": {
            "strict_runner": "experiments.run_pointmaze_rgb_closed_loop_strict",
            "shared_test_roots": test, "shared_physical_block_budget": int(args.max_blocks),
            "shared_root_fixed_32_action_bank": True, "deployable_arms_rgb_only": True,
            "reference_policy": str(args.base_effect_policy.resolve()), "repaired_policy": build["policy"],
        },
        "held_test_support_and_release": {"reference_htsc": reference_diag, "support_expanded_htsc": repaired_diag},
        "held_test_task_outcomes": {
            "support_expanded_adapter_only": repaired_live["aggregates"][REFERENCE],
            "support_expanded_htsc_fallback": repaired_live["aggregates"][HTSC],
            "support_expanded_prior_full_arm": repaired_live["aggregates"][FULL],
        },
        "paired_utility": {"adapter_vs_htsc_fallback": adapter_vs_htsc, "adapter_vs_prior_full_arm": adapter_vs_full},
        "false_release_and_distribution_diagnostics": {"reference": reference_diag, "support_expanded": repaired_diag},
        "decision": decision,
        "artifacts": {"reference_strict": str((reference_dir / "strict_comparison.json").resolve()), "support_expanded_strict": str((repaired_dir / "strict_comparison.json").resolve())},
        "claim_boundary": (
            "This is one legacy PointMaze fixed-bank RGB panel. It reports a repair only under the predeclared joint-support and paired-utility rule; otherwise it is a negative mechanism result, not a world-model or coverage guarantee."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str((args.output_dir / "report.json").resolve()), "decision": decision}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
