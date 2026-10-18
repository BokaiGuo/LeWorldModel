"""Run a paired Cube RGB macro-transfer ablation panel.

Every arm uses the same seed, calibration roots, held-out roots, action bank
family, and runtime budget.  Only the declared contact/push intervention is
changed.  The panel is intended for the first GPU run after the host CUDA
runtime is available; it refuses to silently fall back to CPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from experiments.gpu_preflight import require_cuda_allocation


ROOT = Path(__file__).resolve().parents[1]
RUNNER = "experiments.run_live_cube_rgb_macro_ranker"


ARMS: dict[str, tuple[str, ...]] = {
    "fullscale_reclose": ("--push-scale", "1.0", "--reclose-each-cycle"),
    "fullscale_open_loop": (
        "--push-scale", "1.0", "--reclose-each-cycle",
        "--disable-push-feedback", "--allow-rgb-regression",
    ),
    "lowgain_reclose": ("--push-scale", "0.25", "--reclose-each-cycle"),
    "lowgain_persistent": ("--push-scale", "0.25"),
    "lowgain_persistent_probe": (
        "--push-scale", "0.25", "--contact-probe", "--contact-probe-steps", "1",
    ),
    "lowgain_effect_adaptive": ("--push-scale", "0.25", "--effect-adaptation"),
    "lowgain_backtrack": ("--push-scale", "0.25", "--push-backtrack"),
    "lowgain_mode_reclose": (
        "--push-scale", "0.25", "--mode-conditioned-reclose",
    ),
    "lowgain_effect_stop": (
        "--push-scale", "0.25", "--effect-adaptation",
        "--stop-push-on-regression",
    ),
    "factorized_lowgain_backtrack": (
        "--factorized-bank", "--push-scale", "0.25", "--push-backtrack",
    ),
    "multiphase_lowgain_backtrack": (
        "--multiphase-bank", "--push-scale", "0.25", "--push-backtrack",
    ),
}


def _arm_command(
    *, arm: str, output: Path, device: str, seed: int, calibration_roots: int,
    test_roots: int, macro_cycles: int, action_block: int, approach_steps: int,
    descent_steps: int, probe_steps: int,
    calibration_target: str = "privileged_task",
    contact_aware_features: bool = False,
    tasks: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5),
) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    command = [
        sys.executable, "-m", RUNNER,
        "--output", str(output),
        "--device", device,
        "--seed", str(seed),
        "--calibration-roots", str(calibration_roots),
        "--test-roots", str(test_roots),
        "--macro-cycles", str(macro_cycles),
        "--action-block", str(action_block),
        "--approach-steps", str(approach_steps),
        "--descent-steps", str(descent_steps),
        "--probe-steps", str(probe_steps),
        "--calibration-target", calibration_target,
        "--tasks", *[str(int(task)) for task in tasks],
        *ARMS[arm],
    ]
    if contact_aware_features:
        command.append("--contact-aware-features")
    return command


def _summarize(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("test", [])
    final_distances = [
        float(row["block_goal_distance_trace"][-1])
        for row in rows if row.get("block_goal_distance_trace")
    ]
    best_distances = [
        float(min(row["block_goal_distance_trace"]))
        for row in rows if row.get("block_goal_distance_trace")
    ]
    reclose_reasons: dict[str, int] = {}
    reclose_count = 0
    progress_values: list[float] = []
    valid_progress_rows = 0
    for row in rows:
        for trace in row.get("mode_reclose_trace", []):
            reason = str(trace.get("reason", "unknown"))
            reclose_reasons[reason] = reclose_reasons.get(reason, 0) + 1
            reclose_count += int(bool(trace.get("reclose", False)))
            progress = trace.get("rgb_progress")
            if progress is not None:
                valid_progress_rows += 1
                if isinstance(progress, (int, float)) and progress == progress:
                    progress_values.append(float(progress))
    cycle_count = sum(len(row.get("mode_reclose_trace", [])) for row in rows)
    return {
        "roots": int(len(rows)),
        "success_rate": float(payload["aggregate"]["success_rate"]),
        "contact_observed_rate": float(payload["aggregate"].get("contact_observed_rate", 0.0)),
        "approach_valid_rate": float(payload["aggregate"]["approach_valid_rate"]),
        "final_distance_mean": None if not final_distances else sum(final_distances) / len(final_distances),
        "best_distance_mean": None if not best_distances else sum(best_distances) / len(best_distances),
        "zero_macro_rate": float(payload["aggregate"]["zero_macro_rate"]),
        "stopped_for_rgb_regression": int(sum(
            bool(row.get("stopped_for_rgb_regression", False)) for row in rows
        )),
        "mode_reclose_rate": None if cycle_count == 0 else float(reclose_count / cycle_count),
        "mode_reclose_reason_counts": reclose_reasons,
        "rgb_progress_valid_rate": None if cycle_count == 0 else float(valid_progress_rows / cycle_count),
        "rgb_progress_mean": None if not progress_values else float(sum(progress_values) / len(progress_values)),
        "rgb_progress_last_values": [
            float(row["mode_reclose_trace"][-1]["rgb_progress"])
            for row in rows
            if row.get("mode_reclose_trace")
            and row["mode_reclose_trace"][-1].get("rgb_progress") is not None
        ],
    }


def _paired_summary(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute root-paired deltas without using outcomes to choose an arm."""
    if "fullscale_reclose" not in payloads:
        raise ValueError("fullscale_reclose is required as the paired reference")

    def _row_values(payload: dict[str, Any], key: str) -> list[float | None]:
        values: list[float | None] = []
        for row in payload.get("test", []):
            trace = row.get("block_goal_distance_trace", [])
            if key == "final_distance":
                values.append(float(trace[-1]) if trace else None)
            elif key == "best_distance":
                values.append(float(min(trace)) if trace else None)
            elif key == "success":
                values.append(float(bool(row.get("success", False))))
            elif key == "contact":
                values.append(float(bool(row.get("contact_observed", False))))
            else:
                raise ValueError(f"unknown paired field: {key}")
        return values

    reference = {
        key: _row_values(payloads["fullscale_reclose"], key)
        for key in ("final_distance", "best_distance", "success", "contact")
    }
    result: dict[str, Any] = {}
    for arm, payload in payloads.items():
        if arm == "fullscale_reclose":
            continue
        current = {key: _row_values(payload, key) for key in reference}
        n = min(len(reference["final_distance"]), len(current["final_distance"]))
        paired = []
        for index in range(n):
            ref_distance = reference["final_distance"][index]
            arm_distance = current["final_distance"][index]
            if ref_distance is None or arm_distance is None:
                continue
            paired.append({
                "root_index": index,
                "final_distance_reduction_vs_fullscale": ref_distance - arm_distance,
                "best_distance_reduction_vs_fullscale": (
                    reference["best_distance"][index] - current["best_distance"][index]
                ),
                "success_delta_vs_fullscale": current["success"][index] - reference["success"][index],
                "contact_delta_vs_fullscale": current["contact"][index] - reference["contact"][index],
            })
        result[arm] = {
            "paired_roots": len(paired),
            "mean_final_distance_reduction_vs_fullscale": (
                None if not paired else sum(row["final_distance_reduction_vs_fullscale"] for row in paired) / len(paired)
            ),
            "mean_best_distance_reduction_vs_fullscale": (
                None if not paired else sum(row["best_distance_reduction_vs_fullscale"] for row in paired) / len(paired)
            ),
            "success_delta_sum_vs_fullscale": (
                None if not paired else sum(row["success_delta_vs_fullscale"] for row in paired)
            ),
            "contact_delta_sum_vs_fullscale": (
                None if not paired else sum(row["contact_delta_vs_fullscale"] for row in paired)
            ),
            "root_rows": paired,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261900)
    parser.add_argument("--calibration-roots", type=int, default=40)
    parser.add_argument("--test-roots", type=int, default=8)
    parser.add_argument("--macro-cycles", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=1)
    parser.add_argument("--approach-steps", type=int, default=8)
    parser.add_argument("--descent-steps", type=int, default=15)
    parser.add_argument("--probe-steps", type=int, default=4)
    parser.add_argument(
        "--calibration-target", choices=("privileged_task", "rgb_goal"),
        default="privileged_task",
        help="shared calibration target for every arm",
    )
    parser.add_argument(
        "--contact-aware-features", action="store_true",
        help="append RGB effector/contact chart to every arm",
    )
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--arms", nargs="+", choices=tuple(ARMS),
        default=list(ARMS),
        help="fixed arm subset; include fullscale_reclose for paired summaries",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse completed arm JSON files from the same output directory",
    )
    args = parser.parse_args()
    cuda_runtime = None
    if args.device.startswith("cuda"):
        cuda_runtime = require_cuda_allocation(args.device)
    if args.calibration_roots < 8 or args.test_roots < 1:
        raise ValueError("calibration-roots must be >=8 and test-roots >=1")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    payloads: dict[str, dict[str, Any]] = {}
    selected_arms = list(dict.fromkeys(args.arms))
    if "fullscale_reclose" not in selected_arms:
        raise ValueError("selected arms must include fullscale_reclose as paired reference")
    for arm in selected_arms:
        output = args.output / f"{arm}.json"
        command = _arm_command(
            arm=arm, output=output, device=args.device, seed=args.seed,
            calibration_roots=args.calibration_roots, test_roots=args.test_roots,
            macro_cycles=args.macro_cycles, action_block=args.action_block,
            approach_steps=args.approach_steps, descent_steps=args.descent_steps,
            probe_steps=args.probe_steps, tasks=args.tasks,
            calibration_target=args.calibration_target,
            contact_aware_features=args.contact_aware_features,
        )
        if args.resume and output.exists():
            payload = json.loads(output.read_text(encoding="utf-8"))
            if payload.get("schema") != "live_cube_rgb_macro_ranker.v1":
                raise ValueError(f"cannot resume incompatible arm artifact: {output}")
            print(f"[cube-panel] resume {arm} from {output}", flush=True)
        else:
            print(f"[cube-panel] running {arm}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            payload = json.loads(output.read_text(encoding="utf-8"))
        payloads[arm] = payload
        summaries[arm] = _summarize(payload)
        artifacts[arm] = str(output.resolve())
    report = {
        "schema": "live_cube_rgb_macro_paired_panel.v1",
        "protocol": {
            "device": args.device,
            "shared_seed": args.seed,
            "calibration_roots": args.calibration_roots,
            "test_roots": args.test_roots,
            "root_disjoint": True,
            "same_action_block": args.action_block,
            "same_macro_cycles": args.macro_cycles,
            "arms": selected_arms,
            "outcome_used_for_arm_selection": False,
            "calibration_target": args.calibration_target,
            "contact_aware_features": args.contact_aware_features,
            "cuda_runtime": cuda_runtime,
        },
        "arms": summaries,
        "paired_vs_fullscale_reclose": _paired_summary(payloads),
        "artifacts": artifacts,
        "claim_boundary": (
            "paired RGB online transfer diagnostic; results require independent GPU roots "
            "and do not by themselves establish causal dynamics or universal planning"
        ),
    }
    (args.output / "panel.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
