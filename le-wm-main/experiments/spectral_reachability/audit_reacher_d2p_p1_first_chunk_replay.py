"""Replay the physical first chunk of a P1 receipt without planner inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, install_missing_field_compat
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.reacher_d2p_context import advance_executed_context, collect_executed_context, restore_execution_snapshot


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_report(report: dict[str, Any]) -> None:
    if report.get("schema") != "d2p_reacher_p1_reconfiguration.v1":
        raise ValueError("input is not a Reacher D2P P1 receipt")
    if not report.get("rows"):
        raise ValueError("P1 receipt has no roots")
    protocol = report.get("protocol", {})
    if int(protocol.get("action_block", 0)) < 1:
        raise ValueError("P1 receipt lacks a positive action block")


def _distance(env: gym.Env) -> float:
    raw = env.unwrapped
    return p0._qpos_distance(
        np.asarray(raw.env.physics.data.qpos),
        np.asarray(raw.env.task.target_qpos),
    )


def _raw_action_is_zero(raw_action_block: np.ndarray, *, atol: float) -> bool:
    """Return whether the recorded physical zero-prefix is actually zero.

    The planner parameter for the zero prefix is generally nonzero after the
    checkpoint action normalizer is inverted.  This guard is deliberately on
    *raw* environment actions: the physical control must not silently become
    a mean-action or another learned-space baseline.
    """

    return bool(np.all(np.abs(np.asarray(raw_action_block, dtype=np.float64)) <= float(atol)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument(
        "--zero-action-atol",
        type=float,
        default=1.0e-6,
        help="Raw-action tolerance for float32 normalizer round trips; separate from qpos replay atol.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit output: {args.output}")
    if args.atol <= 0.0 or args.zero_action_atol <= 0.0:
        raise ValueError("replay tolerances must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("P1 replay audit requires host CUDA; CPU fallback is forbidden")
    cuda_host = {"name": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda}
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_report(report)
    action_block = int(report["protocol"]["action_block"])
    install_missing_field_compat()
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    rows: list[dict[str, Any]] = []
    try:
        for row in report["rows"]:
            seed = int(row["seed"])
            context = collect_executed_context(env, seed=seed, history_len=3, action_block=action_block)
            recorded_context = row["executed_context"]
            context_identity = all(
                context[key] == recorded_context[key]
                for key in ("context_mode", "warmup_policy", "root_rgb_sha256", "goal_rgb_sha256", "warmup_actions_sha256")
            )
            if not context_identity:
                raise RuntimeError(f"seed {seed}: deterministic warmup does not reproduce the recorded root")
            root, goal = restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
            if not np.array_equal(root, context["root_frame"]) or not np.array_equal(goal, context["goal"]):
                raise RuntimeError(f"seed {seed}: execution snapshot does not reproduce warmup root/goal RGB")
            method = row["methods"]["reconfiguration"]
            before = _distance(env)
            first = np.asarray(method["first_action_witness_raw"], dtype=np.float32)
            successor = advance_executed_context(env, context=context, raw_action_block=first, action_block=action_block)
            after = _distance(env)
            expected_successor = method["observed_successor_context"]
            successor_identity = all(
                successor[key] == expected_successor[key]
                for key in ("context_mode", "root_rgb_sha256", "goal_rgb_sha256", "advanced_action_sha256")
            )
            stored = method["physical"]
            zero_root, zero_goal = restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
            zero_root_identity = bool(
                np.array_equal(zero_root, context["root_frame"])
                and np.array_equal(zero_goal, context["goal"])
            )
            zero = np.asarray(method["zero_prefix_action_witness_raw"], dtype=np.float32)
            zero_successor = advance_executed_context(
                env, context=context, raw_action_block=zero, action_block=action_block,
            )
            zero_after = _distance(env)
            expected_zero_successor = method["zero_prefix_successor_context"]
            zero_successor_identity = all(
                zero_successor[key] == expected_zero_successor[key]
                for key in ("context_mode", "root_rgb_sha256", "goal_rgb_sha256", "advanced_action_sha256")
            )
            rows.append({
                "seed": seed,
                "warmup_context_replayed": context_identity,
                "successor_context_replayed": successor_identity,
                "initial_qpos_distance_replayed": before,
                "after_first_chunk_qpos_distance_replayed": after,
                "initial_distance_absolute_error": abs(before - float(stored["initial_qpos_distance"])),
                "after_first_distance_absolute_error": abs(after - float(stored["after_first_chunk_qpos_distance"])),
                "physical_first_delta_replayed": after - before,
                "zero_prefix_root_replayed": zero_root_identity,
                "zero_prefix_action_is_raw_zero": _raw_action_is_zero(
                    zero, atol=args.zero_action_atol,
                ),
                "zero_prefix_successor_context_replayed": zero_successor_identity,
                "after_zero_prefix_qpos_distance_replayed": zero_after,
                "after_zero_prefix_distance_absolute_error": abs(
                    zero_after - float(stored["after_zero_prefix_qpos_distance"])
                ),
            })
    finally:
        env.close()
    gates = {
        "all_warmup_contexts_replay": all(row["warmup_context_replayed"] for row in rows),
        "all_successor_contexts_replay": all(row["successor_context_replayed"] for row in rows),
        "all_initial_distances_replay_within_atol": all(row["initial_distance_absolute_error"] <= args.atol for row in rows),
        "all_after_first_distances_replay_within_atol": all(row["after_first_distance_absolute_error"] <= args.atol for row in rows),
        "all_zero_prefix_roots_replay": all(row["zero_prefix_root_replayed"] for row in rows),
        "all_zero_prefix_actions_are_raw_zero": all(row["zero_prefix_action_is_raw_zero"] for row in rows),
        "all_zero_prefix_successor_contexts_replay": all(row["zero_prefix_successor_context_replayed"] for row in rows),
        "all_after_zero_prefix_distances_replay_within_atol": all(
            row["after_zero_prefix_distance_absolute_error"] <= args.atol for row in rows
        ),
    }
    output = {
        "schema": "d2p_reacher_p1_first_chunk_replay_audit.v2",
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "evaluator-only replay of recorded candidate and raw-zero first action blocks; no learned-model inference or efficacy claim",
        "cuda_host": cuda_host,
        "source_report": str(report_path),
        "source_report_sha256": _sha256(report_path),
        "atol": args.atol,
        "zero_action_atol": args.zero_action_atol,
        "gates": gates,
        "rows": rows,
        "claim_boundary": "A pass excludes stale warmup roots, candidate or zero-prefix witness corruption, snapshot restoration, raw-zero control construction, and qpos-distance recording as explanations of the P1 physical non-greediness failure.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "gates": gates, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
