"""CUDA calibration diagnostic for physical consequences of model-ranked P1 prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import _normalized_action_bounds, _parameter_hash, _sha256_file
from experiments.spectral_reachability.reacher_d2p_context import WARMUP_POLICY, advance_executed_context, collect_executed_context, restore_execution_snapshot


def ranked_eligible_indices(
    rows: list[dict[str, Any]], *, count: int,
    selection_metric: str = "analytic_v_reach",
) -> list[int]:
    """Choose a fixed top-k model-ranked diagnostic subset, never by qpos."""

    if count < 1:
        raise ValueError("candidate count must be positive")
    if not rows:
        raise ValueError("candidate rows must be nonempty")
    score_key = "value" if selection_metric == "analytic_v_reach" else "bounded_dls_suffix_goal_cost"
    zero = rows[0]
    eligible = [
        row for row in rows[1:]
        if row["immediate_goal_cost"] > zero["immediate_goal_cost"]
        and row[score_key] < zero[score_key]
    ]
    return [int(row["proposal_index"]) for row in sorted(eligible, key=lambda row: (row[score_key], row["proposal_index"]))[:count]]


def _distance(env: gym.Env) -> float:
    raw = env.unwrapped
    return p0._qpos_distance(np.asarray(raw.env.physics.data.qpos), np.asarray(raw.env.task.target_qpos))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p0-evidence", type=Path, required=True)
    parser.add_argument("--p1-preflight-evidence", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm"))
    parser.add_argument("--adapter-artifact", type=Path, default=Path("results/live_cross_environment_prediction_20261015/reacher_cuda/cross_environment_prediction.json"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--proposals", type=int, default=p1.DEFAULT_RECONFIGURATION_PROPOSALS)
    parser.add_argument(
        "--proposal-distribution", choices=p1.RECONFIGURATION_PROPOSAL_DISTRIBUTIONS,
        default=p1.DEFAULT_RECONFIGURATION_PROPOSAL_DISTRIBUTION,
    )
    parser.add_argument(
        "--selection-metric", choices=p1.RECONFIGURATION_SELECTION_METRICS,
        default=p1.DEFAULT_RECONFIGURATION_SELECTION_METRIC,
    )
    parser.add_argument("--damping", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable diagnostic output: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("candidate-prefix diagnostic requires host CUDA; CPU fallback is forbidden")
    if min(args.candidates, args.action_block, args.proposals) < 1 or args.proposals < 2 or args.damping <= 0.0:
        raise ValueError("invalid candidate-prefix diagnostic dimensions")
    p0_evidence = p1._validate_upstream_p0(args.p0_evidence)
    preflight_report = p1.resolve_preflight_report(args.p1_preflight_evidence)
    preflight = json.loads(preflight_report.read_text(encoding="utf-8"))
    if preflight.get("status") != "pass" or not all(preflight.get("gates", {}).values()):
        raise ValueError("candidate-prefix diagnostic requires a passing P1 preflight")
    if preflight.get("protocol", {}).get("proposal_distribution") != args.proposal_distribution:
        raise ValueError("candidate-prefix diagnostic must use the preflight proposal distribution")
    if preflight.get("protocol", {}).get("selection_metric") != args.selection_metric:
        raise ValueError("candidate-prefix diagnostic must use the preflight selection metric")
    if preflight.get("protocol", {}).get("warmup_policy") != WARMUP_POLICY:
        raise ValueError("candidate-prefix diagnostic must use the current warmup action-packet policy")
    install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = _load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    hash_before = _parameter_hash(model)
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    try:
        action_dim = int(np.prod(env.action_space.shape))
        checkpoint_block = p0._validate_reacher_action_protocol(model, mean, std, action_dim=action_dim, action_block=args.action_block)
        low, high = _normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
        context = collect_executed_context(env, seed=args.seed, history_len=3, action_block=args.action_block)
        pixels, goal, history = p0._prepare(preprocessor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], dtype=np.float32), device)
        _, compilation = p1._compile_reconfiguration(
            model, pixels, goal, history, reconfiguration_horizon=1, future_horizon=2,
            action_lower=lower, action_upper=upper, damping=args.damping, proposals=args.proposals,
            non_greedy_margin=1.0e-6, reachability_margin=1.0e-6, seed=args.seed,
            proposal_distribution=args.proposal_distribution,
            selection_metric=args.selection_metric,
        )
        indices = ranked_eligible_indices(
            compilation["proposal_rows"], count=args.candidates, selection_metric=args.selection_metric,
        )
        bank = p1._reconfiguration_proposal_bank(proposals=args.proposals, horizon=1, action_dim=history.shape[-1], device=device, dtype=history.dtype, seed=args.seed, distribution=args.proposal_distribution)
        zero_raw, _ = p0._to_raw_actions(torch.zeros_like(bank[:1]), mean, std, lower, upper, env.action_space.low, env.action_space.high)
        root, goal_image = restore_execution_snapshot(env, seed=args.seed, snapshot=context["execution_snapshot"])
        root_identity = p0._verify_execution_identity(root, goal_image, context["root_frame"], context["goal"])
        before = _distance(env)
        advance_executed_context(env, context=context, raw_action_block=zero_raw, action_block=args.action_block)
        zero_after = _distance(env)
        selected_rows: list[dict[str, Any]] = []
        rank_by_index = {proposal_index: rank for rank, proposal_index in enumerate(indices, start=1)}
        row_by_index = {int(row["proposal_index"]): row for row in compilation["proposal_rows"]}
        for index in indices:
            raw, normalized = p0._to_raw_actions(bank[index:index + 1], mean, std, lower, upper, env.action_space.low, env.action_space.high)
            root, goal_image = restore_execution_snapshot(env, seed=args.seed, snapshot=context["execution_snapshot"])
            p0._verify_execution_identity(root, goal_image, context["root_frame"], context["goal"])
            successor = advance_executed_context(env, context=context, raw_action_block=raw, action_block=args.action_block)
            after = _distance(env)
            model_row = row_by_index[index]
            selected_rows.append({
                "model_rank_by_future_reachability": rank_by_index[index],
                "proposal_index": index,
                "immediate_goal_cost": model_row["immediate_goal_cost"],
                "future_reachability_value": model_row["value"],
                "bounded_dls_suffix_goal_cost": model_row["bounded_dls_suffix_goal_cost"],
                "unbounded_dls_parameter_l2": model_row["unbounded_dls_parameter_l2"],
                "action_witness_raw": raw.tolist(),
                "action_witness_normalized": normalized.tolist(),
                "successor_root_rgb_sha256": successor["root_rgb_sha256"],
                "after_prefix_qpos_distance": after,
                "candidate_minus_zero_prefix_distance": after - zero_after,
                "physically_non_greedy_relative_to_zero_prefix": bool(after >= zero_after - 1.0e-8),
            })
    finally:
        env.close()
    hash_after = _parameter_hash(model)
    gates = {
        "cuda_only": True,
        "upstream_p0_admission_verified": p0_evidence["p1_unlocked"],
        "frozen_parameter_hash_unchanged": hash_before == hash_after,
        "root_identity_verified": bool(root_identity),
        "at_least_one_model_ranked_candidate": bool(selected_rows),
        "all_raw_actions_within_environment_box": all(np.all(np.asarray(row["action_witness_raw"]) >= env.action_space.low - 1.0e-6) and np.all(np.asarray(row["action_witness_raw"]) <= env.action_space.high + 1.0e-6) for row in selected_rows),
    }
    output = {
        "schema": "d2p_reacher_p1_candidate_prefix_audit.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "CUDA calibration diagnostic; physical qpos is recorded only after a frozen model-ranked candidate set is fixed",
        "protocol": {"seed": args.seed, "candidates": args.candidates, "proposals": args.proposals, "proposal_distribution": args.proposal_distribution, "selection_metric": args.selection_metric, "action_block": args.action_block, "checkpoint_action_block": checkpoint_block, "damping": args.damping, "zero_prefix_reference": "same snapshot and same action-block length"},
        "inputs": {"p0_evidence": str(args.p0_evidence.resolve()), "p1_preflight_evidence": str(preflight_report.resolve()), "runner_sha256": _sha256_file(Path(__file__))},
        "parameter_hash_before": hash_before,
        "parameter_hash_after": hash_after,
        "gates": gates,
        "root_qpos_distance": before,
        "zero_prefix_qpos_distance": zero_after,
        "candidates": selected_rows,
        "claim_boundary": "This diagnostic may locate a ranking/calibration defect for later protocol design. It is not an outcome panel and cannot itself choose an efficacy result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "candidate_count": len(selected_rows), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
