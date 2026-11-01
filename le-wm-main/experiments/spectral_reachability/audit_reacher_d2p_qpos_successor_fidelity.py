"""CUDA audit of predicted-versus-physical qpos successor ranking for Reacher.

P1 selects a reconfiguration prefix from a source-supported action bank.  This
diagnostic holds that bank fixed, starts every candidate from the exact same
rendered root, and compares the frozen model's qpos-readout terminal ranking
with evaluator-only qpos distance after the physically executed packet.  It
does not select a candidate by physical outcomes and does not update a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import gymnasium as gym
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor, _spearman
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.run_live_reacher_sequence_candidate_planning import _qpos_distance
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import _normalized_action_bounds
from experiments.spectral_reachability.reacher_d2p_context import collect_executed_context, restore_execution_snapshot
from experiments.spectral_reachability.audit_reacher_d2p_runtime_support import _source_state_context
from experiments.spectral_reachability.reacher_d2p_packet_support import WARMUP_POLICY, load_history_packet_pool, load_packet_pool, proposal_parameters, receipt_source, sample_history_packets
from experiments.spectral_reachability.reacher_d2p_qpos_readout import PREDICTION_READOUT_TASK, ReacherQposReadoutCostModel, load_qpos_readout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sign_agreement(model_cost: np.ndarray, physical_cost: np.ndarray) -> bool:
    """Check selected-versus-zero direction only after both arrays are paired."""

    model, physical = np.asarray(model_cost, dtype=np.float64), np.asarray(physical_cost, dtype=np.float64)
    if model.shape != physical.shape or model.ndim != 1 or len(model) < 2:
        raise ValueError("successor-fidelity sign agreement requires paired candidate costs")
    return bool(np.all((model[1:] - model[0] >= 0.0) == (physical[1:] - physical[0] >= 0.0)))


def _model_costs(
    model: ReacherQposReadoutCostModel, pixels: torch.Tensor, goal: torch.Tensor,
    history: torch.Tensor, bank: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        predicted = p0._terminal_batch(model, pixels, goal, history, bank, action_lower=lower, action_upper=upper)
        target = p0._goal_latent(model, goal)
        prediction_coordinate = model.predicted_coordinate(predicted)
        target_coordinate = model.goal_coordinate(target).unsqueeze(0)
        return (prediction_coordinate - target_coordinate).square().mean(dim=-1).detach().cpu().numpy().astype(np.float64)


def _root_context(
    *, mode: str, env: gym.Env, dataset: Path, seed: int, histories: np.ndarray,
) -> tuple[dict[str, Any], int | None]:
    """Construct one explicit ranking root without mixing state and goal laws."""

    if mode == "official_source_state_history":
        return _source_state_context(env, dataset=dataset, seed=seed), None
    if mode == "executed_source_history":
        warmup, warmup_index = sample_history_packets(histories, seed=seed)
        return collect_executed_context(
            env, seed=seed, history_len=3, action_block=5, warmup_packets=warmup,
            warmup_policy=WARMUP_POLICY + ".contiguous_history",
        ), warmup_index
    raise ValueError(f"unknown successor-fidelity root context: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--qpos-readout", type=Path, required=True)
    parser.add_argument("--predicted-qpos-readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--proposals", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026102560)
    parser.add_argument("--root-context", choices=("executed_source_history", "official_source_state_history"), default="executed_source_history")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Reacher qpos successor-fidelity audit requires host CUDA exactly; CPU fallback is forbidden")
    if args.roots < 4 or args.proposals < 4 or args.output.exists():
        raise ValueError("successor-fidelity audit needs at least four roots/candidates and a new output")
    policy, adapter, artifact, predicted_artifact, output = args.policy.resolve(), args.adapter_artifact.resolve(), args.qpos_readout.resolve(), args.predicted_qpos_readout.resolve(), args.output.resolve()
    install_missing_field_compat(); device = torch.device("cuda"); torch.empty(0, device=device)
    mean, std = _load_action_stats(adapter, "swm/ReacherDMControl-v0")
    dataset, digest = receipt_source(adapter)
    pool, support = load_packet_pool(dataset, expected_sha256=digest)
    histories, history_support = load_history_packet_pool(dataset, expected_sha256=digest, history_len=3, verify_source=False)
    base = load_compatible_pretrained(policy, device=device).eval(); base.requires_grad_(False); base.interpolate_pos_encoding = True
    model = ReacherQposReadoutCostModel(
        base, load_qpos_readout(artifact),
        load_qpos_readout(predicted_artifact, expected_task=PREDICTION_READOUT_TASK),
    ).to(device).eval()
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    lower_np, upper_np = _normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lower_np, device=device), torch.as_tensor(upper_np, device=device)
    preprocessor = _Preprocessor(ActionNormalizer(mean, std)); rows: list[dict[str, Any]] = []
    try:
        for offset in range(args.roots):
            seed = int(args.seed + offset)
            context, warmup_index = _root_context(
                mode=args.root_context, env=env, dataset=dataset, seed=seed, histories=histories,
            )
            bank, bank_receipt = proposal_parameters(pool, proposals=args.proposals, horizon=1, seed=seed, mean=mean, std=std, lower=lower, upper=upper)
            pixels, goal, history = p0._prepare(preprocessor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], dtype=np.float32), device)
            model_cost = _model_costs(model, pixels, goal, history, bank, lower, upper)
            physical_cost: list[float] = []
            for index in range(args.proposals):
                root, goal_rgb = restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
                if not np.array_equal(root, context["root_frame"]) or not np.array_equal(goal_rgb, context["goal"]):
                    raise RuntimeError("successor-fidelity audit failed exact root restoration")
                raw, _ = p0._to_raw_actions(bank[index:index + 1], mean, std, lower, upper, env.action_space.low, env.action_space.high)
                for action in raw:
                    env.step(action)
                native = env.unwrapped
                physical_cost.append(_qpos_distance(np.asarray(native.env.physics.data.qpos), np.asarray(native.env.task.target_qpos)))
            physical = np.asarray(physical_cost, dtype=np.float64)
            rows.append({"seed": seed, "root_context": context["context_mode"], "root_warmup_policy": context["warmup_policy"], "source_episode": context.get("source_episode"), "source_start_row": context.get("source_start_row"), "contiguous_warmup_sequence_index": warmup_index, "proposal_support": bank_receipt, "spearman_model_qpos_cost_vs_physical_qpos_distance": _spearman(model_cost, physical), "candidate_model_qpos_cost": model_cost.tolist(), "candidate_physical_qpos_distance": physical.tolist(), "all_candidate_zero_relative_signs_agree": _sign_agreement(model_cost, physical), "model_argmin_index": int(np.argmin(model_cost)), "physical_argmin_index": int(np.argmin(physical))})
    finally:
        env.close()
    correlations = np.asarray([row["spearman_model_qpos_cost_vs_physical_qpos_distance"] for row in rows], dtype=np.float64)
    result: dict[str, Any] = {"schema": "reacher_d2p_qpos_successor_fidelity.v3", "status": "completed", "scope": "fresh root-paired CUDA ranking fidelity of source-supported first packets; physical qpos is evaluator-only and never selects proposals", "inputs": {"policy": str(policy), "weights_sha256": _sha256(policy / "weights.pt"), "adapter": str(adapter), "adapter_sha256": _sha256(adapter), "qpos_readout": str(artifact), "qpos_readout_sha256": _sha256(artifact), "predicted_qpos_readout": str(predicted_artifact), "predicted_qpos_readout_sha256": _sha256(predicted_artifact), "source_packet_support": support, "contiguous_warmup_support": history_support}, "protocol": {"device": str(device), "roots": int(args.roots), "root_seeds": [row["seed"] for row in rows], "root_context": args.root_context, "proposals": int(args.proposals), "proposal_distribution": "source_locked_training_packet_support_with_raw_zero_at_index_0", "model_coordinate": "separate_frozen_encoder_and_prediction_qpos_sincos_readouts", "physical_coordinate": "evaluator_only_wrapped_qpos_distance"}, "aggregate": {"spearman_by_root": correlations.tolist(), "mean_spearman": float(correlations.mean()), "positive_spearman_every_root": bool(np.all(correlations > 0.0)), "zero_relative_sign_agreement_every_root": bool(all(row["all_candidate_zero_relative_signs_agree"] for row in rows)), "argmin_agreement_count": int(sum(row["model_argmin_index"] == row["physical_argmin_index"] for row in rows))}, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
