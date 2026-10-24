"""CUDA-only first-chunk model-to-observation fidelity audit for a sealed P1 receipt.

For each already executed Reacher-D2P P1 reconfiguration witness and its
same-root raw-zero control, this evaluator rebuilds the recorded warmup
context, advances the real environment through the fixed first packet, and
compares the frozen model successor with the encoder representation of the
observed RGB successor.  It never reads qpos or ranks/selects an action.

The audit answers a narrow diagnostic question: does the model preserve the
candidate-versus-zero immediate goal ordering after the action reaches the
real renderer?  It therefore distinguishes an execution-path defect from a
short-horizon model-to-observation calibration boundary, without converting a
calibration receipt into an efficacy result.
"""

from __future__ import annotations

import argparse
import hashlib
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
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.reacher_d2p_context import advance_executed_context, collect_executed_context, restore_execution_snapshot


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_report(report: dict[str, Any]) -> None:
    if report.get("schema") != "d2p_reacher_p1_reconfiguration.v1":
        raise ValueError("input is not a Reacher D2P P1 receipt")
    protocol = report.get("protocol", {})
    if int(protocol.get("action_block", 0)) < 1 or int(protocol.get("reconfiguration_horizon", 0)) != 1:
        raise ValueError("first-chunk fidelity audit requires a one-token P1 action block")
    if not report.get("rows"):
        raise ValueError("P1 receipt has no roots")


def _target_delta_from_report(report: dict[str, Any]) -> tuple[float, float]:
    """Recover and validate the renderer geometry sealed by the P1 receipt.

    The fidelity replay compares rendered successors, so silently falling back
    to the wrapper default would audit a different physical target if a later
    P1 protocol changed its target offset.
    """

    value = report.get("protocol", {}).get("target_delta")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("P1 receipt is missing a two-coordinate target_delta")
    delta = tuple(float(item) for item in value)
    if not np.isfinite(delta).all():
        raise ValueError("P1 receipt target_delta must be finite")
    return delta


def _validate_runtime_inputs(report: dict[str, Any], *, policy: Path, adapter: Path) -> dict[str, str]:
    """Ensure the replay uses exactly the checkpoint and adapter in its source.

    A replay that loads another local checkpoint can still produce finite RGB
    contexts and action round-trips.  It cannot, however, validate the model
    ordering on which the source calibration was selected.
    """

    weights = policy / "weights.pt"
    if not weights.is_file() or not adapter.is_file():
        raise ValueError("fidelity replay checkpoint weights or adapter artifact is missing")
    inputs = report.get("inputs", {})
    expected_weights = inputs.get("weights_sha256")
    expected_adapter = inputs.get("adapter_sha256")
    actual_weights = _sha256(weights)
    actual_adapter = _sha256(adapter)
    if not isinstance(expected_weights, str) or actual_weights != expected_weights:
        raise ValueError("fidelity replay checkpoint does not match the sealed P1 receipt")
    if not isinstance(expected_adapter, str) or actual_adapter != expected_adapter:
        raise ValueError("fidelity replay adapter does not match the sealed P1 receipt")
    return {
        "checkpoint_weights_sha256": actual_weights,
        "adapter_sha256": actual_adapter,
    }


def _first_successor_prediction(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    history: torch.Tensor,
    normalized_first_block: torch.Tensor,
) -> torch.Tensor:
    """Mirror P1's root-anchored one-packet successor recurrence exactly.

    ``normalized_first_block`` is the already bounded checkpoint action token
    stored in the P1 receipt.  It must not be passed through tanh a second
    time: doing so would audit a different action from the packet executed in
    the environment.
    """

    if pixels.ndim != 5 or history.ndim != 3 or normalized_first_block.ndim != 2:
        raise ValueError("invalid first-chunk fidelity tensor ranks")
    if pixels.shape[0] != history.shape[0] != 1 or pixels.shape[1] != history.shape[1]:
        raise ValueError("first-chunk fidelity context is not root anchored")
    if tuple(normalized_first_block.shape) != (1, history.shape[-1]):
        raise ValueError("recorded normalized first block has incompatible shape")
    history_size = int(pixels.shape[1])
    encoded = p0._latent_context(model, pixels, history)
    action_context = torch.cat([history[:, 1:], normalized_first_block[None]], dim=1)
    with torch.no_grad():
        return model.predict(
            encoded[:, -history_size:], model.action_encoder(action_context[:, -history_size:]),
        )[0, -1].detach()


def _goal_cost(latent: torch.Tensor, goal_latent: torch.Tensor) -> float:
    if latent.shape != goal_latent.shape:
        raise ValueError("successor and goal latent shapes differ")
    return float((latent - goal_latent).square().mean().detach())


def _strict_order_agrees(left: float, right: float, observed_left: float, observed_right: float) -> bool:
    """Require the sign of a non-tied model ordering to survive observation."""

    model_delta = float(left) - float(right)
    observed_delta = float(observed_left) - float(observed_right)
    return bool(model_delta != 0.0 and observed_delta != 0.0 and np.sign(model_delta) == np.sign(observed_delta))


def _evaluate_packet(
    *,
    name: str,
    model: torch.nn.Module,
    preprocessor: _Preprocessor,
    env: gym.Env,
    context: dict[str, Any],
    normalized: np.ndarray,
    raw: np.ndarray,
    action_block: int,
    device: torch.device,
    goal_latent: torch.Tensor,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> dict[str, Any]:
    pixels, _, history = p0._prepare(
        preprocessor, context["frames"], context["root_frame"], context["goal"],
        np.asarray(context["history_action_blocks"], dtype=np.float32), device,
    )
    normalized_tensor = torch.as_tensor(normalized, device=device, dtype=history.dtype)
    predicted = _first_successor_prediction(model, pixels, history, normalized_tensor)
    root, goal = restore_execution_snapshot(env, seed=int(context["seed"]), snapshot=context["execution_snapshot"])
    root_identity = p0._verify_execution_identity(root, goal, context["root_frame"], context["goal"])
    advanced = advance_executed_context(env, context=context, raw_action_block=raw, action_block=action_block)
    observed_pixels, _, observed_history = p0._prepare(
        preprocessor, advanced["frames"], advanced["root_frame"], advanced["goal"],
        np.asarray(advanced["history_action_blocks"], dtype=np.float32), device,
    )
    observed = p0._latent_context(model, observed_pixels, observed_history)[0, -1]
    raw_from_normalized = p0._normalized_blocks_to_raw_actions(
        normalized, action_mean, action_std, action_dim=2,
    )
    return {
        "name": name,
        "root_identity": root_identity,
        "advanced_context": {
            key: advanced[key]
            for key in ("context_mode", "root_rgb_sha256", "goal_rgb_sha256", "advanced_action_sha256")
        },
        "predicted_goal_cost": _goal_cost(predicted, goal_latent),
        "observed_encoded_goal_cost": _goal_cost(observed, goal_latent),
        "latent_l2_error": float(torch.linalg.vector_norm(predicted - observed).detach()),
        "raw_from_normalized_max_abs_error": float(np.max(np.abs(raw_from_normalized - raw))),
        "predicted_latent_finite": bool(torch.isfinite(predicted).all()),
        "observed_latent_finite": bool(torch.isfinite(observed).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm"))
    parser.add_argument("--adapter-artifact", type=Path, default=Path("results/live_cross_environment_prediction_20261015/reacher_cuda/cross_environment_prediction.json"))
    parser.add_argument("--raw-action-atol", type=float, default=1.0e-6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit output: {args.output}")
    if args.raw_action_atol <= 0.0 or not torch.cuda.is_available():
        raise RuntimeError("P1 first-chunk fidelity audit requires host CUDA and a positive raw-action tolerance; CPU fallback is forbidden")
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_report(report)
    target_delta = _target_delta_from_report(report)
    policy = args.policy.expanduser().resolve()
    adapter_artifact = args.adapter_artifact.expanduser().resolve()
    runtime_input_binding = _validate_runtime_inputs(
        report, policy=policy, adapter=adapter_artifact,
    )
    install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = _load_action_stats(adapter_artifact, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    action_block = int(report["protocol"]["action_block"])
    env = ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
        target_delta=target_delta,
    )
    rows: list[dict[str, Any]] = []
    try:
        for receipt_row in report["rows"]:
            method = receipt_row["methods"]["reconfiguration"]
            if not method["first_receipt"]["non_greedy_witness_found"]:
                continue
            seed = int(receipt_row["seed"])
            context = collect_executed_context(env, seed=seed, history_len=3, action_block=action_block)
            context["seed"] = seed
            recorded_context = receipt_row["executed_context"]
            context_replayed = all(
                context[key] == recorded_context[key]
                for key in ("context_mode", "warmup_policy", "root_rgb_sha256", "goal_rgb_sha256", "warmup_actions_sha256")
            )
            if not context_replayed:
                raise RuntimeError(f"seed {seed}: deterministic warmup does not reproduce the sealed P1 root")
            pixels, goal, history = p0._prepare(
                preprocessor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], dtype=np.float32), device,
            )
            del pixels, history
            goal_latent = p0._goal_latent(model, goal)
            candidate = _evaluate_packet(
                name="reconfiguration", model=model, preprocessor=preprocessor, env=env, context=context,
                normalized=np.asarray(method["first_action_witness_normalized"], dtype=np.float32),
                raw=np.asarray(method["first_action_witness_raw"], dtype=np.float32), action_block=action_block,
                device=device, goal_latent=goal_latent, action_mean=mean, action_std=std,
            )
            zero = _evaluate_packet(
                name="raw_zero_prefix", model=model, preprocessor=preprocessor, env=env, context=context,
                normalized=np.asarray(method["zero_prefix_action_witness_normalized"], dtype=np.float32),
                raw=np.asarray(method["zero_prefix_action_witness_raw"], dtype=np.float32), action_block=action_block,
                device=device, goal_latent=goal_latent, action_mean=mean, action_std=std,
            )
            expected_candidate = method["observed_successor_context"]
            expected_zero = method["zero_prefix_successor_context"]
            candidate_successor_replayed = all(candidate["advanced_context"][key] == expected_candidate[key] for key in expected_candidate)
            zero_successor_replayed = all(zero["advanced_context"][key] == expected_zero[key] for key in expected_zero)
            rows.append({
                "seed": seed,
                "warmup_context_replayed": context_replayed,
                "candidate_successor_context_replayed": candidate_successor_replayed,
                "zero_successor_context_replayed": zero_successor_replayed,
                "candidate": candidate,
                "zero_prefix": zero,
                "model_candidate_minus_zero_goal_cost": candidate["predicted_goal_cost"] - zero["predicted_goal_cost"],
                "observed_candidate_minus_zero_goal_cost": candidate["observed_encoded_goal_cost"] - zero["observed_encoded_goal_cost"],
                "model_to_observation_goal_order_agrees": _strict_order_agrees(
                    candidate["predicted_goal_cost"], zero["predicted_goal_cost"],
                    candidate["observed_encoded_goal_cost"], zero["observed_encoded_goal_cost"],
                ),
            })
    finally:
        env.close()
    gates = {
        "cuda_only": True,
        "all_warmup_contexts_replay": bool(rows) and all(row["warmup_context_replayed"] for row in rows),
        "all_candidate_successor_contexts_replay": bool(rows) and all(row["candidate_successor_context_replayed"] for row in rows),
        "all_zero_successor_contexts_replay": bool(rows) and all(row["zero_successor_context_replayed"] for row in rows),
        "all_latents_finite": bool(rows) and all(
            packet["predicted_latent_finite"] and packet["observed_latent_finite"]
            for row in rows for packet in (row["candidate"], row["zero_prefix"])
        ),
        "all_recorded_normalized_packets_invert_to_executed_raw_actions": bool(rows) and all(
            packet["raw_from_normalized_max_abs_error"] <= args.raw_action_atol
            for row in rows for packet in (row["candidate"], row["zero_prefix"])
        ),
    }
    result = {
        "schema": "d2p_reacher_p1_first_chunk_fidelity_audit.v1",
        "status": "completed" if all(gates.values()) else "fail",
        "scope": "CUDA-only evaluator replay of sealed P1 candidate and raw-zero packets; qpos is never read, no action is selected, and no efficacy conclusion is made",
        "source_report": str(report_path),
        "source_report_sha256": _sha256(report_path),
        "source_binding": {
            **runtime_input_binding,
            "target_delta": list(target_delta),
            "audit_runner_sha256": _sha256(Path(__file__)),
        },
        "cuda_host": {"name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "raw_action_atol": args.raw_action_atol,
        "gates": gates,
        "summary": {
            "eligible_roots_audited": len(rows),
            "model_to_observation_goal_order_agreement_count": int(sum(row["model_to_observation_goal_order_agrees"] for row in rows)),
            "candidate_mean_latent_l2_error": float(np.mean([row["candidate"]["latent_l2_error"] for row in rows])) if rows else None,
            "zero_mean_latent_l2_error": float(np.mean([row["zero_prefix"]["latent_l2_error"] for row in rows])) if rows else None,
        },
        "rows": rows,
        "claim_boundary": "An ordering disagreement is evidence of a first-chunk model-to-observation calibration failure for this frozen checkpoint/protocol, not a P1 efficacy result or a theoretical negative. Agreement only rules out this particular execution/observation path defect; it does not establish downstream planning efficacy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "gates": gates, "summary": result["summary"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
