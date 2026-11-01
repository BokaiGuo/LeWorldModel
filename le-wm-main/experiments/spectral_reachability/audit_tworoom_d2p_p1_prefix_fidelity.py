"""CUDA-only audit of TwoRoom P1 prefix timing, layout, and RGB fidelity.

This audit is deliberately downstream of immutable P1 selection: it does not
choose a prefix, fit a readout, or open outcome roots.  For each sealed prefix,
same-support immediate-greedy control, and raw-zero OOD diagnostic it
reconstructs the warmup, checks every low-level
action in the packet, compares the P1 recurrence with ``JEPA.rollout``, and
then compares that predicted successor with the embedding of the rendered
successor actually reached after the five actions.

The distinction is essential: a reversed physical goal-distance ordering is a
runtime bug only if the packet, causal shift, or model invocation contract is
wrong.  A completed audit records a model-fidelity discrepancy separately; it
never silently turns it into a theoretical claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 -- registers the environment
import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_tworoom_planning import TwoRoomGoalAliasWrapper, _image_transform
from experiments.spectral_reachability import preflight_tworoom_d2p_p0 as p0
from experiments.spectral_reachability import preflight_tworoom_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_tworoom_d2p_p1_physical_calibration as calibration
from experiments.spectral_reachability.build_tworoom_d2p_training_action_stats import validate_training_semantic_receipt
from experiments.spectral_reachability import tworoom_d2p_packet_support as packet_support


SCHEMA = "tworoom_d2p_p1_prefix_fidelity.v2"


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load(path: Path, schema: str) -> dict[str, Any]:
    report_path = path / "report.json" if path.is_dir() else path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != schema:
        raise ValueError(f"expected {schema}, got {report.get('schema')}")
    return report


def _distance(info: dict[str, Any]) -> float:
    value = np.asarray(info.get("distance_to_target", np.nan), dtype=np.float64).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise RuntimeError("TwoRoom evaluator did not provide a finite distance_to_target")
    return float(value[0])


def _predict_successor(
    model: torch.nn.Module, context: dict[str, Any], prefix: np.ndarray, *, mean: np.ndarray, std: np.ndarray,
    action_block: int, action_dim: int, transform: Any, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return P1 recurrence and native-rollout successors for one raw packet."""

    emb, actions, goal = p1._root_context(
        model, context, mean=mean, std=std, action_block=action_block, action_dim=action_dim,
        transform=transform, device=device,
    )
    raw = torch.as_tensor(prefix, dtype=torch.float32, device=device)
    if tuple(raw.shape) != (1, action_block, action_dim):
        raise ValueError("fidelity audit requires exactly one [packet, step, action] prefix")
    token = p0._normalize_raw_packets(raw, mean, std, action_block=action_block, action_dim=action_dim)[None]
    action_context = torch.cat((actions[:, 1:], token), dim=1)
    with torch.no_grad():
        recurrence = model.predict(emb, model.action_encoder(action_context))[:, -1]
        # With T == history_size, rollout performs precisely its final native
        # prediction.  This guards the hand-written P1 recurrence itself.
        pixels = p0._frames_tensor(
            [np.asarray(frame, dtype=np.uint8) for frame in context["frames"][1:]] + [np.asarray(context["root_frame"], dtype=np.uint8)],
            transform=transform, device=device,
        )
        native = model.rollout({"pixels": pixels[:, None]}, action_context[:, None], history_size=int(pixels.shape[1]))["predicted_emb"][0, 0, -1]
    layout = {
        "encoded_frames": [_digest(np.asarray(frame, dtype=np.uint8)) for frame in context["frames"][1:]] + [_digest(np.asarray(context["root_frame"], dtype=np.uint8))],
        "action_context_shape": [int(v) for v in action_context.shape],
        "action_context_sha256": _digest(action_context.detach().cpu().numpy()),
        "prefix_token_sha256": _digest(token.detach().cpu().numpy()),
    }
    return recurrence[0].detach(), native.detach(), goal.detach(), layout


def _execute_and_encode(
    env: gym.Env, context: dict[str, Any], prefix: np.ndarray, *, model: torch.nn.Module,
    mean: np.ndarray, std: np.ndarray, action_block: int, action_dim: int, transform: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Execute each raw action once and retain the exact successor context."""

    step_distances: list[float] = []
    step_hashes: list[str] = []
    for action in np.asarray(prefix, dtype=np.float32).reshape(-1, action_dim):
        _obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            raise RuntimeError("sealed P1 prefix ended the episode before its packet boundary")
        step_distances.append(_distance(info))
        step_hashes.append(_digest(np.asarray(env.render(), dtype=np.uint8)))
    successor = np.asarray(env.render(), dtype=np.uint8).copy()
    advanced = {
        "frames": [
            np.asarray(context["frames"][1], dtype=np.uint8), np.asarray(context["frames"][2], dtype=np.uint8),
            np.asarray(context["root_frame"], dtype=np.uint8),
        ],
        "history_raw_packets": np.concatenate((np.asarray(context["history_raw_packets"][1:]), np.asarray(prefix, dtype=np.float32)), axis=0),
        "root_frame": successor,
        "goal": context["goal"],
    }
    # The next compiler call must use [x2,x3,x4] and [a2,p,suffix].  Calling
    # _root_context here validates that its tensor interfaces accept the
    # precisely reconstructed post-prefix context.
    post_emb, post_actions, _goal = p1._root_context(
        model, advanced, mean=mean, std=std, action_block=action_block, action_dim=action_dim,
        transform=transform, device=device,
    )
    with torch.no_grad():
        observed = model.encode({"pixels": p0._image_tensor(successor, transform=transform, device=device)})["emb"][0, 0].detach()
    return {
        "successor": successor,
        "observed_embedding": observed,
        "step_distances": step_distances,
        "step_rgb_sha256": step_hashes,
        "post_prefix_context": {
            "frames_sha256": [_digest(np.asarray(frame, dtype=np.uint8)) for frame in advanced["frames"][1:]] + [_digest(successor)],
            "actions_sha256": _digest(np.asarray(advanced["history_raw_packets"][1:])),
            "encoded_shape": [int(v) for v in post_emb.shape],
            "action_shape": [int(v) for v in post_actions.shape],
        },
    }


def _branch(
    env: gym.Env, *, seed: int, prefix: np.ndarray, expected: dict[str, Any], model: torch.nn.Module,
    mean: np.ndarray, std: np.ndarray, action_block: int, action_dim: int, transform: Any, device: torch.device,
    packet_pool: np.ndarray,
) -> dict[str, Any]:
    warmup, warmup_indices = packet_support.warmup_packets(packet_pool, history_len=3, seed=seed)
    context = p0._collect_executed_context(env, seed=seed, history_len=3, action_block=action_block, warmup_packets=warmup)
    prediction, native, goal, layout = _predict_successor(
        model, context, prefix, mean=mean, std=std, action_block=action_block, action_dim=action_dim,
        transform=transform, device=device,
    )
    executed = _execute_and_encode(
        env, context, prefix, model=model, mean=mean, std=std, action_block=action_block,
        action_dim=action_dim, transform=transform, device=device,
    )
    observed = executed.pop("observed_embedding")
    successor = executed.pop("successor")
    recurrence_native_linf = float((prediction - native).abs().max().cpu())
    expected_prefix = np.asarray(expected["prefix_raw_packet"], dtype=np.float32)
    return {
        "prefix_raw_sha256": _digest(np.asarray(prefix, dtype=np.float32)),
        "warmup_training_packet_indices": warmup_indices.tolist(),
        "warmup_root_rgb_sha256": _digest(np.asarray(context["root_frame"], dtype=np.uint8)),
        "successor_rgb_sha256": _digest(successor),
        "prediction_contract": {
            **layout,
            "p1_recurrence_native_rollout_linf": recurrence_native_linf,
            "p1_recurrence_matches_native_rollout": recurrence_native_linf <= 1.0e-6,
            "predicted_goal_cost": float((prediction - goal).square().mean().cpu()),
            "observed_encoded_goal_cost": float((observed - goal).square().mean().cpu()),
            "latent_l2_error": float(torch.linalg.vector_norm(prediction - observed).cpu()),
        },
        "execution_contract": {
            **executed,
            "packet_boundary_distance": executed["step_distances"][-1],
            "matches_calibration_prefix": _digest(np.asarray(prefix, dtype=np.float32)) == _digest(expected_prefix),
            "matches_calibration_successor_hash": _digest(successor) == expected["successor_rgb_sha256"],
            "matches_calibration_packet_distance": bool(np.isclose(executed["step_distances"][-1], expected["after_prefix_distance"], atol=1.0e-8)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p1-evidence", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--training-action-stats", type=Path, default=Path("results/tworoom_d2p_training_action_stats_20261024/tworoom_training_action_stats.json"))
    parser.add_argument("--training-dataset", type=Path, default=Path("/home/gbk/.stable_worldmodel/datasets/tworoom.h5"))
    parser.add_argument("--policy", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable TwoRoom P1 fidelity output: {args.output}")
    preflight = _load(args.p1_evidence, p1.SCHEMA)
    calibration_report = _load(args.calibration, calibration.SCHEMA)
    runtime = require_cuda_allocation(args.device)
    device = torch.device(args.device)
    stat = validate_training_semantic_receipt(args.training_action_stats)
    mean, std = np.asarray(stat["calibration_action_mean"], dtype=np.float32), np.asarray(stat["calibration_action_std"], dtype=np.float32)
    model = load_compatible_pretrained(args.policy.resolve(), device=device).eval()
    model.requires_grad_(False)
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    action_dim = int(np.prod(env.action_space.shape))
    action_block = p0._validate_tworoom_action_protocol(model, mean, std, action_dim=action_dim, action_block=int(preflight["protocol"]["action_block"]))
    before = p0._parameter_hash(model)
    transform = _image_transform()
    receipt_dataset, receipt_sha256 = packet_support.receipt_source(args.training_action_stats)
    if args.training_dataset.resolve() != receipt_dataset.resolve():
        raise ValueError("TwoRoom fidelity audit training dataset differs from its statistic receipt")
    packet_pool, support_meta = packet_support.load_packet_pool(args.training_dataset.resolve(), expected_sha256=receipt_sha256, action_block=action_block, action_dim=action_dim)
    if support_meta["pool_sha256"] != preflight.get("training_packet_support", {}).get("pool_sha256") or support_meta["pool_sha256"] != calibration_report.get("training_packet_support", {}).get("pool_sha256"):
        raise ValueError("TwoRoom fidelity audit source packet support differs from its sealed upstream receipts")
    started = time.time()
    try:
        calibration_rows = {int(row["seed"]): row for row in calibration_report["rows"]}
        rows: list[dict[str, Any]] = []
        for sealed in preflight["rows"]:
            seed = int(sealed["seed"])
            calibration_row = calibration_rows.get(seed)
            if calibration_row is None:
                raise ValueError(f"calibration does not contain sealed P1 root {seed}")
            selected = sealed["compilation"]["selected_prefix_raw_packets"]
            selected_prefix = np.zeros((1, action_block, action_dim), dtype=np.float32) if selected is None else np.asarray(selected, dtype=np.float32)
            selected_result = _branch(env, seed=seed, prefix=selected_prefix, expected=calibration_row["selected"], model=model, mean=mean, std=std, action_block=action_block, action_dim=action_dim, transform=transform, device=device, packet_pool=packet_pool)
            greedy_result = None
            greedy_recovery_matches_calibration = None
            if selected is not None:
                greedy_receipt = calibration._same_bank_greedy_prefix(
                    sealed["compilation"], packet_pool, prefix_horizon=int(preflight["protocol"]["prefix_horizon"]),
                    action_block=action_block, action_dim=action_dim,
                )
                recorded_greedy = calibration_row.get("source_matched_greedy_receipt")
                if not isinstance(recorded_greedy, dict):
                    raise ValueError("model-eligible calibration row omits its source-matched greedy receipt")
                greedy_recovery_matches_calibration = bool(
                    greedy_receipt["proposal_index"] == recorded_greedy.get("proposal_index")
                    and greedy_receipt["source_indices"] == recorded_greedy.get("source_indices")
                    and np.array_equal(np.asarray(greedy_receipt["prefix_raw_packets"], dtype=np.float32), np.asarray(recorded_greedy.get("prefix_raw_packets"), dtype=np.float32))
                )
                greedy_result = _branch(env, seed=seed, prefix=np.asarray(greedy_receipt["prefix_raw_packets"], dtype=np.float32), expected=calibration_row["source_matched_greedy"], model=model, mean=mean, std=std, action_block=action_block, action_dim=action_dim, transform=transform, device=device, packet_pool=packet_pool)
            zero_prefix = np.zeros_like(selected_prefix)
            zero_result = _branch(env, seed=seed, prefix=zero_prefix, expected=calibration_row["raw_zero_ood_diagnostic"], model=model, mean=mean, std=std, action_block=action_block, action_dim=action_dim, transform=transform, device=device, packet_pool=packet_pool)
            rows.append({"seed": seed, "model_eligible": selected is not None, "selected": selected_result, "source_matched_greedy": greedy_result, "source_matched_greedy_recovery_matches_calibration": greedy_recovery_matches_calibration, "raw_zero_ood_diagnostic": zero_result})
            print(json.dumps({"seed": seed, "selected_l2": selected_result["prediction_contract"]["latent_l2_error"], "same_bank_greedy_l2": None if greedy_result is None else greedy_result["prediction_contract"]["latent_l2_error"], "zero_ood_l2": zero_result["prediction_contract"]["latent_l2_error"]}, sort_keys=True))
    finally:
        env.close()
    branches = [branch for row in rows for branch in (row["selected"], row["source_matched_greedy"], row["raw_zero_ood_diagnostic"]) if branch is not None]
    gates = {
        "cuda_only": True,
        "frozen_parameter_hash_unchanged": before == p0._parameter_hash(model),
        "all_prefixes_have_exactly_five_low_level_labels": all(len(branch["execution_contract"]["step_distances"]) == action_block for branch in branches),
        "prefix_bytes_match_calibration": all(branch["execution_contract"]["matches_calibration_prefix"] for branch in branches),
        "packet_boundary_label_matches_calibration": all(branch["execution_contract"]["matches_calibration_packet_distance"] for branch in branches),
        "successor_rgb_matches_calibration": all(branch["execution_contract"]["matches_calibration_successor_hash"] for branch in branches),
        "source_matched_greedy_recovery_matches_calibration": all(row["source_matched_greedy_recovery_matches_calibration"] for row in rows if row["model_eligible"]),
        "p1_recurrence_matches_native_rollout": all(branch["prediction_contract"]["p1_recurrence_matches_native_rollout"] for branch in branches),
        "post_prefix_context_is_three_frames_and_three_packets": all(branch["execution_contract"]["post_prefix_context"]["encoded_shape"][1] == 3 and branch["execution_contract"]["post_prefix_context"]["action_shape"][1] == 3 for branch in branches),
    }
    report = {
        "schema": SCHEMA,
        "status": "completed" if all(gates.values()) else "contract_failure",
        "scope": "CUDA-only audit; model/physical prediction residuals are diagnostics, not selection labels or an outcome result.",
        "cuda_host": {**runtime, "python": platform.python_version()},
        "upstream": {"p1": str(args.p1_evidence.resolve()), "calibration": str(args.calibration.resolve())},
        "training_packet_support": support_meta,
        "gates": gates,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
