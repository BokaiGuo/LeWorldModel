"""Physical calibration for sealed TwoRoom D2P-P1 prefix selections.

This consumes one immutable P1 preflight.  It never reselects a prefix using
physical labels: selected and raw-zero prefixes are replayed from the same
causal warmup, each observed RGB successor is recompiled with the frozen
model, and environment distance/success remain evaluator-only labels.
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
import stable_worldmodel  # noqa: F401
import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_tworoom_planning import TwoRoomGoalAliasWrapper, _image_transform
from experiments.spectral_reachability import preflight_tworoom_d2p_p0 as p0
from experiments.spectral_reachability import preflight_tworoom_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability.build_tworoom_d2p_training_action_stats import validate_training_semantic_receipt
from experiments.spectral_reachability import tworoom_d2p_packet_support as packet_support


SCHEMA = "tworoom_d2p_p1_physical_calibration.v2"


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_preflight(path: Path) -> dict[str, Any]:
    report_path = path / "report.json" if path.is_dir() else path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != p1.SCHEMA or report.get("status") != "pass" or not all(report.get("gates", {}).values()):
        raise ValueError("physical calibration requires a passing immutable TwoRoom P1 preflight")
    return report


def _distance(info: dict[str, Any]) -> float:
    value = np.asarray(info.get("distance_to_target", np.nan), dtype=np.float64).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise RuntimeError("TwoRoom evaluator did not provide a finite distance_to_target")
    return float(value[0])


def _same_bank_greedy_prefix(compilation: dict[str, Any], packet_pool: np.ndarray, *, prefix_horizon: int, action_block: int, action_dim: int) -> dict[str, Any]:
    """Recover the sealed source-supported immediate-greedy control.

    Index zero is raw zero, intentionally outside the source packet support.
    This helper selects only among the already-scored, nonzero source rows;
    it never evaluates a physical label and uses proposal index as a stable
    deterministic tie-break.  The selected source indices rehydrate the exact
    packet from the pool whose hash is bound by the upstream preflight.
    """

    rows = compilation.get("candidate_rows")
    source_indices = compilation.get("proposal_source_indices")
    pool = np.asarray(packet_pool, dtype=np.float32)
    if not isinstance(rows, list) or not isinstance(source_indices, list) or prefix_horizon < 1:
        raise ValueError("sealed TwoRoom P1 compilation has no source-supported candidate bank")
    if pool.ndim != 3 or tuple(pool.shape[1:]) != (action_block, action_dim):
        raise ValueError("TwoRoom source packet pool layout is incompatible with sealed P1")
    expected_indices = set(range(1, len(source_indices) + 1))
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("sealed TwoRoom candidate row is malformed")
        index = row.get("proposal_index")
        cost = row.get("immediate_goal_cost")
        if not isinstance(index, int) or not isinstance(cost, (int, float)) or not np.isfinite(cost) or index in by_index:
            raise ValueError("sealed TwoRoom candidate row has invalid index or immediate cost")
        by_index[index] = row
    if set(by_index) != {0, *expected_indices} or len(rows) != len(source_indices) + 1:
        raise ValueError("sealed TwoRoom candidate rows do not exactly cover the source proposal bank")
    expected_source_shape = (prefix_horizon,)
    recovered: list[tuple[dict[str, Any], np.ndarray]] = []
    for proposal_index in sorted(expected_indices):
        indices = np.asarray(source_indices[proposal_index - 1], dtype=np.int64)
        if indices.shape != expected_source_shape or np.any(indices < 0) or np.any(indices >= len(pool)):
            raise ValueError("sealed TwoRoom source proposal indices are invalid")
        recovered.append((by_index[proposal_index], np.ascontiguousarray(pool[indices], dtype=np.float32)))
    row, prefix = min(recovered, key=lambda item: (float(item[0]["immediate_goal_cost"]), int(item[0]["proposal_index"])))
    return {
        "proposal_index": int(row["proposal_index"]),
        "immediate_goal_cost": float(row["immediate_goal_cost"]),
        "source_indices": np.asarray(source_indices[int(row["proposal_index"]) - 1], dtype=np.int64).tolist(),
        "prefix_raw_packets": prefix.tolist(),
    }


def _execute_packet(env: gym.Env, packet: np.ndarray) -> tuple[np.ndarray, dict[str, Any], bool]:
    info: dict[str, Any] = {}
    terminated = truncated = False
    steps = np.asarray(packet, dtype=np.float32).reshape(-1, env.action_space.shape[0])
    if steps.shape[0] < 1 or not np.isfinite(steps).all():
        raise ValueError("TwoRoom executed packet must contain finite physical actions")
    for action in steps:
        _obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return np.asarray(env.render(), dtype=np.uint8).copy(), info, bool(terminated)


def _suffix_packet(
    model: torch.nn.Module, context: dict[str, Any], *, mean: np.ndarray, std: np.ndarray,
    action_block: int, action_dim: int, damping: float, sketch: int, seed: int,
    transform: Any, device: torch.device, low: torch.Tensor, high: torch.Tensor,
) -> tuple[np.ndarray, dict[str, Any]]:
    emb, actions, goal = p1._root_context(model, context, mean=mean, std=std, action_block=action_block, action_dim=action_dim, transform=transform, device=device)
    receipt = p1._reachability(model, goal, emb, actions, suffix_horizon=1, damping=damping, sketch=sketch, seed=seed, mean=mean, std=std, action_block=action_block, action_dim=action_dim, low=low, high=high)
    packet = np.asarray(receipt["dls_raw_packet"], dtype=np.float32)[0]
    return packet, receipt


def _branch(
    env: gym.Env, *, seed: int, prefix: np.ndarray, model: torch.nn.Module,
    mean: np.ndarray, std: np.ndarray, action_block: int, action_dim: int,
    damping: float, sketch: int, transform: Any, device: torch.device, low: torch.Tensor, high: torch.Tensor,
    packet_pool: np.ndarray,
) -> dict[str, Any]:
    prefix = np.asarray(prefix, dtype=np.float32)
    if prefix.shape != (1, action_block, action_dim):
        raise ValueError("TwoRoom P1 calibration seals exactly one full prefix packet")
    warmup, warmup_indices = packet_support.warmup_packets(packet_pool, history_len=3, seed=seed)
    context = p0._collect_executed_context(env, seed=seed, history_len=3, action_block=action_block, warmup_packets=warmup)
    root_before = np.asarray(context["root_frame"], dtype=np.uint8)
    prefix_root, prefix_info, prefix_success = _execute_packet(env, prefix)
    if bool(prefix_info.get("truncated", False)):
        raise RuntimeError("TwoRoom prefix truncated before suffix recompilation")
    advanced = {
        "frames": [np.asarray(context["frames"][1], dtype=np.uint8), np.asarray(context["frames"][2], dtype=np.uint8), root_before],
        "history_raw_packets": np.concatenate((np.asarray(context["history_raw_packets"][1:]), np.asarray(prefix, dtype=np.float32)), axis=0),
        "root_frame": prefix_root,
        "goal": context["goal"],
    }
    suffix, suffix_receipt = _suffix_packet(model, advanced, mean=mean, std=std, action_block=action_block, action_dim=action_dim, damping=damping, sketch=sketch, seed=seed + 40_000, transform=transform, device=device, low=low, high=high)
    final_root, final_info, final_success = _execute_packet(env, suffix)
    return {
        "warmup_root_rgb_sha256": _digest(root_before), "warmup_training_packet_indices": warmup_indices.tolist(), "successor_rgb_sha256": _digest(prefix_root), "final_rgb_sha256": _digest(final_root),
        "prefix_raw_packet": np.asarray(prefix, dtype=np.float32).tolist(), "suffix_raw_packet": suffix.tolist(),
        "after_prefix_distance": _distance(prefix_info), "final_distance": _distance(final_info),
        "after_prefix_success": prefix_success, "final_success": final_success,
        "suffix_recompile": suffix_receipt,
    }


def _same_replay(left: dict[str, Any], right: dict[str, Any]) -> bool:
    hashes = ("warmup_root_rgb_sha256", "successor_rgb_sha256", "final_rgb_sha256")
    return bool(all(left[key] == right[key] for key in hashes) and np.isclose(left["after_prefix_distance"], right["after_prefix_distance"], atol=1e-8) and np.isclose(left["final_distance"], right["final_distance"], atol=1e-8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--p1-evidence", type=Path, required=True)
    parser.add_argument("--training-action-stats", type=Path, default=Path("results/tworoom_d2p_training_action_stats_20261024/tworoom_training_action_stats.json")); parser.add_argument("--training-dataset", type=Path, default=Path("/home/gbk/.stable_worldmodel/datasets/tworoom.h5")); parser.add_argument("--policy", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm")); parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite immutable TwoRoom P1 calibration output: {args.output}")
    preflight = _load_preflight(args.p1_evidence); protocol = preflight["protocol"]
    runtime = require_cuda_allocation(args.device); device = torch.device(args.device)
    stat_row = validate_training_semantic_receipt(args.training_action_stats); mean, std = np.asarray(stat_row["calibration_action_mean"], dtype=np.float32), np.asarray(stat_row["calibration_action_std"], dtype=np.float32)
    model = load_compatible_pretrained(args.policy.resolve(), device=device).eval(); model.requires_grad_(False)
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array")); action_dim = int(np.prod(env.action_space.shape)); action_block = p0._validate_tworoom_action_protocol(model, mean, std, action_dim=action_dim, action_block=int(protocol["action_block"])); low, high = torch.as_tensor(env.action_space.low, device=device), torch.as_tensor(env.action_space.high, device=device); transform = _image_transform(); before = p0._parameter_hash(model); started = time.time()
    receipt_dataset, receipt_sha256 = packet_support.receipt_source(args.training_action_stats)
    if args.training_dataset.resolve() != receipt_dataset.resolve(): raise ValueError("TwoRoom calibration training dataset differs from its statistic receipt")
    packet_pool, support_meta = packet_support.load_packet_pool(args.training_dataset.resolve(), expected_sha256=receipt_sha256, action_block=action_block, action_dim=action_dim)
    if support_meta["pool_sha256"] != preflight.get("training_packet_support", {}).get("pool_sha256"): raise ValueError("TwoRoom calibration source packet support differs from sealed P1 preflight")
    try:
        rows = []
        for sealed in preflight["rows"]:
            seed, compilation = int(sealed["seed"]), sealed["compilation"]
            selected = compilation["selected_prefix_raw_packets"]
            model_eligible = selected is not None
            prefix = np.zeros((int(protocol["prefix_horizon"]), action_block, action_dim), dtype=np.float32) if selected is None else np.asarray(selected, dtype=np.float32)
            greedy = None if not model_eligible else _same_bank_greedy_prefix(
                compilation, packet_pool, prefix_horizon=int(protocol["prefix_horizon"]), action_block=action_block, action_dim=action_dim,
            )
            selected_branch = _branch(env, seed=seed, prefix=prefix, model=model, mean=mean, std=std, action_block=action_block, action_dim=action_dim, damping=float(protocol["damping"]), sketch=int(protocol["sketch_dims"]), transform=transform, device=device, low=low, high=high, packet_pool=packet_pool)
            zero_branch = _branch(env, seed=seed, prefix=np.zeros_like(prefix), model=model, mean=mean, std=std, action_block=action_block, action_dim=action_dim, damping=float(protocol["damping"]), sketch=int(protocol["sketch_dims"]), transform=transform, device=device, low=low, high=high, packet_pool=packet_pool)
            greedy_branch = None if greedy is None else _branch(env, seed=seed, prefix=np.asarray(greedy["prefix_raw_packets"], dtype=np.float32), model=model, mean=mean, std=std, action_block=action_block, action_dim=action_dim, damping=float(protocol["damping"]), sketch=int(protocol["sketch_dims"]), transform=transform, device=device, low=low, high=high, packet_pool=packet_pool)
            replay = _branch(env, seed=seed, prefix=prefix, model=model, mean=mean, std=std, action_block=action_block, action_dim=action_dim, damping=float(protocol["damping"]), sketch=int(protocol["sketch_dims"]), transform=transform, device=device, low=low, high=high, packet_pool=packet_pool)
            rows.append({"seed": seed, "model_eligible": model_eligible, "selected": selected_branch, "source_matched_greedy": greedy_branch, "source_matched_greedy_receipt": greedy, "raw_zero_ood_diagnostic": zero_branch, "selected_independent_replay_matches": _same_replay(selected_branch, replay)})
            print(json.dumps({"seed": seed, "eligible": model_eligible, "selected_after_prefix": selected_branch["after_prefix_distance"], "same_bank_greedy_after_prefix": None if greedy_branch is None else greedy_branch["after_prefix_distance"], "zero_ood_after_prefix": zero_branch["after_prefix_distance"], "replay": rows[-1]["selected_independent_replay_matches"]}, sort_keys=True))
    finally:
        env.close()
    eligible = [row for row in rows if row["model_eligible"]]
    gates = {"cuda_only": True, "frozen_parameter_hash_unchanged": before == p0._parameter_hash(model), "source_packet_support_matches_p1": True, "sealed_roots_and_prefixes_consumed_without_reselection": True, "same_root_source_matched_greedy_controls": all(row["source_matched_greedy"] is not None for row in eligible), "raw_zero_retained_as_ood_diagnostic": True, "observed_successors_recompiled": all(row["selected"]["suffix_recompile"]["jacobian_frobenius_norm"] > 1e-10 and row["raw_zero_ood_diagnostic"]["suffix_recompile"]["jacobian_frobenius_norm"] > 1e-10 and (not row["model_eligible"] or row["source_matched_greedy"]["suffix_recompile"]["jacobian_frobenius_norm"] > 1e-10) for row in rows), "at_least_two_model_eligible_roots": len(eligible) >= 2, "selected_prefixes_physically_no_more_goal_greedy_than_same_bank_greedy": bool(eligible) and all(row["selected"]["after_prefix_distance"] >= row["source_matched_greedy"]["after_prefix_distance"] - 1e-8 for row in eligible), "independent_replay_exact": all(row["selected_independent_replay_matches"] for row in rows)}
    report = {"schema": SCHEMA, "status": "pass" if all(gates.values()) else "fail", "scope": "CUDA-only TwoRoom D2P P1 calibration; it validates the reconfiguration mechanism only and does not open P2/P5 outcome roots.", "cuda_host": {**runtime, "python": platform.python_version()}, "upstream_p1": str(args.p1_evidence.resolve()), "training_packet_support": support_meta, "protocol": {"cohort": "calibration", "root_seeds": [row["seed"] for row in rows], "outcome_roots_opened": False, "evaluator_only_labels": "distance_to_target and success after physical execution", "formal_mechanism_comparator": "same source-supported candidate bank, minimum sealed frozen immediate_goal_cost; raw-zero is OOD diagnostic only", "stages": ["sealed prefix", "observed successor RGB recompile", "DLS suffix"]}, "gates": gates, "rows": rows, "elapsed_seconds": time.time() - started}
    args.output.mkdir(parents=True); (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
