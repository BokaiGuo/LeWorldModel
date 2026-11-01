"""CUDA-only model-side P1 admission for the independent frozen TwoRoom path.

P1 selects a short prefix that is deliberately less goal-greedy in the frozen
model, but leaves a lower local suffix reachability value.  No candidate is
executed here: this receipt only seals model-side selections for a fresh,
root-disjoint physical calibration.  It is intentionally independent of all
legacy adapters, teachers, readouts, and reactive policies.
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
from experiments.spectral_reachability.build_tworoom_d2p_training_action_stats import validate_training_semantic_receipt
from experiments.spectral_reachability import tworoom_d2p_packet_support as packet_support


# v2 makes the source-supported, same-feasible-set physical comparator part
# of the sealed preflight contract.  Raw zero remains recorded as an explicit
# OOD diagnostic, but it is not a member of the training-packet proposal bank.
SCHEMA = "tworoom_d2p_p1_reconfiguration_preflight.v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_p0(path: Path) -> dict[str, Any]:
    report_path = path / "report.json" if path.is_dir() else path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != p0.SCHEMA or report.get("status") != "pass" or not all(report.get("gates", {}).values()):
        raise ValueError("TwoRoom P1 requires a passing immutable D2P P0 receipt")
    return report


def _root_context(
    model: torch.nn.Module, context: dict[str, Any], *, mean: np.ndarray, std: np.ndarray,
    action_block: int, action_dim: int, transform: Any, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode observed root frames and retain the causal action shift explicitly."""

    frames = [np.asarray(frame, dtype=np.uint8) for frame in context["frames"][1:]]
    frames.append(np.asarray(context["root_frame"], dtype=np.uint8))
    raw_history = np.asarray(context["history_raw_packets"], dtype=np.float32)
    if len(frames) != raw_history.shape[0] or len(frames) < 2:
        raise ValueError("root-anchored TwoRoom RGB/action context is incompatible")
    pixels = p0._frames_tensor(frames, transform=transform, device=device)
    tokens = p0._normalize_raw_packets(
        torch.as_tensor(raw_history, dtype=torch.float32, device=device), mean, std,
        action_block=action_block, action_dim=action_dim,
    )[None]
    with torch.no_grad():
        emb = model.encode({"pixels": pixels, "action": tokens})["emb"].detach()
    goal = p0._goal_latent(model, context["goal"], transform=transform, device=device)
    return emb, tokens, goal


def _proposal_bank(*, pool: np.ndarray, proposals: int, horizon: int, seed: int, low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Predeclare source-supported packets; raw zero remains the fixed control."""

    return packet_support.support_proposal_parameters(pool, proposals=proposals, horizon=horizon, seed=seed, low=low, high=high)


def _bounded_packets(params: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Apply P0's physical box map without collapsing a packet time axis."""

    if params.ndim != 4 or params.shape[-1] != low.numel():
        raise ValueError("TwoRoom packet parameters must be [batch, horizon, step, action]")
    flattened = params.reshape(-1, 1, params.shape[-1])
    return p0._bounded_live_actions(flattened, low, high).reshape_as(params)


def _advance(
    model: torch.nn.Module, emb: torch.Tensor, actions: torch.Tensor, prefix_params: torch.Tensor,
    *, mean: np.ndarray, std: np.ndarray, action_block: int, action_dim: int,
    low: torch.Tensor, high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance root-anchored state using only the selected candidate packet."""

    if prefix_params.ndim != 4 or emb.shape[0] != 1 or actions.shape[:2] != emb.shape[:2]:
        raise ValueError("invalid TwoRoom P1 prefix/context layout")
    samples, horizon = prefix_params.shape[:2]
    prefix = p0._normalize_raw_packets(
        _bounded_packets(prefix_params, low, high).reshape(-1, action_block, action_dim), mean, std,
        action_block=action_block, action_dim=action_dim,
    ).reshape(samples, horizon, -1)
    h = int(emb.shape[1])
    latent = emb.expand(samples, -1, -1)
    # x1,x2,x3 are paired with a1,a2,prefix0; a0 is intentionally discarded.
    action_context = torch.cat((actions[:, 1:].expand(samples, -1, -1), prefix[:, :1]), dim=1)
    for step in range(horizon):
        predicted = model.predict(latent[:, -h:], model.action_encoder(action_context[:, -h:]))[:, -1]
        latent = torch.cat((latent, predicted[:, None]), dim=1)
        if step + 1 < horizon:
            action_context = torch.cat((action_context, prefix[:, step + 1:step + 2]), dim=1)
    return latent[:, -h:], action_context[:, -h:]


def _suffix_terminal(
    model: torch.nn.Module, emb: torch.Tensor, actions: torch.Tensor, params: torch.Tensor,
    *, mean: np.ndarray, std: np.ndarray, action_block: int, action_dim: int,
    low: torch.Tensor, high: torch.Tensor,
) -> torch.Tensor:
    if params.ndim != 4 or params.shape[0] != emb.shape[0]:
        raise ValueError("TwoRoom P1 suffix does not match state batch")
    samples, horizon = params.shape[:2]
    future = p0._normalize_raw_packets(
        _bounded_packets(params, low, high).reshape(-1, action_block, action_dim), mean, std,
        action_block=action_block, action_dim=action_dim,
    ).reshape(samples, horizon, -1)
    h, latent, current_actions = int(emb.shape[1]), emb, actions
    terminal = None
    for step in range(horizon):
        current_actions = torch.cat((current_actions, future[:, step:step + 1]), dim=1)
        terminal = model.predict(latent[:, -h:], model.action_encoder(current_actions[:, -h:]))[:, -1]
        if step + 1 < horizon:
            latent = torch.cat((latent, terminal[:, None]), dim=1)
    if terminal is None:
        raise RuntimeError("TwoRoom P1 suffix horizon is empty")
    return terminal


def _reachability(
    model: torch.nn.Module, goal: torch.Tensor, emb: torch.Tensor, actions: torch.Tensor,
    *, suffix_horizon: int, damping: float, sketch: int, seed: int,
    mean: np.ndarray, std: np.ndarray, action_block: int, action_dim: int, low: torch.Tensor, high: torch.Tensor,
) -> dict[str, Any]:
    flat = torch.zeros((suffix_horizon, action_block, action_dim), device=emb.device, requires_grad=True)
    with torch.no_grad():
        dim = int(_suffix_terminal(model, emb, actions, flat.detach()[None], mean=mean, std=std, action_block=action_block, action_dim=action_dim, low=low, high=high)[0].numel())
    generator = torch.Generator(device=emb.device).manual_seed(int(seed))
    projection = torch.randn((min(sketch, dim), dim), generator=generator, device=emb.device)
    projection = projection / projection.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def terminal(value: torch.Tensor) -> torch.Tensor:
        return _suffix_terminal(model, emb, actions, value.reshape(1, suffix_horizon, action_block, action_dim), mean=mean, std=std, action_block=action_block, action_dim=action_dim, low=low, high=high)[0]

    predicted = terminal(flat)
    jacobian = torch.autograd.functional.jacobian(lambda value: projection @ terminal(value.reshape(suffix_horizon, action_block, action_dim)), flat.reshape(-1), vectorize=True)
    if not torch.isfinite(jacobian).all() or float(jacobian.norm()) <= 1e-10:
        raise RuntimeError("TwoRoom P1 post-prefix Jacobian is non-finite or zero")
    left, singular, right_h = torch.linalg.svd(jacobian, full_matrices=False)
    residual = projection @ (goal - predicted)
    coordinate = left.T @ residual
    orthogonal = residual - left @ coordinate
    value = orthogonal.square().sum() + (float(damping) / (singular.square() + float(damping)) * coordinate.square()).sum()
    update = right_h.T @ (singular / (singular.square() + float(damping)) * coordinate)
    with torch.no_grad():
        bounded = terminal(update)
    return {
        "reachability_value": float(value.detach()),
        "immediate_suffix_goal_cost": float((predicted - goal).square().mean().detach()),
        "bounded_dls_suffix_goal_cost": float((bounded - goal).square().mean().detach()),
        "jacobian_frobenius_norm": float(jacobian.norm().detach()),
        "dls_raw_packet": _bounded_packets(update.reshape(1, suffix_horizon, action_block, action_dim), low, high)[0].detach().cpu().tolist(),
    }


def compile_reconfiguration(
    model: torch.nn.Module, context: dict[str, Any], *, mean: np.ndarray, std: np.ndarray, action_block: int,
    action_dim: int, transform: Any, device: torch.device, proposals: int, prefix_horizon: int,
    suffix_horizon: int, damping: float, sketch: int, seed: int, low: torch.Tensor, high: torch.Tensor, packet_pool: np.ndarray,
) -> dict[str, Any]:
    emb, actions, goal = _root_context(model, context, mean=mean, std=std, action_block=action_block, action_dim=action_dim, transform=transform, device=device)
    bank, support = _proposal_bank(pool=packet_pool, proposals=proposals, horizon=prefix_horizon, seed=seed, low=low, high=high)
    states, state_actions = _advance(model, emb, actions, bank, mean=mean, std=std, action_block=action_block, action_dim=action_dim, low=low, high=high)
    immediate_cost = (states[:, -1] - goal[None]).square().mean(dim=1)
    zero_cost = float(immediate_cost[0].detach())
    rows = []
    for index in range(proposals):
        receipt = _reachability(model, goal, states[index:index + 1], state_actions[index:index + 1], suffix_horizon=suffix_horizon, damping=damping, sketch=sketch, seed=seed + 1000 + index, mean=mean, std=std, action_block=action_block, action_dim=action_dim, low=low, high=high)
        rows.append({"proposal_index": index, "immediate_goal_cost": float(immediate_cost[index].detach()), **receipt})
    zero_value = rows[0]["reachability_value"]
    eligible = [row for row in rows[1:] if row["immediate_goal_cost"] > zero_cost + 1e-8 and row["reachability_value"] < zero_value - 1e-8]
    selected = min(eligible, key=lambda row: row["reachability_value"]) if eligible else None
    return {
        "proposal_count": proposals,
        **support,
        "raw_zero_immediate_goal_cost": zero_cost,
        "raw_zero_reachability_value": zero_value,
        "candidate_rows": rows,
        "selected_proposal_index": None if selected is None else selected["proposal_index"],
        "selected_prefix_raw_packets": None if selected is None else _bounded_packets(bank[selected["proposal_index"]:selected["proposal_index"] + 1], low, high)[0].detach().cpu().tolist(),
        "non_greedy_witness_found": selected is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p0-evidence", type=Path, required=True)
    parser.add_argument("--training-action-stats", type=Path, default=Path("results/tworoom_d2p_training_action_stats_20261024/tworoom_training_action_stats.json"))
    parser.add_argument("--training-dataset", type=Path, default=Path("/home/gbk/.stable_worldmodel/datasets/tworoom.h5"))
    parser.add_argument("--policy", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--roots", type=int, default=4); parser.add_argument("--seed", type=int, default=202610270)
    parser.add_argument("--proposals", type=int, default=16); parser.add_argument("--prefix-horizon", type=int, default=1); parser.add_argument("--suffix-horizon", type=int, default=1)
    parser.add_argument("--damping", type=float, default=0.1); parser.add_argument("--sketch-dims", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite immutable TwoRoom P1 output: {args.output}")
    if min(args.roots, args.proposals, args.prefix_horizon, args.suffix_horizon, args.sketch_dims) < 1 or args.proposals < 2 or args.damping <= 0.0: raise ValueError("invalid TwoRoom P1 dimensions")
    p0_report = _load_p0(args.p0_evidence)
    runtime = require_cuda_allocation(args.device); device = torch.device(args.device)
    row = validate_training_semantic_receipt(args.training_action_stats)
    mean, std = np.asarray(row["calibration_action_mean"], dtype=np.float32), np.asarray(row["calibration_action_std"], dtype=np.float32)
    model = load_compatible_pretrained(args.policy.resolve(), device=device).eval(); model.requires_grad_(False)
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    action_dim = int(np.prod(env.action_space.shape)); action_block = p0._validate_tworoom_action_protocol(model, mean, std, action_dim=action_dim, action_block=5)
    receipt_dataset, receipt_sha256 = packet_support.receipt_source(args.training_action_stats)
    if args.training_dataset.resolve() != receipt_dataset.resolve(): raise ValueError("TwoRoom P1 training dataset differs from its statistic receipt")
    packet_pool, support_meta = packet_support.load_packet_pool(args.training_dataset.resolve(), expected_sha256=receipt_sha256, action_block=action_block, action_dim=action_dim)
    if p0_report.get("inputs", {}).get("training_packet_support", {}).get("pool_sha256") != support_meta["pool_sha256"]: raise ValueError("P1 source packet support differs from immutable P0 evidence")
    before = p0._parameter_hash(model); low = torch.as_tensor(env.action_space.low, device=device); high = torch.as_tensor(env.action_space.high, device=device); transform = _image_transform(); started = time.time()
    try:
        prior = set(p0_report["protocol"]["root_seeds"])
        seeds = [int(args.seed + i) for i in range(args.roots)]
        if prior & set(seeds): raise RuntimeError("P1 roots overlap P0 roots")
        rows = []
        for index, seed in enumerate(seeds):
            warmup, warmup_indices = packet_support.warmup_packets(packet_pool, history_len=3, seed=seed)
            context = p0._collect_executed_context(env, seed=seed, history_len=3, action_block=action_block, warmup_packets=warmup)
            compilation = compile_reconfiguration(model, context, mean=mean, std=std, action_block=action_block, action_dim=action_dim, transform=transform, device=device, proposals=args.proposals, prefix_horizon=args.prefix_horizon, suffix_horizon=args.suffix_horizon, damping=args.damping, sketch=args.sketch_dims, seed=seed, low=low, high=high, packet_pool=packet_pool)
            rows.append({"root_id": index, "seed": seed, "context": {key: context[key] for key in ("history_rgb_sha256", "root_rgb_sha256", "goal_rgb_sha256", "warmup_actions_sha256")} | {"training_packet_indices": warmup_indices.tolist()}, "compilation": compilation})
            print(json.dumps({"seed": seed, "selected": compilation["selected_proposal_index"], "witness": compilation["non_greedy_witness_found"]}, sort_keys=True))
    finally:
        env.close()
    gates = {"cuda_only": True, "frozen_parameter_hash_unchanged": before == p0._parameter_hash(model), "fresh_roots_disjoint_from_p0": True, "training_semantic_full_packets": True, "warmup_and_proposals_source_supported": True, "rgb_goal_and_causal_action_only": True, "no_adapter_teacher_or_privileged_ranker": True, "physical_candidates_not_executed": True, "at_least_two_model_non_greedy_witnesses": sum(bool(row["compilation"]["non_greedy_witness_found"]) for row in rows) >= 2}
    report = {"schema": SCHEMA, "status": "pass" if all(gates.values()) else "fail", "scope": "CUDA-only frozen TwoRoom P1 model-side reconfiguration admission; no physical candidate or outcome root is executed.", "cuda_host": {**runtime, "python": platform.python_version()}, "upstream_p0": {"path": str(args.p0_evidence.resolve()), "sha256": _sha256(args.p0_evidence / "report.json" if args.p0_evidence.is_dir() else args.p0_evidence)}, "training_packet_support": support_meta, "protocol": {"root_seed": args.seed, "root_count": args.roots, "history_len": 3, "action_block": action_block, "prefix_proposals": args.proposals, "prefix_horizon": args.prefix_horizon, "suffix_horizon": args.suffix_horizon, "damping": args.damping, "sketch_dims": args.sketch_dims, "selection": "strictly less immediate frozen-goal-greedy than raw-zero and lower recompiled local reachability value", "physical_mechanism_comparator": "among source-supported nonzero candidate packets sealed here, choose the minimum frozen immediate_goal_cost; raw-zero is OOD diagnostic only", "physical_execution": "forbidden until a separate root-disjoint calibration receipt"}, "gates": gates, "rows": rows, "elapsed_seconds": time.time() - started}
    args.output.mkdir(parents=True); (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
