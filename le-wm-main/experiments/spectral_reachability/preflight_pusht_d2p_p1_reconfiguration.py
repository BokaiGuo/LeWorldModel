"""CUDA-only model-side admission for an independent PushT D2P P1 branch.

The runner deliberately opens only fresh *calibration* roots.  It compiles a
bounded non-greedy prefix from frozen RGB dynamics and asks whether the
predicted post-prefix local reachability value improves.  It never executes a
candidate, reads an oracle successor, or opens the reserved outcome cohort;
those are separate P1 calibration and outcome stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_reachability_horizon import collect_nonterminal_boundary_contact_roots
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.online_planning_protocol import validate_action_protocol
from experiments.pusht_protocol_contracts import root_fingerprint
from experiments.run_pusht_counterfactual import image_transform
from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0


SCHEMA = "pusht_d2p_p1_reconfiguration_preflight.v3"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_p0(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != p0.SCHEMA or report.get("status") != "pass" or not all(report.get("gates", {}).values()):
        raise ValueError("P1 requires a passing immutable PushT D2P P0 receipt")
    return report


def _encode_context(model: torch.nn.Module, frames: list[np.ndarray], actions: np.ndarray, transform: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = p0._frames_tensor(frames, transform=transform, device=device)
    action_tokens = p0._model_action_tokens(torch.as_tensor(actions, dtype=torch.float32, device=device), action_block=5)[None]
    if pixels.shape[1] != action_tokens.shape[1]:
        raise ValueError("P1 causal RGB/action context is misaligned")
    with torch.no_grad():
        emb = model.encode({"pixels": pixels, "action": action_tokens})["emb"].detach()
    return emb, action_tokens


def _batched_live_action_tokens(params: torch.Tensor, *, action_block: int) -> torch.Tensor:
    """Map batched compiler coordinates to released PushT action packets.

    P0's public helper accepts a single ``[horizon, 2]`` sequence and emits
    ``[horizon, 10]`` packets.  P1 evaluates a bank of prefixes in parallel,
    so applying that helper directly to ``[batch, horizon, 2]`` silently
    repeats the *horizon* axis rather than forming five time-major coordinates
    inside each packet.  Keep this conversion local and explicit.
    """

    if params.ndim != 3 or params.shape[-1] != 2 or action_block < 1:
        raise ValueError("batched PushT compiler parameters must have shape [batch, horizon, 2]")
    mean = torch.as_tensor(p0.PUBLIC_ACTION_MEAN, dtype=params.dtype, device=params.device)
    std = torch.as_tensor(p0.PUBLIC_ACTION_STD, dtype=params.dtype, device=params.device)
    live = torch.tanh(params)
    normalized = (live / 100.0 - mean) / std
    return normalized.repeat(1, 1, int(action_block))


def _advance(model: torch.nn.Module, emb: torch.Tensor, actions: torch.Tensor, prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a frozen latent context through a legal live-action prefix."""
    if prefix.ndim != 3 or emb.shape[0] != 1 or actions.shape[:2] != emb.shape[:2]:
        raise ValueError("invalid P1 prefix/context layout")
    h = int(emb.shape[1])
    token = _batched_live_action_tokens(prefix, action_block=5)
    samples = int(token.shape[0])
    latent = emb.expand(samples, -1, -1)
    action_context = actions.expand(samples, -1, -1)
    # ``LeWM.rollout`` first predicts the successor implied by its complete
    # observed action history, *then* appends the first proposed future
    # action and predicts again.  The former implementation silently dropped
    # the oldest observed action and inserted the proposal before this
    # bootstrap prediction.  That has compatible tensor shapes but shifts
    # action/transition alignment by one time step, so a model-side
    # "non-greedy" witness need not refer to the executed physical prefix.
    bootstrap = model.predict(latent[:, -h:], model.action_encoder(action_context[:, -h:]))[:, -1]
    latent = torch.cat((latent, bootstrap[:, None]), dim=1)
    for step in range(token.shape[1]):
        action_context = torch.cat((action_context, token[:, step:step + 1]), dim=1)
        next_latent = model.predict(latent[:, -h:], model.action_encoder(action_context[:, -h:]))[:, -1]
        latent = torch.cat((latent, next_latent[:, None]), dim=1)
    return latent[:, -h:], action_context[:, -h:]


def _suffix_terminal(model: torch.nn.Module, emb: torch.Tensor, actions: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    h = int(emb.shape[1])
    future = _batched_live_action_tokens(params, action_block=5)
    if future.shape[0] != emb.shape[0]:
        raise ValueError("P1 suffix samples and context samples differ")
    latent, current_actions, terminal = emb, actions, None
    for step in range(future.shape[1]):
        next_actions = torch.cat((current_actions, future[:, step:step + 1]), dim=1)
        terminal = model.predict(latent[:, -h:], model.action_encoder(next_actions[:, -h:]))[:, -1]
        latent, current_actions = torch.cat((latent, terminal[:, None]), dim=1), next_actions
    if terminal is None:
        raise RuntimeError("P1 suffix requires a positive horizon")
    return terminal


def _frozen_projection(*, latent_dim: int, sketch: int, device: torch.device, seed: int) -> torch.Tensor:
    """Create one root-level task sketch shared by every candidate prefix."""

    generator = torch.Generator(device=device).manual_seed(int(seed))
    projection = torch.randn((min(int(sketch), int(latent_dim)), int(latent_dim)), generator=generator, device=device)
    return projection / projection.norm(dim=1, keepdim=True).clamp_min(1e-8)


def _reachability(model: torch.nn.Module, goal: torch.Tensor, emb: torch.Tensor, actions: torch.Tensor, *, horizon: int, damping: float, projection: torch.Tensor) -> dict[str, Any]:
    flat = torch.zeros((horizon, 2), device=emb.device, requires_grad=True)
    def terminal(value: torch.Tensor) -> torch.Tensor:
        return _suffix_terminal(model, emb, actions, value.reshape(1, horizon, 2))[0]
    predicted = terminal(flat)
    if projection.ndim != 2 or projection.shape[1] != predicted.numel() or not torch.isfinite(projection).all():
        raise ValueError("P1 reachability projection is incompatible with the predicted latent")
    def projected(value: torch.Tensor) -> torch.Tensor:
        return projection @ terminal(value.reshape(horizon, 2))
    jac = torch.autograd.functional.jacobian(projected, flat.reshape(-1), vectorize=True)
    if not torch.isfinite(jac).all() or float(jac.norm()) <= 1e-10:
        raise RuntimeError("P1 post-prefix action Jacobian is non-finite or zero")
    residual = goal - predicted
    left, singular, right_h = torch.linalg.svd(jac, full_matrices=False)
    # The sketch only supplies the controllable basis; project the full goal
    # residual into the same frozen random task basis before scoring.
    projected_residual = projection @ residual
    coord = left.T @ projected_residual
    orthogonal = projected_residual - left @ coord
    value = orthogonal.square().sum() + (float(damping) / (singular.square() + float(damping)) * coord.square()).sum()
    update = right_h.T @ (singular / (singular.square() + float(damping)) * coord)
    with torch.no_grad():
        bounded = _suffix_terminal(model, emb, actions, update.reshape(1, horizon, 2))[0]
    return {"value": float(value.detach()), "predicted_goal_cost": float((predicted - goal).square().mean().detach()), "bounded_suffix_goal_cost": float((bounded - goal).square().mean().detach()), "jacobian_frobenius_norm": float(jac.norm().detach()), "singular_values": [float(x) for x in singular.detach().cpu()], "dls_params": update.detach().reshape(1, horizon, 2)}


def compile_reconfiguration(model: torch.nn.Module, frames: list[np.ndarray], history: np.ndarray, goal_rgb: np.ndarray, *, transform: Any, device: torch.device, proposals: int, prefix_horizon: int, suffix_horizon: int, damping: float, sketch: int, seed: int) -> dict[str, Any]:
    """Choose only from a frozen uniform action bank, retaining zero control."""
    emb, actions = _encode_context(model, frames, history, transform, device)
    goal = p0._goal_latent(model, goal_rgb, transform=transform, device=device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    physical = 2.0 * torch.rand((proposals, prefix_horizon, 2), generator=gen, device=device) - 1.0
    bank = torch.atanh(physical.clamp(-1 + torch.finfo(torch.float32).eps, 1 - torch.finfo(torch.float32).eps))
    bank[0].zero_()
    projection = _frozen_projection(latent_dim=int(goal.numel()), sketch=sketch, device=device, seed=seed + 10_000)
    rows: list[dict[str, Any]] = []
    for index in range(proposals):
        after_emb, after_actions = _advance(model, emb, actions, bank[index:index + 1])
        immediate = float((after_emb[0, -1] - goal).square().mean().detach())
        reach = _reachability(model, goal, after_emb, after_actions, horizon=suffix_horizon, damping=damping, projection=projection)
        rows.append({"proposal_index": index, "immediate_goal_cost": immediate, **{k: v for k, v in reach.items() if k != "dls_params"}})
    zero = rows[0]
    eligible = [row for row in rows[1:] if row["immediate_goal_cost"] > zero["immediate_goal_cost"] and row["value"] < zero["value"]]
    selected = min(eligible, key=lambda row: (row["value"], row["proposal_index"])) if eligible else None
    return {"non_greedy_witness_found": selected is not None, "selected_proposal_index": None if selected is None else int(selected["proposal_index"]), "selected_prefix_live_action": None if selected is None else torch.tanh(bank[selected["proposal_index"]]).detach().cpu().tolist(), "model_margin_vs_zero": None if selected is None else float(selected["immediate_goal_cost"] - zero["immediate_goal_cost"]), "reachability_value_improvement": None if selected is None else float(zero["value"] - selected["value"]), "proposal_rows": rows, "proposal_bank_sha256": hashlib.sha256(bank.detach().cpu().numpy().tobytes()).hexdigest(), "reachability_projection_sha256": hashlib.sha256(projection.detach().cpu().numpy().tobytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p0-evidence", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--cohort", choices=("calibration", "outcome"), default="calibration")
    parser.add_argument("--roots", type=int, default=4); parser.add_argument("--seed", type=int, default=202610251)
    parser.add_argument("--proposals", type=int, default=16); parser.add_argument("--prefix-horizon", type=int, default=1); parser.add_argument("--suffix-horizon", type=int, default=1)
    parser.add_argument("--damping", type=float, default=1e-2); parser.add_argument("--sketch-dims", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite immutable P1 output: {args.output}")
    if min(args.roots, args.proposals, args.prefix_horizon, args.suffix_horizon, args.sketch_dims) < 1 or args.proposals < 2 or args.damping <= 0: raise ValueError("invalid P1 dimensions or damping")
    p0_report = _load_p0(args.p0_evidence)
    runtime = require_cuda_allocation("cuda"); device = torch.device("cuda")
    model = load_compatible_pretrained("pusht/lewm", device=device).eval(); model.requires_grad_(False)
    if validate_action_protocol(model, action_dim=2, action_block=5) != 5: raise RuntimeError("PushT checkpoint action block drift")
    before = p0._parameter_hash(model); transform = image_transform(); env = gym.make("swm/PushT-v1", render_mode="rgb_array"); started = time.time()
    try:
        roots = collect_nonterminal_boundary_contact_roots(env, count=args.roots, seed=args.seed, oversample_factor=4)
        prior = {row["root_sha256"] for row in p0_report["calibration_roots"]}
        if prior & {root_fingerprint(root) for root in roots}: raise RuntimeError("P1 calibration roots overlap P0 roots")
        rows = []
        for index, root in enumerate(roots):
            frames, history, goal = p0._collect_causal_rgb_context(env, root, history_len=3)
            compilation = compile_reconfiguration(model, frames, history, goal, transform=transform, device=device, proposals=args.proposals, prefix_horizon=args.prefix_horizon, suffix_horizon=args.suffix_horizon, damping=args.damping, sketch=args.sketch_dims, seed=args.seed + index)
            rows.append({"root_id": index, "root_sha256": root_fingerprint(root), "history_rgb_sha256": hashlib.sha256(np.stack(frames).tobytes()).hexdigest(), "goal_rgb_sha256": hashlib.sha256(goal.tobytes()).hexdigest(), "compilation": compilation})
            print(json.dumps({"root_id": index, "selected": compilation["selected_proposal_index"], "witness": compilation["non_greedy_witness_found"]}))
    finally: env.close()
    gates = {"cuda_only": True, "frozen_parameter_hash_unchanged": before == p0._parameter_hash(model), "fresh_roots_disjoint_from_p0": True, "rgb_goal_action_only": True, "no_oracle_successor_or_state_in_score": True, "at_least_one_model_non_greedy_reconfiguration_witness": any(row["compilation"]["non_greedy_witness_found"] for row in rows), "physical_candidates_not_executed": True}
    report = {"schema": SCHEMA, "status": "pass" if all(gates.values()) else "fail", "scope": "CUDA-only model-side PushT P1 admission; no physical candidate is executed and no efficacy/outcome conclusion is licensed.", "cuda_host": {**runtime, "python": platform.python_version()}, "upstream_p0": {"path": str(args.p0_evidence.resolve()), "sha256": _sha256(args.p0_evidence)}, "protocol": {"cohort": args.cohort, "planner_input": "causal RGB/action history and rendered goal RGB", "root_count": args.roots, "root_seed": args.seed, "root_oversample_factor": 4, "prefix_proposals": args.proposals, "prefix_horizon": args.prefix_horizon, "suffix_horizon": args.suffix_horizon, "damping": args.damping, "sketch_dims": args.sketch_dims, "proposal_distribution": "frozen physical uniform live-action box", "selection": "strictly non-greedy immediate frozen goal cost plus lower recompiled local reachability value", "physical_execution": "deferred to a separately root-disjoint calibration or outcome receipt"}, "gates": gates, "rows": rows, "elapsed_seconds": time.time() - started}
    args.output.mkdir(parents=True); (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "gates": gates, "output": str(args.output)}, indent=2))


if __name__ == "__main__": main()
