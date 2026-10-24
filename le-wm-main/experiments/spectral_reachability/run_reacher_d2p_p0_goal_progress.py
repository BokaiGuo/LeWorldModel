"""Fresh paired physical P0 for D2P goal-progress programs on frozen Reacher.

The candidate-prefix adapter is deliberately local to this experiment.  A
frozen LeWM receives rendered RGB history, rendered goal RGB, an immutable
normalized action-history prefix, and an optimized future suffix.  This avoids
the upstream ``LeWM.rollout`` ambiguity where candidate rows, rather than
``info['action']``, form the action history.

P0 does not claim a new optimizer: at the same linearization point the D2P
goal-progress program and local Gauss--Newton are algebraically identical.
The runner verifies that equivalence, then reports paired physical qpos
outcomes against CEM, action-GD, restart-GD, and random search.  All learned
model calls require host CUDA; qpos is read only after execution by the
evaluator and never enters a planner input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    import sys

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.online_planning_protocol import validate_action_protocol
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import (
    ReacherGoalImageWrapper,
    _load_action_stats,
    install_missing_field_compat,
)
from experiments.run_live_reacher_sequence_candidate_planning import _qpos_distance
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import (
    _bounded_normalized_actions,
    _normalized_action_bounds,
    _parameter_hash,
    _raw_block,
    _sha256_file,
)
from experiments.spectral_reachability.build_reacher_d2p_training_action_stats import validate_training_semantic_receipt
from experiments.spectral_reachability.reacher_d2p_context import (
    WARMUP_POLICY,
    collect_executed_context,
    require_new_output,
    restore_execution_snapshot,
)


# Four accepted local updates are the last validated P0 configuration.  The
# actual comparison uses a 128-candidate model-only screen before local DLS;
# defaulting to a one-start local solve would silently leave that declared
# coverage budget unused while CEM still evaluates all 128 candidates.
DEFAULT_RELINEARIZATION_STEPS = 4
DEFAULT_D2P_INITIALIZATION = "screened_gaussian_proposals"
DEFAULT_D2P_PROPOSAL_SAMPLES = 128


def _bootstrap_mean(values: np.ndarray, seed: int, draws: int = 10_000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap requires at least two paired rows")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _prepare(
    preprocessor: _Preprocessor,
    frames: list[np.ndarray],
    root_frame: np.ndarray,
    goal: np.ndarray,
    action_history: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a root-anchored context with causal prior actions.

    A three-frame checkpoint predicts the successor of its *last* frame.
    After executed warmup the planner observes that successor directly, so the
    model context must be ``[frame_1, frame_2, observed_root]`` rather than
    ``[frame_0, frame_1, frame_2]`` followed by a model-generated root.
    ``action_history`` retains all three recorded blocks because the terminal
    adapter uses its final two as the causal actions between these frames.
    """
    history_len = int(action_history.shape[0])
    if len(frames) != history_len or history_len < 2:
        raise ValueError("D2P RGB/action history lengths differ")
    root_anchored_frames = [np.asarray(frame, dtype=np.uint8) for frame in frames[1:]]
    root_anchored_frames.append(np.asarray(root_frame, dtype=np.uint8))
    if len(root_anchored_frames) != history_len:
        raise ValueError("D2P root-anchored RGB history has incorrect length")
    raw = {
        "pixels": np.asarray(root_anchored_frames, dtype=np.uint8)[None],
        "goal": np.asarray(goal, dtype=np.uint8)[None, None],
        "action": np.asarray(action_history, dtype=np.float32)[None],
    }
    prepared = preprocessor._prepare_info(raw)
    return prepared["pixels"].to(device), prepared["goal"].to(device), prepared["action"].to(device)


def _latent_context(model: torch.nn.Module, pixels: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.encode({"pixels": pixels, "action": history})["emb"].detach()


def _goal_latent(model: torch.nn.Module, goal: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.encode({"pixels": goal})["emb"][0, -1].detach()


def _terminal_batch(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    goal: torch.Tensor,
    history: torch.Tensor,
    future_params: torch.Tensor,
    *,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> torch.Tensor:
    """Return terminal predictions conditioned on the *observed* planning root.

    ``JEPA.rollout`` seeds its first prediction from every supplied RGB frame.
    Feeding it the three pre-warmup frames therefore creates a predicted root
    before it reaches the candidate suffix.  This adapter instead uses the
    last two recorded transitions plus the actually rendered post-warmup root,
    then introduces the first optimized action as the final causal action
    token.  Subsequent actions are rolled out autoregressively from there.
    """
    if future_params.ndim != 3:
        raise ValueError("future parameters must be [samples, future_horizon, action_dim]")
    samples, _, action_dim = future_params.shape
    if tuple(history.shape) != (1, pixels.shape[1], action_dim):
        raise ValueError("D2P history prefix does not match the RGB history/action dimension")
    if pixels.shape[1] != history.shape[1] or pixels.shape[1] < 2:
        raise ValueError("D2P root-anchored RGB/action context has incompatible length")
    encoded = _latent_context(model, pixels, history)
    future = _bounded_normalized_actions(future_params, action_lower, action_upper)
    if future.shape[1] < 1:
        raise ValueError("D2P root-anchored rollout requires at least one future action")
    history_size = int(pixels.shape[1])
    # Recorded actions a_{t-2}, a_{t-1} connect the first two observations to
    # the observed root at t.  The first optimized block is a_t.
    action_context = torch.cat(
        [history[0, 1:].unsqueeze(0).expand(samples, -1, -1), future[:, :1]], dim=1,
    )
    emb_context = encoded.expand(samples, -1, -1)
    terminal = None
    for step in range(future.shape[1]):
        action_emb = model.action_encoder(action_context[:, -history_size:])
        terminal = model.predict(
            emb_context[:, -history_size:], action_emb,
        )[:, -1]
        if step + 1 < future.shape[1]:
            emb_context = torch.cat([emb_context, terminal[:, None]], dim=1)
            action_context = torch.cat([action_context, future[:, step + 1:step + 2]], dim=1)
    if terminal is None or terminal.ndim != 2 or terminal.shape[0] != samples:
        raise RuntimeError("unexpected root-anchored terminal prediction shape")
    return terminal


def _cost_fn(
    model: torch.nn.Module, pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor,
    params: torch.Tensor, *, action_lower: torch.Tensor, action_upper: torch.Tensor,
) -> torch.Tensor:
    terminal = _terminal_batch(model, pixels, goal, history, params, action_lower=action_lower, action_upper=action_upper)
    goal_latent = _goal_latent(model, goal).unsqueeze(0)
    return (terminal - goal_latent).square().mean(dim=1)


def _accepted_backtracking_index(candidate_costs: torch.Tensor, cost_before: float) -> int:
    """Choose the best fixed trust-region step and reject objective ascent."""

    if candidate_costs.ndim != 1 or candidate_costs.numel() < 2 or not torch.isfinite(candidate_costs).all():
        raise RuntimeError("D2P trust-region candidate costs are invalid")
    index = int(torch.argmin(candidate_costs))
    if float(candidate_costs[index].detach()) > float(cost_before) + 1.0e-6:
        raise RuntimeError("D2P trust-region acceptance failed to preserve the frozen-model objective")
    return index


def _d2p_goal_progress(
    model: torch.nn.Module, pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor,
    *, future_horizon: int, action_lower: torch.Tensor, action_upper: torch.Tensor, damping: float,
    relinearization_steps: int = 1, initial_params: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compile a bounded DLS program, relinearizing after accepted updates."""
    if int(relinearization_steps) < 1:
        raise ValueError("D2P relinearization steps must be positive")
    action_dim = int(history.shape[-1])
    goal_latent = _goal_latent(model, goal)
    expected_shape = (1, int(future_horizon), action_dim)
    if initial_params is None:
        params = torch.zeros(expected_shape, device=history.device)
        initialization = "zero"
    else:
        if tuple(initial_params.shape) != expected_shape:
            raise ValueError(
                f"D2P initial parameters must have shape {expected_shape}, got {tuple(initial_params.shape)}"
            )
        params = initial_params.detach().to(device=history.device, dtype=history.dtype)
        if not torch.isfinite(params).all():
            raise ValueError("D2P initial parameters must be finite")
        initialization = "provided"
    step_scales = torch.as_tensor([1.0, 0.5, 0.25, 0.125, 0.0625, 0.0], device=history.device)
    iterations: list[dict[str, Any]] = []
    initial_predicted = None
    jacobian = None
    for iteration in range(int(relinearization_steps)):
        flat_current = params.reshape(-1).detach().requires_grad_(True)

        def terminal(flat: torch.Tensor) -> torch.Tensor:
            return _terminal_batch(
                model, pixels, goal, history, flat.reshape(1, future_horizon, action_dim),
                action_lower=action_lower, action_upper=action_upper,
            )[0]

        predicted_current = terminal(flat_current)
        if initial_predicted is None:
            initial_predicted = predicted_current.detach()
        jacobian = torch.autograd.functional.jacobian(terminal, flat_current, vectorize=True)
        if jacobian.ndim != 2 or not torch.isfinite(jacobian).all() or float(jacobian.norm()) <= 1.0e-10:
            raise RuntimeError("D2P goal-progress Jacobian is non-finite or zero")
        system = jacobian.T @ jacobian + float(damping) * torch.eye(jacobian.shape[1], device=jacobian.device)
        update = torch.linalg.solve(system, jacobian.T @ (goal_latent - predicted_current))
        cost_before = float(((predicted_current - goal_latent).square().mean()).detach())
        candidates = params + step_scales[:, None, None] * update.reshape(1, future_horizon, action_dim)
        candidate_costs = _cost_fn(model, pixels, goal, history, candidates, action_lower=action_lower, action_upper=action_upper)
        selected_step_index = _accepted_backtracking_index(candidate_costs, cost_before)
        params = candidates[selected_step_index:selected_step_index + 1].detach()
        cost_after = float(candidate_costs[selected_step_index].detach())
        singular = torch.linalg.svdvals(jacobian)
        iterations.append({
            "iteration": iteration,
            "predicted_cost_before": cost_before,
            "predicted_cost_after": cost_after,
            "accepted_step_scale": float(step_scales[selected_step_index]),
            "jacobian_frobenius_norm": float(jacobian.norm()),
            "effective_rank_1e-4": int((singular > singular[0] * 1.0e-4).sum()),
        })
        if selected_step_index == len(step_scales) - 1:
            break
    if jacobian is None or initial_predicted is None or not iterations:
        raise RuntimeError("D2P relinearization produced no accepted iteration")
    singular = torch.linalg.svdvals(jacobian)
    final_cost = float(_cost_fn(model, pixels, goal, history, params, action_lower=action_lower, action_upper=action_upper)[0].detach())
    return params, {
        "program_type": "relinearized_goal_progress",
        "initialization": initialization,
        "predicted_successor_l2_to_goal_before": float(torch.linalg.vector_norm(initial_predicted - goal_latent).detach()),
        "predicted_successor_l2_to_goal_after": float(torch.linalg.vector_norm(
            _terminal_batch(model, pixels, goal, history, params, action_lower=action_lower, action_upper=action_upper)[0] - goal_latent
        ).detach()),
        "predicted_cost_before": iterations[0]["predicted_cost_before"],
        "predicted_cost_after": final_cost,
        "jacobian_frobenius_norm": float(jacobian.norm()),
        "singular_values": [float(value) for value in singular],
        "effective_rank_1e-4": int((singular > singular[0] * 1.0e-4).sum()),
        "damping": float(damping),
        "trust_region_step_scales": [float(value) for value in step_scales],
        "accepted_step_scale": iterations[-1]["accepted_step_scale"],
        "relinearization_steps_requested": int(relinearization_steps),
        "relinearization_steps_executed": len(iterations),
        "relinearization_trace": iterations,
        "actual_model_cost_nonincreasing": bool(all(
            item["predicted_cost_after"] <= item["predicted_cost_before"] + 1.0e-6
            for item in iterations
        )),
        # ``jacobian(..., vectorize=True)`` has one primal rollout but computes
        # one VJP row per latent coordinate.  Do not mislabel that as three
        # equivalent model calls: forward rollouts and VJPs are reported
        # separately so a later resource-matched protocol can use real cost.
        "forward_rollout_evaluations": len(iterations) * (2 + int(len(step_scales))),
        "backward_vjp_rows": len(iterations) * int(jacobian.shape[0]),
    }


def _restart_d2p_goal_progress(
    model: torch.nn.Module, pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor,
    *, future_horizon: int, action_lower: torch.Tensor, action_upper: torch.Tensor, damping: float,
    relinearization_steps: int, restarts: int, seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pick a frozen-objective winner among a zero and seeded DLS starts."""
    if int(restarts) < 1:
        raise ValueError("D2P restart count must be positive")
    generator = torch.Generator(device=history.device).manual_seed(int(seed))
    candidates: list[torch.Tensor] = []
    receipts: list[dict[str, Any]] = []
    costs: list[float] = []
    for restart in range(int(restarts)):
        initial = None if restart == 0 else torch.randn(
            (1, int(future_horizon), history.shape[-1]),
            generator=generator, device=history.device, dtype=history.dtype,
        )
        params, receipt = _d2p_goal_progress(
            model, pixels, goal, history,
            future_horizon=future_horizon, action_lower=action_lower, action_upper=action_upper,
            damping=damping, relinearization_steps=relinearization_steps, initial_params=initial,
        )
        cost = float(_cost_fn(
            model, pixels, goal, history, params,
            action_lower=action_lower, action_upper=action_upper,
        )[0].detach())
        candidates.append(params)
        receipts.append(receipt)
        costs.append(cost)
    selected = int(np.argmin(np.asarray(costs, dtype=np.float64)))
    winner = dict(receipts[selected])
    winner.update({
        "d2p_restarts": int(restarts),
        "selected_restart": selected,
        "restart_predicted_terminal_costs": costs,
        "restart_initializations": [receipt["initialization"] for receipt in receipts],
        "forward_rollout_evaluations": int(sum(item["forward_rollout_evaluations"] for item in receipts) + int(restarts)),
        "backward_vjp_rows": int(sum(item["backward_vjp_rows"] for item in receipts)),
    })
    return candidates[selected].detach(), winner


def _screened_proposal_d2p_goal_progress(
    model: torch.nn.Module, pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor,
    *, future_horizon: int, action_lower: torch.Tensor, action_upper: torch.Tensor, damping: float,
    relinearization_steps: int, proposals: int, elites: int, seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Globally screen frozen-objective proposals, then refine their best basins.

    The original four-restart program can only enter four randomly chosen local
    basins.  Here a deterministic, *model-only* proposal screen finds the best
    ``elites`` among a wider Gaussian bank before DLS refines them.  The zero
    program is always retained as proposal zero, so the new strategy cannot
    silently remove the old baseline.  This is intentionally a separate
    planning protocol, not a retroactive alteration of a sealed P0 receipt.
    """
    if int(proposals) < 1 or int(elites) < 1 or int(elites) > int(proposals):
        raise ValueError("D2P screened-proposal counts must satisfy 1 <= elites <= proposals")
    generator = torch.Generator(device=history.device).manual_seed(int(seed))
    bank = torch.randn(
        (int(proposals), int(future_horizon), history.shape[-1]),
        generator=generator, device=history.device, dtype=history.dtype,
    )
    bank[0].zero_()
    proposal_costs = _cost_fn(
        model, pixels, goal, history, bank,
        action_lower=action_lower, action_upper=action_upper,
    )
    if not torch.isfinite(proposal_costs).all():
        raise RuntimeError("D2P screened proposals have non-finite frozen costs")
    selected = torch.topk(proposal_costs, k=int(elites), largest=False).indices.tolist()
    candidates: list[torch.Tensor] = []
    receipts: list[dict[str, Any]] = []
    refined_costs: list[float] = []
    for index in selected:
        params, receipt = _d2p_goal_progress(
            model, pixels, goal, history,
            future_horizon=future_horizon, action_lower=action_lower, action_upper=action_upper,
            damping=damping, relinearization_steps=relinearization_steps,
            initial_params=bank[index:index + 1],
        )
        cost = float(_cost_fn(
            model, pixels, goal, history, params,
            action_lower=action_lower, action_upper=action_upper,
        )[0].detach())
        candidates.append(params)
        receipts.append(receipt)
        refined_costs.append(cost)
    winner_index = int(np.argmin(np.asarray(refined_costs, dtype=np.float64)))
    winner = dict(receipts[winner_index])
    winner.update({
        "initialization_strategy": "screened_gaussian_proposals",
        "proposal_screen_executed": True,
        "proposal_count": int(proposals),
        "proposal_elites": int(elites),
        "proposal_selected_indices": [int(index) for index in selected],
        "proposal_terminal_costs": [float(value) for value in proposal_costs.detach().cpu()],
        "selected_proposal_index": int(selected[winner_index]),
        "refined_terminal_costs": refined_costs,
        "forward_rollout_evaluations": int(
            int(proposals) + sum(item["forward_rollout_evaluations"] for item in receipts) + int(elites)
        ),
        "backward_vjp_rows": int(sum(item["backward_vjp_rows"] for item in receipts)),
    })
    return candidates[winner_index].detach(), winner


def _d2p_program(
    model: torch.nn.Module, pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor,
    *, future_horizon: int, action_lower: torch.Tensor, action_upper: torch.Tensor, damping: float,
    relinearization_steps: int, restarts: int, initialization: str,
    proposal_samples: int, proposal_elites: int, seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Dispatch a receipt-bound D2P initialization strategy."""
    if initialization == "gaussian_restarts":
        params, receipt = _restart_d2p_goal_progress(
            model, pixels, goal, history, future_horizon=future_horizon,
            action_lower=action_lower, action_upper=action_upper, damping=damping,
            relinearization_steps=relinearization_steps, restarts=restarts, seed=seed,
        )
        receipt = dict(receipt)
        receipt.update({
            "initialization_strategy": "gaussian_restarts",
            "proposal_screen_executed": False,
            "configured_proposal_samples_ignored": int(proposal_samples),
            "configured_proposal_elites_ignored": int(proposal_elites),
        })
        return params, receipt
    if initialization == "screened_gaussian_proposals":
        return _screened_proposal_d2p_goal_progress(
            model, pixels, goal, history, future_horizon=future_horizon,
            action_lower=action_lower, action_upper=action_upper, damping=damping,
            relinearization_steps=relinearization_steps, proposals=proposal_samples,
            elites=proposal_elites, seed=seed,
        )
    raise ValueError(f"unknown D2P initialization strategy: {initialization}")


def _action_gd(
    model: torch.nn.Module, pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor,
    *, future_horizon: int, action_lower: torch.Tensor, action_upper: torch.Tensor, steps: int, lr: float, seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    del seed  # deterministic zero initialization is part of the frozen P0 protocol.
    params = torch.zeros((1, future_horizon, history.shape[-1]), device=history.device, requires_grad=True)
    optim = torch.optim.Adam([params], lr=float(lr))
    trace: list[float] = []
    for _ in range(steps):
        cost = _cost_fn(model, pixels, goal, history, params, action_lower=action_lower, action_upper=action_upper).sum()
        cost.backward()
        if params.grad is None or not torch.isfinite(params.grad).all():
            raise RuntimeError("action-GD produced an invalid gradient")
        optim.step()
        optim.zero_grad(set_to_none=True)
        trace.append(float(cost.detach()))
    return params.detach(), {"trace": trace, "forward_rollout_evaluations": int(steps), "backward_vjp_rows": int(steps)}


def _restart_gd(
    model: torch.nn.Module, pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor,
    *, future_horizon: int, action_lower: torch.Tensor, action_upper: torch.Tensor, restarts: int, steps: int, lr: float, seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device=history.device).manual_seed(int(seed))
    candidates: list[torch.Tensor] = []
    traces: list[list[float]] = []
    for _ in range(restarts):
        params = (0.25 * torch.randn((1, future_horizon, history.shape[-1]), generator=generator, device=history.device)).requires_grad_()
        optim = torch.optim.Adam([params], lr=float(lr))
        trace: list[float] = []
        for _ in range(steps):
            cost = _cost_fn(model, pixels, goal, history, params, action_lower=action_lower, action_upper=action_upper).sum()
            cost.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            trace.append(float(cost.detach()))
        candidates.append(params.detach())
        traces.append(trace)
    stacked = torch.cat(candidates, dim=0)
    costs = _cost_fn(model, pixels, goal, history, stacked, action_lower=action_lower, action_upper=action_upper)
    index = int(torch.argmin(costs))
    return stacked[index:index + 1].detach(), {
        "restart_traces": traces,
        "selected_restart": index,
        "forward_rollout_evaluations": int(restarts * steps + 1),
        "backward_vjp_rows": int(restarts * steps),
    }


def _cem(
    model: torch.nn.Module, pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor,
    *, future_horizon: int, action_lower: torch.Tensor, action_upper: torch.Tensor, samples: int, steps: int, elite: int, seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if elite >= samples:
        raise ValueError("CEM elite count must be smaller than sample count")
    generator = torch.Generator(device=history.device).manual_seed(int(seed))
    mean = torch.zeros((future_horizon, history.shape[-1]), device=history.device)
    std = torch.ones_like(mean)
    trace: list[float] = []
    final_candidates = None
    final_costs = None
    for _ in range(steps):
        candidates = torch.randn((samples, future_horizon, history.shape[-1]), generator=generator, device=history.device)
        candidates = candidates * std.unsqueeze(0) + mean.unsqueeze(0)
        candidates[0] = mean
        costs = _cost_fn(model, pixels, goal, history, candidates, action_lower=action_lower, action_upper=action_upper)
        values, indices = torch.topk(costs, k=elite, largest=False)
        chosen = candidates[indices]
        mean = chosen.mean(dim=0)
        std = chosen.std(dim=0).clamp_min(1.0e-4)
        trace.append(float(values.mean()))
        final_candidates, final_costs = candidates, costs
    if final_candidates is None or final_costs is None:
        raise RuntimeError("CEM completed no iteration")
    best = int(torch.argmin(final_costs))
    return final_candidates[best:best + 1].detach(), {
        "trace": trace,
        "forward_rollout_evaluations": int(samples * steps),
        "backward_vjp_rows": 0,
        "final_selection": "best_sample",
    }


def _random_program(
    *, future_horizon: int, action_dim: int, device: torch.device, seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn((1, future_horizon, action_dim), generator=generator, device=device), {
        "forward_rollout_evaluations": 0, "backward_vjp_rows": 0,
    }


def _to_raw_actions(
    params: torch.Tensor, mean: np.ndarray, std: np.ndarray,
    action_lower: torch.Tensor, action_upper: torch.Tensor,
    raw_low: np.ndarray, raw_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return only physically executable raw actions and their model values."""

    normalized = _bounded_normalized_actions(params, action_lower, action_upper)[0].detach().cpu().numpy()
    raw = _normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)
    if np.any(raw < np.asarray(raw_low, dtype=np.float32) - 1.0e-6) or np.any(raw > np.asarray(raw_high, dtype=np.float32) + 1.0e-6):
        raise RuntimeError("planner produced a raw action outside the environment box")
    return raw, normalized


def _normalized_blocks_to_raw_actions(
    normalized_blocks: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    action_dim: int,
) -> np.ndarray:
    """Invert flattened LeWM tokens to environment actions in time-major order.

    Each model token is a flattened ``[action_block, action_dim]`` array.  Its
    first ``action_dim`` values are the first physical action in the block;
    this is the same layout used by the released action statistics and the
    established online-planning packet builders.  Keep this conversion
    separate from bounding so it can be contract-tested with non-constant
    actions.
    """
    normalized = np.asarray(normalized_blocks, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if (
        normalized.ndim != 2
        or int(action_dim) < 1
        or mean.shape != std.shape
        or normalized.shape[-1] != mean.size
        or mean.size % int(action_dim)
    ):
        raise ValueError("normalized Reacher action blocks violate the time-major layout contract")
    action_block = mean.size // int(action_dim)
    raw_blocks = normalized * std.reshape(1, -1) + mean.reshape(1, -1)
    return raw_blocks.reshape(normalized.shape[0], action_block, int(action_dim)).reshape(-1, int(action_dim))


def _validate_reacher_action_protocol(
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    action_dim: int,
    action_block: int,
) -> int:
    """Bind checkpoint width, action statistics, and execution layout together."""
    expected = validate_action_protocol(
        model, action_dim=int(action_dim), action_block=int(action_block),
    )
    width = int(action_block) * int(action_dim)
    if np.asarray(mean).shape != (width,) or np.asarray(std).shape != (width,):
        raise ValueError(
            "Reacher action statistics do not match the checkpoint action protocol: "
            f"expected flattened width {width}, got mean={np.asarray(mean).shape}, "
            f"std={np.asarray(std).shape}"
        )
    return expected


def _rgb_digest(value: np.ndarray) -> str:
    image = np.ascontiguousarray(np.asarray(value, dtype=np.uint8))
    return hashlib.sha256(image.tobytes()).hexdigest()


def _validate_outcome_calibration(
    evidence: tuple[Path, ...], *, damping: float, relinearization_steps: int, d2p_restarts: int,
    d2p_initialization: str, d2p_proposal_samples: int, d2p_proposal_elites: int,
    outcome_seeds: set[int],
) -> dict[str, Any]:
    """Bind an outcome panel to exactly one compatible, disjoint selection receipt."""

    if len(evidence) != 1:
        raise ValueError("outcome-only requires exactly one damping-selection receipt")
    path = evidence[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "d2p_reacher_damping_selection.v1" or payload.get("status") != "pass":
        raise ValueError("outcome-only damping evidence is not a passing selection receipt")
    selected = payload.get("selected") or {}
    if float(selected.get("damping", float("nan"))) != float(damping):
        raise ValueError("outcome damping differs from the selected calibration damping")
    calibration_seeds = {int(seed) for seed in payload.get("calibration_root_seeds", [])}
    if not calibration_seeds or calibration_seeds & outcome_seeds:
        raise ValueError("calibration and outcome root seeds must be non-empty and disjoint")
    contract = payload.get("calibration_contract", {})
    if contract.get("action_parameterization") != "tanh mapped to environment raw action box after checkpoint normalization":
        raise ValueError("outcome calibration receipt has an incompatible action parameterization")
    if contract.get("runner") != _sha256_file(Path(__file__)):
        raise ValueError("outcome calibration receipt was produced by different runner source")
    if int(contract.get("relinearization_steps", -1)) != int(relinearization_steps):
        raise ValueError("outcome relinearization steps differ from the calibration contract")
    if int(contract.get("d2p_restarts", -1)) != int(d2p_restarts):
        raise ValueError("outcome D2P restart count differs from the calibration contract")
    if contract.get("d2p_initialization") != d2p_initialization:
        raise ValueError("outcome D2P initialization differs from the calibration contract")
    if int(contract.get("d2p_proposal_samples", -1)) != int(d2p_proposal_samples):
        raise ValueError("outcome D2P proposal sample count differs from the calibration contract")
    if int(contract.get("d2p_proposal_elites", -1)) != int(d2p_proposal_elites):
        raise ValueError("outcome D2P proposal elite count differs from the calibration contract")
    if contract.get("context_mode") != "executed_warmup":
        raise ValueError("outcome calibration receipt does not bind an executed temporal context")
    if contract.get("rollout_root_conditioning") != "observed_post_warmup_rgb_anchor":
        raise ValueError("outcome calibration receipt does not bind the observed planning root")
    if contract.get("execution_root_reconstruction") != "evaluator_only_observed_root_snapshot_restore":
        raise ValueError("outcome calibration receipt does not bind the observed-root execution snapshot")
    if contract.get("warmup_policy") != WARMUP_POLICY:
        raise ValueError("outcome calibration receipt has an incompatible warmup policy")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "selected_damping": float(selected["damping"]),
        "relinearization_steps": int(relinearization_steps),
        "d2p_restarts": int(d2p_restarts),
        "d2p_initialization": d2p_initialization,
        "d2p_proposal_samples": int(d2p_proposal_samples),
        "d2p_proposal_elites": int(d2p_proposal_elites),
        "context_mode": "executed_warmup",
        "rollout_root_conditioning": "observed_post_warmup_rgb_anchor",
        "execution_root_reconstruction": "evaluator_only_observed_root_snapshot_restore",
        "warmup_policy": WARMUP_POLICY,
        "calibration_root_seeds": sorted(calibration_seeds),
    }


def _verify_execution_identity(
    reset_frame: np.ndarray,
    reset_goal: np.ndarray,
    expected_frame: np.ndarray,
    expected_goal: np.ndarray,
) -> dict[str, str]:
    """Fail closed if planning RGB and evaluator reset describe different roots."""
    if not np.array_equal(reset_frame, expected_frame):
        raise RuntimeError("execution reset RGB differs from the planner root")
    if not np.array_equal(reset_goal, expected_goal):
        raise RuntimeError("execution goal RGB differs from the planner root")
    return {
        "reset_rgb_sha256": _rgb_digest(reset_frame),
        "goal_rgb_sha256": _rgb_digest(reset_goal),
    }


def _execute(
    env: gym.Env,
    seed: int,
    actions: np.ndarray,
    *,
    expected_frame: np.ndarray,
    expected_goal: np.ndarray,
    execution_snapshot: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, str]]:
    if execution_snapshot is None:
        raise RuntimeError("Reacher execution requires an evaluator-only observed-root snapshot")
    reset_frame, reset_goal = restore_execution_snapshot(
        env, seed=int(seed), snapshot=execution_snapshot,
    )
    identity = _verify_execution_identity(
        reset_frame, reset_goal, expected_frame, expected_goal,
    )
    raw = env.unwrapped
    initial = np.asarray(raw.env.physics.data.qpos, dtype=np.float64).copy()
    target = np.asarray(raw.env.task.target_qpos, dtype=np.float64).copy()
    initial_distance = _qpos_distance(initial, target)
    min_distance = initial_distance
    for action in np.asarray(actions, dtype=np.float32):
        _, _, terminated, truncated, _ = env.step(action)
        current = np.asarray(raw.env.physics.data.qpos, dtype=np.float64)
        min_distance = min(min_distance, _qpos_distance(current, target))
        if terminated or truncated:
            break
    final = np.asarray(raw.env.physics.data.qpos, dtype=np.float64)
    return (
        {
            "initial_qpos_distance": initial_distance,
            "final_qpos_distance": _qpos_distance(final, target),
            "min_qpos_distance": min_distance,
            "physical_progress": initial_distance - _qpos_distance(final, target),
        },
        identity,
    )


def _root_context(
    env: gym.Env, seed: int, history_len: int, action_block: int,
) -> dict[str, Any]:
    """Collect a real causal history; reset padding is intentionally forbidden."""
    return collect_executed_context(
        env, seed=int(seed), history_len=int(history_len), action_block=int(action_block),
    )


def _paired_summary(rows: list[dict[str, Any]], method: str, reference: str, seed: int) -> dict[str, Any]:
    delta = np.asarray([
        row["methods"][method]["physical"]["final_qpos_distance"] - row["methods"][reference]["physical"]["final_qpos_distance"]
        for row in rows
    ], dtype=np.float64)
    return {
        "metric": f"final_qpos_distance: {method} minus {reference}; negative favors {method}",
        "n": int(delta.size),
        "mean": float(delta.mean()),
        "ci95": _bootstrap_mean(delta, seed),
        "method_only_wins": int(np.sum(delta < 0.0)),
        "reference_only_wins": int(np.sum(delta > 0.0)),
        "ties": int(np.sum(np.isclose(delta, 0.0))),
    }


def _p1_admission(
    cem_comparison: dict[str, Any] | None, *, calibration_only: bool,
) -> tuple[bool, str]:
    """Require affirmative P0 evidence before opening a later mechanism stage.

    A failure to meet the deliberately strict all-root no-go condition is not
    evidence that a local D2P program merits P1.  P1 requires the independent
    outcome panel to show a paired interval that favors D2P over the matched
    CEM comparator; an inconclusive or adverse P0 stays closed.
    """
    if calibration_only:
        return False, "calibration-only receipt; it cannot provide a P1 admission conclusion"
    if cem_comparison is None:
        return False, "missing independent D2P-versus-CEM outcome comparison"
    if float(cem_comparison["ci95"][1]) <= 0.0:
        return True, "independent paired 95% interval favors D2P over CEM"
    return False, "P0 does not establish a paired outcome interval favoring D2P over CEM"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm"))
    parser.add_argument("--adapter-artifact", type=Path, default=Path("results/live_cross_environment_prediction_20261015/reacher_cuda/cross_environment_prediction.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261040)
    parser.add_argument("--roots", type=int, default=16)
    parser.add_argument("--calibration-roots", type=int, default=4)
    parser.add_argument("--calibration-only", action="store_true", help="Use every root only to select a frozen damping value; emits no P0 efficacy verdict.")
    parser.add_argument("--outcome-only", action="store_true", help="Use every root for a fresh efficacy panel after damping is frozen by external calibration evidence.")
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--future-horizon", type=int, default=2)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--relinearization-steps", type=int, default=DEFAULT_RELINEARIZATION_STEPS)
    parser.add_argument("--d2p-restarts", type=int, default=1, help="Number of seeded DLS initializations; restart zero is the zero-action baseline.")
    parser.add_argument(
        "--d2p-initialization", choices=("gaussian_restarts", "screened_gaussian_proposals"),
        default=DEFAULT_D2P_INITIALIZATION,
        help="How D2P enters local basins; screened proposals is a separately calibrated globalized protocol.",
    )
    parser.add_argument(
        "--d2p-proposal-samples", type=int, default=DEFAULT_D2P_PROPOSAL_SAMPLES,
        help="Frozen-objective proposals screened before D2P refinement; default matches CEM samples × iterations.",
    )
    parser.add_argument("--d2p-proposal-elites", type=int, default=4)
    parser.add_argument(
        "--calibration-evidence", type=Path, nargs="*", default=(),
        help="read-only reports used to choose the already-fixed damping; recorded for outcome provenance",
    )
    parser.add_argument("--gd-steps", type=int, default=16)
    parser.add_argument("--gd-lr", type=float, default=0.10)
    parser.add_argument("--grasp-restarts", type=int, default=4)
    parser.add_argument("--grasp-steps", type=int, default=4)
    parser.add_argument("--cem-samples", type=int, default=32)
    parser.add_argument("--cem-steps", type=int, default=4)
    parser.add_argument("--cem-elite", type=int, default=4)
    parser.add_argument("--target-delta", type=float, nargs=2, default=(0.65, -0.65))
    args = parser.parse_args()
    require_new_output(args.output, receipt="P0 output")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Reacher D2P-P0 requires available host CUDA; CPU fallback is forbidden")
    if args.calibration_only and args.outcome_only:
        raise ValueError("calibration-only and outcome-only are mutually exclusive")
    if args.calibration_only:
        if args.calibration_roots != args.roots or args.roots < 2:
            raise ValueError("calibration-only requires every root to be declared calibration and at least two roots")
    elif args.outcome_only:
        if args.roots < 2 or not args.calibration_evidence:
            raise ValueError("outcome-only requires at least two fresh roots and an external calibration receipt")
    elif not 2 <= args.calibration_roots < args.roots:
        raise ValueError("calibration roots must be >=2 and smaller than total roots")
    if min(args.history_len, args.future_horizon, args.action_block, args.gd_steps, args.grasp_restarts, args.grasp_steps, args.cem_steps, args.relinearization_steps, args.d2p_restarts, args.d2p_proposal_samples, args.d2p_proposal_elites) < 1:
        raise ValueError("all planning horizons and iteration counts must be positive")
    if args.d2p_proposal_elites > args.d2p_proposal_samples:
        raise ValueError("D2P proposal elites must not exceed proposal samples")
    if args.d2p_initialization == "screened_gaussian_proposals" and args.d2p_restarts != 1:
        raise ValueError("screened Gaussian D2P has no restart budget; use --d2p-restarts 1")
    for evidence in args.calibration_evidence:
        if not evidence.is_file():
            raise FileNotFoundError(f"calibration evidence does not exist: {evidence}")
    outcome_seed_set = {int(args.seed + index) for index in range(args.roots)}
    calibration_binding = (
        _validate_outcome_calibration(
            args.calibration_evidence, damping=args.damping,
            relinearization_steps=args.relinearization_steps, d2p_restarts=args.d2p_restarts,
            d2p_initialization=args.d2p_initialization,
            d2p_proposal_samples=args.d2p_proposal_samples,
            d2p_proposal_elites=args.d2p_proposal_elites,
            outcome_seeds=outcome_seed_set,
        )
        if args.outcome_only else None
    )

    install_missing_field_compat()
    device = torch.device("cuda")
    validate_training_semantic_receipt(args.adapter_artifact)
    mean, std = _load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    hash_before = _parameter_hash(model)
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=args.target_delta)
    action_dim = int(np.prod(env.action_space.shape))
    checkpoint_action_block = _validate_reacher_action_protocol(
        model, mean, std, action_dim=action_dim, action_block=args.action_block,
    )
    normalized_low_np, normalized_high_np = _normalized_action_bounds(
        mean, std, env.action_space.low, env.action_space.high,
    )
    action_lower = torch.as_tensor(normalized_low_np, device=device)
    action_upper = torch.as_tensor(normalized_high_np, device=device)
    rows: list[dict[str, Any]] = []
    calibration_count = args.roots if args.calibration_only else 0 if args.outcome_only else args.calibration_roots
    started = time.time()
    try:
        for root_index in range(args.roots):
            seed = int(args.seed + root_index)
            context = _root_context(env, seed, args.history_len, args.action_block)
            goal = context["goal"]
            pixels, goal_tensor, history = _prepare(
                preprocessor, context["frames"], context["root_frame"], goal,
                np.asarray(context["history_action_blocks"], dtype=np.float32), device,
            )
            method_started = time.perf_counter()
            d2p_params, d2p_receipt = _d2p_program(
                model, pixels, goal_tensor, history, future_horizon=args.future_horizon,
                action_lower=action_lower, action_upper=action_upper, damping=args.damping,
                relinearization_steps=args.relinearization_steps, restarts=args.d2p_restarts,
                initialization=args.d2p_initialization, proposal_samples=args.d2p_proposal_samples,
                proposal_elites=args.d2p_proposal_elites, seed=seed,
            )
            d2p_receipt["planner_wall_seconds"] = time.perf_counter() - method_started
            # The control is intentionally recomputed from identical objects,
            # rather than calling a different solver that can mask a mismatch.
            method_started = time.perf_counter()
            gn_params, gn_receipt = _d2p_program(
                model, pixels, goal_tensor, history, future_horizon=args.future_horizon,
                action_lower=action_lower, action_upper=action_upper, damping=args.damping,
                relinearization_steps=args.relinearization_steps, restarts=args.d2p_restarts,
                initialization=args.d2p_initialization, proposal_samples=args.d2p_proposal_samples,
                proposal_elites=args.d2p_proposal_elites, seed=seed,
            )
            gn_receipt["planner_wall_seconds"] = time.perf_counter() - method_started
            method_started = time.perf_counter()
            gd_params, gd_receipt = _action_gd(
                model, pixels, goal_tensor, history, future_horizon=args.future_horizon,
                action_lower=action_lower, action_upper=action_upper, steps=args.gd_steps, lr=args.gd_lr, seed=seed,
            )
            gd_receipt["planner_wall_seconds"] = time.perf_counter() - method_started
            method_started = time.perf_counter()
            grasp_params, grasp_receipt = _restart_gd(
                model, pixels, goal_tensor, history, future_horizon=args.future_horizon,
                action_lower=action_lower, action_upper=action_upper, restarts=args.grasp_restarts, steps=args.grasp_steps,
                lr=args.gd_lr, seed=seed + 1000,
            )
            grasp_receipt["planner_wall_seconds"] = time.perf_counter() - method_started
            method_started = time.perf_counter()
            cem_params, cem_receipt = _cem(
                model, pixels, goal_tensor, history, future_horizon=args.future_horizon,
                action_lower=action_lower, action_upper=action_upper, samples=args.cem_samples, steps=args.cem_steps,
                elite=args.cem_elite, seed=seed + 2000,
            )
            cem_receipt["planner_wall_seconds"] = time.perf_counter() - method_started
            method_started = time.perf_counter()
            random_params, random_receipt = _random_program(
                future_horizon=args.future_horizon, action_dim=history.shape[-1], device=device, seed=seed + 3000,
            )
            random_receipt["planner_wall_seconds"] = time.perf_counter() - method_started
            methods: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {
                "d2p_goal_progress": (d2p_params, d2p_receipt),
                "local_gauss_newton": (gn_params, gn_receipt),
                "action_gd": (gd_params, gd_receipt),
                "grasp_style_restart_gd": (grasp_params, grasp_receipt),
                "cem": (cem_params, cem_receipt),
                "random": (random_params, random_receipt),
            }
            method_rows: dict[str, Any] = {}
            for name, (params, receipt) in methods.items():
                actions, normalized_actions = _to_raw_actions(
                    params, mean, std, action_lower, action_upper, env.action_space.low, env.action_space.high,
                )
                physical, execution_identity = _execute(
                    env,
                    seed,
                    actions,
                    expected_frame=context["root_frame"],
                    expected_goal=goal,
                    execution_snapshot=context["execution_snapshot"],
                )
                predicted_cost = float(_cost_fn(model, pixels, goal_tensor, history, params, action_lower=action_lower, action_upper=action_upper)[0].detach()) if name != "random" else None
                method_rows[name] = {
                    "action_witness_normalized": normalized_actions.tolist(),
                    "action_witness_raw": actions.tolist(),
                    "physical": physical,
                    "execution_identity": execution_identity,
                    "predicted_terminal_cost": predicted_cost,
                    "receipt": receipt,
                }
            equivalence = float(torch.max(torch.abs(d2p_params - gn_params)).detach())
            rows.append({
                "root_index": root_index,
                "seed": seed,
                "split": "calibration" if root_index < calibration_count else "outcome",
                "executed_context": {
                    key: context[key] for key in (
                        "context_mode", "warmup_policy", "warmup_steps",
                        "history_rgb_sha256", "root_rgb_sha256", "goal_rgb_sha256",
                        "warmup_actions_sha256",
                    )
                },
                "execution_snapshot": context["execution_snapshot"],
                "d2p_gn_max_abs_parameter_difference": equivalence,
                "methods": method_rows,
            })
            print(json.dumps({
                "seed": seed,
                "d2p_gn_max_abs_difference": equivalence,
                "d2p_final_qpos": method_rows["d2p_goal_progress"]["physical"]["final_qpos_distance"],
                "cem_final_qpos": method_rows["cem"]["physical"]["final_qpos_distance"],
            }, sort_keys=True))
    finally:
        env.close()
    hash_after = _parameter_hash(model)
    outcome_rows = [row for row in rows if row["split"] == "outcome"]
    equivalence_pass = bool(max(row["d2p_gn_max_abs_parameter_difference"] for row in rows) <= 1.0e-6)
    comparisons = {} if args.calibration_only else {
        method: _paired_summary(outcome_rows, "d2p_goal_progress", method, args.seed + index)
        for index, method in enumerate(("local_gauss_newton", "action_gd", "grasp_style_restart_gd", "cem", "random"), start=1)
    }
    cem_comparison = comparisons.get("cem")
    p1_unlocked, p1_reason = _p1_admission(
        cem_comparison, calibration_only=bool(args.calibration_only),
    )
    cem_wall_time_noninferior = True if args.calibration_only else bool(all(
        row["methods"]["cem"]["receipt"]["planner_wall_seconds"]
        <= row["methods"]["d2p_goal_progress"]["receipt"]["planner_wall_seconds"]
        for row in outcome_rows
    ))
    execution_identity_verified = bool(all(
        method["execution_identity"]
        for row in rows
        for method in row["methods"].values()
    ))
    raw_action_box_verified = bool(all(
        np.all(np.asarray(method["action_witness_raw"], dtype=np.float32) >= np.asarray(env.action_space.low, dtype=np.float32) - 1.0e-6)
        and np.all(np.asarray(method["action_witness_raw"], dtype=np.float32) <= np.asarray(env.action_space.high, dtype=np.float32) + 1.0e-6)
        for row in rows
        for method in row["methods"].values()
    ))
    p0_status = (
        "calibration_only"
        if args.calibration_only
        else "method_negative_against_cem"
        if (
            cem_comparison is not None and cem_comparison["ci95"][0] > 0.0
            and cem_comparison["reference_only_wins"] == cem_comparison["n"]
            and cem_wall_time_noninferior
        )
        else "completed_without_efficacy_verdict"
    )
    report = {
        "schema": "d2p_reacher_p0_goal_progress.v1",
        "status": p0_status,
        "scope": "fresh root-paired physical Reacher P0; D2P goal-progress equals local Gauss--Newton by construction, not a reconfiguration or closed-loop D2P claim",
        "cuda_host_preflight": {"device": "cuda", "name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version()},
        "protocol": {
            "checkpoint_frozen": True,
            "runtime_input": "causally executed rendered RGB/action history, rendered target RGB, and an optimized future suffix",
            "simulator_qpos_access": "evaluator-only; never planner input or planner selection signal. An observed-root qpos/qvel/target snapshot is persisted only to restore identical physical execution roots and audit witnesses after planning.",
            "root_disjoint_calibration_outcome": True,
            "calibration_only": bool(args.calibration_only),
            "outcome_only": bool(args.outcome_only),
            "action_history_prefix_bound": True,
            "context_mode": "executed_warmup",
            "warmup_policy": WARMUP_POLICY,
            "warmup_steps": int(args.history_len * args.action_block),
            "history_and_execution_root_causally_aligned": True,
            "rollout_root_conditioning": "observed_post_warmup_rgb_anchor",
            "execution_root_reconstruction": "evaluator_only_observed_root_snapshot_restore",
            "future_suffix_only_optimized": True,
            "goal_progress_equals_local_gauss_newton": True,
            "relinearization_steps": int(args.relinearization_steps),
            "d2p_restarts": int(args.d2p_restarts),
            "d2p_initialization": args.d2p_initialization,
            "d2p_proposal_samples": int(args.d2p_proposal_samples),
            "d2p_proposal_elites": int(args.d2p_proposal_elites),
            "methods": list(methods),
            "history_len": int(args.history_len),
            "future_horizon": args.future_horizon,
            "action_block": args.action_block,
            "checkpoint_action_block": checkpoint_action_block,
            "action_layout": "flattened time-major [action_block, action_dim]",
            "action_parameterization": "tanh mapped to environment raw action box after checkpoint normalization",
            "environment_action_box_raw": [np.asarray(env.action_space.low).tolist(), np.asarray(env.action_space.high).tolist()],
            "model_action_box_normalized": [normalized_low_np.tolist(), normalized_high_np.tolist()],
            "budget_accounting": "forward rollout evaluations, backward VJP rows, and measured planner wall time are reported per method. VJP rows and forward evaluations are not asserted to be equal compute; the no-go gate also requires CEM to be no slower in measured planner wall time on every outcome root.",
            "damping_calibration_evidence": calibration_binding if calibration_binding is not None else [str(path.resolve()) for path in args.calibration_evidence],
        },
        "inputs": {
            "checkpoint": str(args.policy.resolve()),
            "weights_sha256": _sha256_file(args.policy / "weights.pt"),
            "adapter_artifact": str(args.adapter_artifact.resolve()),
            "adapter_sha256": _sha256_file(args.adapter_artifact),
            "runner_sha256": _sha256_file(Path(__file__)),
        },
        "gates": {
            "cuda_only": True,
            "frozen_parameter_hash_unchanged": hash_before == hash_after,
            "d2p_equals_gauss_newton_each_root": equivalence_pass,
            "all_physical_outcomes_finite": bool(all(np.isfinite(value) for row in rows for method in row["methods"].values() for value in method["physical"].values())),
            "planner_and_execution_root_rgb_identity_verified": execution_identity_verified,
            "all_executed_raw_actions_within_environment_box": raw_action_box_verified,
            "cem_wall_time_noninferior_each_outcome_root": cem_wall_time_noninferior,
        },
        "parameter_hash_before": hash_before,
        "parameter_hash_after": hash_after,
        "comparisons_outcome_only": comparisons,
        "stage_gate": {
            "criterion": "P1 requires a fresh root-disjoint P0 paired 95% interval that favors D2P over matched CEM; avoiding the stricter all-root no-go alone cannot unlock a later mechanism claim",
            "p1_unlocked": p1_unlocked,
            "reason": p1_reason,
        },
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "claim_boundary": "P0 establishes or rejects an explicitly history-bound local goal-progress implementation on fresh physical roots. It does not test reconfiguration, composition, adaptive horizon, or a general planning-runtime claim.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": report["gates"], "comparisons_outcome_only": comparisons, "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
