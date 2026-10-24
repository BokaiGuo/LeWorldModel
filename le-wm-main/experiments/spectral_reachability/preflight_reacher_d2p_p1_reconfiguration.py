"""Fail-closed CUDA preflight for Reacher D2P reconfiguration programs.

This is not a P1 efficacy experiment.  It verifies the data path required
before one can interpret a reconfiguration result: a bounded first action
chunk is selected *only* by a future, recompiled local reachability value; it
is deliberately not the immediately goal-greedy chunk; the chunk is executed
from the observed root; and the second compiler invocation receives the newly
observed RGB/action context.  qpos is read only after physical execution for
diagnostics and never participates in candidate selection.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

# Retain the historical ``python experiments/...`` entry point used by the
# CUDA service launchers as well as module execution from the project root.
if __package__ in (None, ""):
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import torch

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import (
    _jacobian_and_fd,
    _normalized_action_bounds,
    _parameter_hash,
    _sha256_file,
)
from experiments.spectral_reachability.build_reacher_d2p_training_action_stats import validate_training_semantic_receipt
from experiments.spectral_reachability.reacher_d2p_context import (
    WARMUP_POLICY,
    advance_executed_context,
    collect_executed_context,
    restore_execution_snapshot,
)

DEFAULT_RECONFIGURATION_PROPOSALS = 128
DEFAULT_MIN_MODEL_WITNESS_ROOTS = 2
DEFAULT_RECONFIGURATION_PROPOSAL_DISTRIBUTION = "physical_uniform"
RECONFIGURATION_PROPOSAL_DISTRIBUTIONS = (
    "physical_uniform",
    "legacy_tanh_gaussian",
)
DEFAULT_RECONFIGURATION_SELECTION_METRIC = "bounded_dls_suffix_goal_cost"
RECONFIGURATION_SELECTION_METRICS = (
    "bounded_dls_suffix_goal_cost",
    "analytic_v_reach",
)


def _declared_future_score_improvement(
    receipt: dict[str, Any], selection_metric: str,
) -> float:
    """Return the improvement of the exact score used to select a prefix.

    The analytic DLS value is a useful diagnostic, but it is not the score
    selected by the bounded-DLS protocol.  This single conversion prevents a
    later gate from silently reverting to the legacy unbounded surrogate.
    """

    if selection_metric == "analytic_v_reach":
        key = "future_reachability_value_improvement"
    elif selection_metric == "bounded_dls_suffix_goal_cost":
        key = "bounded_dls_suffix_goal_cost_improvement"
    else:
        raise ValueError(f"unknown reconfiguration selection metric: {selection_metric}")
    return float(receipt[key])


def resolve_preflight_report(path: Path) -> Path:
    """Resolve an immutable preflight output directory to its report file.

    P0/P1 preflights emit an output directory containing ``report.json``.
    Accepting the report file as well makes downstream callers explicit while
    still reading old nested outputs created by the former ambiguous CLI use.
    """

    candidate = Path(path)
    for _ in range(3):
        if candidate.is_file():
            return candidate
        if not candidate.is_dir():
            break
        candidate = candidate / "report.json"
    raise ValueError(f"P1 preflight report is not a readable JSON file: {path}")


def _reconfiguration_proposal_bank(
    *, proposals: int, horizon: int, action_dim: int, device: torch.device, dtype: torch.dtype, seed: int,
    distribution: str = DEFAULT_RECONFIGURATION_PROPOSAL_DISTRIBUTION,
) -> torch.Tensor:
    """Return a deterministic physical-action bank while retaining zero.

    Prefix stability is intentional: increasing coverage appends proposals
    rather than changing the first candidates, so a failed small-bank
    diagnostic can be traced to the omitted candidate set rather than a
    different random draw. Planner parameters are transformed by ``tanh``
    before execution. Sampling those parameters from a Normal distribution
    therefore does *not* sample physical actions from the uniform Box used by
    the frozen Reacher calibration data. The default samples the physical
    coordinate first, then applies the exact inverse transform. The legacy
    path remains explicit solely for replaying old immutable receipts.
    """

    if int(proposals) < 2 or int(horizon) < 1 or int(action_dim) < 1:
        raise ValueError("proposal bank dimensions are invalid")
    if distribution not in RECONFIGURATION_PROPOSAL_DISTRIBUTIONS:
        raise ValueError(f"unknown reconfiguration proposal distribution: {distribution}")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    shape = (int(proposals), int(horizon), int(action_dim))
    if distribution == "physical_uniform":
        # _bounded_normalized_actions maps tanh(params) affinely onto each
        # physical action Box. Thus atanh(U[-1, 1]) gives an exactly uniform
        # physical-action proposal after that map, regardless of checkpoint
        # normalization statistics.
        physical_coordinate = 2.0 * torch.rand(shape, generator=generator, device=device, dtype=dtype) - 1.0
        epsilon = torch.finfo(dtype).eps
        bank = torch.atanh(physical_coordinate.clamp(min=-1.0 + epsilon, max=1.0 - epsilon))
    else:
        bank = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    bank[0].zero_()
    return bank


def _state_after_prefix(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    history: torch.Tensor,
    prefix_params: torch.Tensor,
    *,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the frozen latent/action state through a bounded first chunk."""

    if prefix_params.ndim != 3 or prefix_params.shape[0] < 1:
        raise ValueError("reconfiguration prefix must be [samples, horizon, action_dim]")
    samples, _, action_dim = prefix_params.shape
    if tuple(history.shape) != (1, pixels.shape[1], action_dim) or pixels.shape[1] < 2:
        raise ValueError("reconfiguration prefix has incompatible observed context")
    prefix = p0._bounded_normalized_actions(prefix_params, action_lower, action_upper)
    history_size = int(pixels.shape[1])
    emb_context = p0._latent_context(model, pixels, history).expand(samples, -1, -1)
    action_context = torch.cat(
        [history[:, 1:].expand(samples, -1, -1), prefix[:, :1]], dim=1,
    )
    for step in range(prefix.shape[1]):
        predicted = model.predict(
            emb_context[:, -history_size:],
            model.action_encoder(action_context[:, -history_size:]),
        )[:, -1]
        emb_context = torch.cat([emb_context, predicted[:, None]], dim=1)
        if step + 1 < prefix.shape[1]:
            action_context = torch.cat([action_context, prefix[:, step + 1:step + 2]], dim=1)
    return emb_context[:, -history_size:], action_context[:, -history_size:]


def _terminal_from_state(
    model: torch.nn.Module,
    emb_context: torch.Tensor,
    action_context: torch.Tensor,
    future_params: torch.Tensor,
    *,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> torch.Tensor:
    """Predict a suffix from an already advanced frozen model state."""

    if future_params.ndim != 3 or future_params.shape[0] != emb_context.shape[0]:
        raise ValueError("future suffix batch does not match reconfiguration state")
    if emb_context.shape[:2] != action_context.shape[:2] or emb_context.shape[1] < 2:
        raise ValueError("reconfiguration state has incompatible history")
    future = p0._bounded_normalized_actions(future_params, action_lower, action_upper)
    history_size = int(emb_context.shape[1])
    actions = torch.cat([action_context, future[:, :1]], dim=1)
    embeddings = emb_context
    terminal = None
    for step in range(future.shape[1]):
        terminal = model.predict(
            embeddings[:, -history_size:],
            model.action_encoder(actions[:, -history_size:]),
        )[:, -1]
        if step + 1 < future.shape[1]:
            embeddings = torch.cat([embeddings, terminal[:, None]], dim=1)
            actions = torch.cat([actions, future[:, step + 1:step + 2]], dim=1)
    if terminal is None:
        raise RuntimeError("recompiled suffix has no terminal prediction")
    return terminal


def _reachability_value(
    model: torch.nn.Module,
    goal_latent: torch.Tensor,
    emb_context: torch.Tensor,
    action_context: torch.Tensor,
    *,
    future_horizon: int,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
    damping: float,
    return_suffix_params: bool = False,
) -> dict[str, Any]:
    """Compute the local DLS reachability value at one post-prefix state."""

    if emb_context.shape[0] != action_context.shape[0] or emb_context.shape[0] != 1:
        raise ValueError("local reachability requires exactly one reconfiguration state")
    action_dim = int(action_context.shape[-1])
    origin = torch.zeros(future_horizon * action_dim, device=emb_context.device, requires_grad=True)

    def terminal(flat: torch.Tensor) -> torch.Tensor:
        return _terminal_from_state(
            model, emb_context, action_context,
            flat.reshape(1, future_horizon, action_dim),
            action_lower=action_lower, action_upper=action_upper,
        )[0]

    predicted = terminal(origin)
    jacobian = torch.autograd.functional.jacobian(terminal, origin, vectorize=True)
    if jacobian.ndim != 2 or not torch.isfinite(jacobian).all() or float(jacobian.norm()) <= 1.0e-10:
        raise RuntimeError("post-reconfiguration suffix Jacobian is non-finite or zero")
    residual = goal_latent - predicted
    left, singular, right_h = torch.linalg.svd(jacobian, full_matrices=False)
    coordinates = left.T @ residual
    orthogonal = residual - left @ coordinates
    value = orthogonal.square().sum() + (
        float(damping) / (singular.square() + float(damping)) * coordinates.square()
    ).sum()
    # ``value`` is the documented unconstrained local DLS objective.  The
    # actually executed suffix, however, is bounded by tanh.  Record the
    # terminal cost reached by the same DLS update after that exact bound so
    # selection audits can detect an analytic-versus-executable rank reversal
    # without consulting evaluator-only qpos.
    update = right_h.T @ (
        singular / (singular.square() + float(damping)) * coordinates
    )
    with torch.no_grad():
        bounded_terminal = terminal(update)
    receipt: dict[str, Any] = {
        "value": float(value.detach()),
        "predicted_terminal_goal_cost": float((predicted - goal_latent).square().mean().detach()),
        "bounded_dls_suffix_goal_cost": float((bounded_terminal - goal_latent).square().mean().detach()),
        "unbounded_dls_parameter_l2": float(torch.linalg.vector_norm(update).detach()),
        "jacobian_frobenius_norm": float(jacobian.norm().detach()),
        "singular_values": [float(item) for item in singular.detach().cpu()],
        "effective_rank_1e-4": int((singular > singular[0] * 1.0e-4).sum()),
    }
    if return_suffix_params:
        receipt["_bounded_dls_suffix_params"] = update.detach().reshape(1, future_horizon, action_dim)
    return receipt


def _bounded_single_step_dls_program(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    goal: torch.Tensor,
    history: torch.Tensor,
    *,
    future_horizon: int,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
    damping: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compile the exact bounded one-step suffix used by P1 ranking.

    The prefix selector ranks a predicted successor by this solver. P1 must
    recompile it at the observed successor, but must not silently execute a
    different screened multi-step optimizer.
    """

    emb_context = p0._latent_context(model, pixels, history)
    goal_latent = p0._goal_latent(model, goal)
    reach = _reachability_value(
        model, goal_latent, emb_context, history,
        future_horizon=future_horizon, action_lower=action_lower,
        action_upper=action_upper, damping=damping, return_suffix_params=True,
    )
    params = reach.pop("_bounded_dls_suffix_params")
    return params, {
        "program_type": "bounded_single_step_dls",
        "initialization": "zero",
        "selection_equivalent_suffix_solver": True,
        "predicted_cost_before": reach["predicted_terminal_goal_cost"],
        "predicted_cost_after": reach["bounded_dls_suffix_goal_cost"],
        "jacobian_frobenius_norm": reach["jacobian_frobenius_norm"],
        "singular_values": reach["singular_values"],
        "effective_rank_1e-4": reach["effective_rank_1e-4"],
        "damping": float(damping),
        "relinearization_steps_requested": 1,
        "relinearization_steps_executed": 1,
        "forward_rollout_evaluations": 1,
        "backward_vjp_rows": len(reach["singular_values"]),
        "actual_model_cost_nonincreasing": reach["bounded_dls_suffix_goal_cost"] <= reach["predicted_terminal_goal_cost"] + 1.0e-6,
    }


def _compile_reconfiguration(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    goal: torch.Tensor,
    history: torch.Tensor,
    *,
    reconfiguration_horizon: int,
    future_horizon: int,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
    damping: float,
    proposals: int,
    non_greedy_margin: float,
    reachability_margin: float,
    seed: int,
    max_non_greedy_margin: float | None = None,
    max_relative_reachability_improvement: float | None = None,
    min_second_to_first_singular_ratio: float | None = None,
    proposal_distribution: str = DEFAULT_RECONFIGURATION_PROPOSAL_DISTRIBUTION,
    selection_metric: str = DEFAULT_RECONFIGURATION_SELECTION_METRIC,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Select a non-greedy prefix by its *future* local reachability value."""

    if proposals < 2 or reconfiguration_horizon < 1 or future_horizon < 1:
        raise ValueError("reconfiguration proposals and horizons must be positive")
    if non_greedy_margin < 0.0 or (max_non_greedy_margin is not None and max_non_greedy_margin < non_greedy_margin):
        raise ValueError("reconfiguration non-greedy margin band is invalid")
    if max_relative_reachability_improvement is not None and not 0.0 < max_relative_reachability_improvement <= 1.0:
        raise ValueError("maximum relative reachability improvement must lie in (0, 1]")
    if min_second_to_first_singular_ratio is not None and not 0.0 < min_second_to_first_singular_ratio <= 1.0:
        raise ValueError("minimum second-to-first singular ratio must lie in (0, 1]")
    if selection_metric not in RECONFIGURATION_SELECTION_METRICS:
        raise ValueError(f"unknown reconfiguration selection metric: {selection_metric}")
    bank = _reconfiguration_proposal_bank(
        proposals=proposals, horizon=reconfiguration_horizon, action_dim=history.shape[-1],
        device=history.device, dtype=history.dtype, seed=seed, distribution=proposal_distribution,
    )
    goal_latent = p0._goal_latent(model, goal)
    immediate = p0._cost_fn(
        model, pixels, goal, history, bank,
        action_lower=action_lower, action_upper=action_upper,
    ).detach()
    rows: list[dict[str, Any]] = []
    for index in range(int(proposals)):
        state = _state_after_prefix(
            model, pixels, history, bank[index:index + 1],
            action_lower=action_lower, action_upper=action_upper,
        )
        reach = _reachability_value(
            model, goal_latent, *state, future_horizon=future_horizon,
            action_lower=action_lower, action_upper=action_upper, damping=damping,
        )
        rows.append({"proposal_index": index, "immediate_goal_cost": float(immediate[index]), **reach})
    zero = rows[0]
    if not np.isfinite(zero["value"]) or zero["value"] <= 0.0:
        raise RuntimeError("zero-prefix reachability value must be finite and positive")
    for row in rows:
        row["relative_reachability_value_improvement"] = float((zero["value"] - row["value"]) / zero["value"])
        row["bounded_dls_suffix_goal_cost_improvement"] = float(
            zero["bounded_dls_suffix_goal_cost"] - row["bounded_dls_suffix_goal_cost"]
        )
        singular = row["singular_values"]
        if len(singular) < 2 or singular[0] <= 0.0:
            raise RuntimeError("reconfiguration Jacobian requires two positive leading singular values")
        row["second_to_first_singular_ratio"] = float(singular[1] / singular[0])
    selected_score = "value" if selection_metric == "analytic_v_reach" else "bounded_dls_suffix_goal_cost"
    eligible = [
        row for row in rows[1:]
        if row["immediate_goal_cost"] >= zero["immediate_goal_cost"] + float(non_greedy_margin)
        and (max_non_greedy_margin is None or row["immediate_goal_cost"] <= zero["immediate_goal_cost"] + float(max_non_greedy_margin))
        and row[selected_score] <= zero[selected_score] - float(reachability_margin)
        and (max_relative_reachability_improvement is None or row["relative_reachability_value_improvement"] <= max_relative_reachability_improvement)
        and (min_second_to_first_singular_ratio is None or row["second_to_first_singular_ratio"] >= min_second_to_first_singular_ratio)
    ]
    if not eligible:
        return None, {
            "program_type": "reconfiguration_by_recompiled_local_reachability",
            "proposal_distribution": proposal_distribution,
            "selection_metric": selection_metric,
            "proposal_rows": rows,
            "selected_proposal_index": None,
            "non_greedy_witness_found": False,
            "reason": "no proposal lies in the frozen immediate-cost, relative-reachability-improvement, and spectral-balance bands while improving the declared bounded or analytic future score versus no reconfiguration",
            "forward_rollout_evaluations": int(proposals),
            "backward_vjp_rows": int(sum(len(row["singular_values"]) for row in rows)),
        }
    selected = min(eligible, key=lambda row: (row[selected_score], row["proposal_index"]))
    return bank[selected["proposal_index"]:selected["proposal_index"] + 1].detach(), {
        "program_type": "reconfiguration_by_recompiled_local_reachability",
        "proposal_distribution": proposal_distribution,
        "selection_metric": selection_metric,
        "proposal_rows": rows,
        "selected_proposal_index": int(selected["proposal_index"]),
        "non_greedy_witness_found": True,
        "selection_rule": "minimum declared future score among chunks in frozen immediate-cost, relative-reachability-improvement, and spectral-balance bands that improve that score versus no reconfiguration",
        "non_greedy_model_margin": float(selected["immediate_goal_cost"] - zero["immediate_goal_cost"]),
        "min_non_greedy_margin": float(non_greedy_margin),
        "max_non_greedy_margin": None if max_non_greedy_margin is None else float(max_non_greedy_margin),
        "future_reachability_value_improvement": float(zero["value"] - selected["value"]),
        "bounded_dls_suffix_goal_cost_improvement": selected["bounded_dls_suffix_goal_cost_improvement"],
        "relative_reachability_value_improvement": selected["relative_reachability_value_improvement"],
        "max_relative_reachability_improvement": (
            None if max_relative_reachability_improvement is None else float(max_relative_reachability_improvement)
        ),
        "second_to_first_singular_ratio": selected["second_to_first_singular_ratio"],
        "min_second_to_first_singular_ratio": (
            None if min_second_to_first_singular_ratio is None else float(min_second_to_first_singular_ratio)
        ),
        "forward_rollout_evaluations": int(proposals),
        "backward_vjp_rows": int(sum(len(row["singular_values"]) for row in rows)),
    }


def _validate_upstream_p0(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "d2p_reacher_p0_goal_progress.v1":
        raise ValueError("P1 requires a Reacher D2P P0 report")
    if not report.get("stage_gate", {}).get("p1_unlocked"):
        raise ValueError("P1 requires an affirmative P0 admission report")
    if not all(report.get("gates", {}).values()):
        raise ValueError("P1 upstream P0 report has a failed gate")
    if report.get("protocol", {}).get("warmup_policy") != WARMUP_POLICY:
        raise ValueError("P1 upstream P0 report uses an incompatible warmup action-packet policy")
    return {"path": str(path.resolve()), "sha256": _sha256_file(path), "p1_unlocked": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New immutable preflight output directory.")
    parser.add_argument("--p0-evidence", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm"))
    parser.add_argument("--adapter-artifact", type=Path, default=Path("results/live_cross_environment_prediction_20261015/reacher_cuda/cross_environment_prediction.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20262700)
    parser.add_argument("--roots", type=int, default=2)
    parser.add_argument(
        "--min-model-witness-roots", type=int, default=DEFAULT_MIN_MODEL_WITNESS_ROOTS,
        help="Minimum model-eligible roots required to exercise the physical reconfiguration plumbing.",
    )
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--reconfiguration-horizon", type=int, default=1)
    parser.add_argument("--future-horizon", type=int, default=2)
    parser.add_argument(
        "--proposals", type=int, default=DEFAULT_RECONFIGURATION_PROPOSALS,
        help="Model-only reconfiguration candidates; defaults to the P0 coverage-matched 128-proposal budget.",
    )
    parser.add_argument(
        "--proposal-distribution", choices=RECONFIGURATION_PROPOSAL_DISTRIBUTIONS,
        default=DEFAULT_RECONFIGURATION_PROPOSAL_DISTRIBUTION,
        help="Frozen candidate-bank measure; physical_uniform matches the calibration action Box.",
    )
    parser.add_argument(
        "--selection-metric", choices=RECONFIGURATION_SELECTION_METRICS,
        default=DEFAULT_RECONFIGURATION_SELECTION_METRIC,
        help="Future candidate score; bounded DLS cost enforces the physical action map during ranking.",
    )
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--non-greedy-margin", type=float, default=1.0e-6)
    parser.add_argument(
        "--max-relative-reachability-improvement", type=float, default=None,
        help="Optional model-only upper guard on relative V_reach improvement.",
    )
    parser.add_argument(
        "--min-second-to-first-singular-ratio", type=float, default=None,
        help="Optional model-only lower guard on spectral balance after the reconfiguration prefix.",
    )
    parser.add_argument("--reachability-margin", type=float, default=1.0e-6)
    parser.add_argument("--target-delta", type=float, nargs=2, default=(0.65, -0.65))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable preflight output: {args.output}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Reacher D2P-P1 requires host CUDA exactly; CPU fallback is forbidden")
    if min(args.roots, args.history_len, args.action_block, args.reconfiguration_horizon, args.future_horizon, args.proposals) < 1 or args.proposals < 2:
        raise ValueError("invalid positive reconfiguration preflight dimensions")
    if not 1 <= args.min_model_witness_roots <= args.roots:
        raise ValueError("preflight minimum model witness roots must lie in [1, roots]")
    if args.damping <= 0 or args.non_greedy_margin < 0 or args.reachability_margin < 0:
        raise ValueError("damping must be positive and margins non-negative")
    if args.max_relative_reachability_improvement is not None and not 0.0 < args.max_relative_reachability_improvement <= 1.0:
        raise ValueError("maximum relative reachability improvement must lie in (0, 1]")
    if args.min_second_to_first_singular_ratio is not None and not 0.0 < args.min_second_to_first_singular_ratio <= 1.0:
        raise ValueError("minimum second-to-first singular ratio must lie in (0, 1]")
    upstream = _validate_upstream_p0(args.p0_evidence)

    install_missing_field_compat()
    device = torch.device("cuda")
    validate_training_semantic_receipt(args.adapter_artifact)
    mean, std = _load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    hash_before = _parameter_hash(model)
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=args.target_delta)
    action_dim = int(np.prod(env.action_space.shape))
    checkpoint_block = p0._validate_reacher_action_protocol(model, mean, std, action_dim=action_dim, action_block=args.action_block)
    normalized_low, normalized_high = _normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(normalized_low, device=device), torch.as_tensor(normalized_high, device=device)
    rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for offset in range(args.roots):
            seed = int(args.seed + offset)
            context = collect_executed_context(env, seed=seed, history_len=args.history_len, action_block=args.action_block)
            pixels, goal, history = p0._prepare(preprocessor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], dtype=np.float32), device)
            params, compilation = _compile_reconfiguration(
                model, pixels, goal, history, reconfiguration_horizon=args.reconfiguration_horizon,
                future_horizon=args.future_horizon, action_lower=lower, action_upper=upper,
                damping=args.damping, proposals=args.proposals,
                non_greedy_margin=args.non_greedy_margin, reachability_margin=args.reachability_margin,
                seed=seed, max_relative_reachability_improvement=args.max_relative_reachability_improvement,
                min_second_to_first_singular_ratio=args.min_second_to_first_singular_ratio,
                proposal_distribution=args.proposal_distribution,
                selection_metric=args.selection_metric,
            )
            row: dict[str, Any] = {
                "seed": seed,
                "initial_context": {key: context[key] for key in ("context_mode", "warmup_policy", "root_rgb_sha256", "goal_rgb_sha256", "warmup_actions_sha256")},
                "reconfiguration": compilation,
            }
            if params is not None:
                raw, normalized = p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)
                root, rendered_goal = restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
                initial_identity = p0._verify_execution_identity(root, rendered_goal, context["root_frame"], context["goal"])
                advanced = advance_executed_context(env, context=context, raw_action_block=raw, action_block=args.action_block)
                successor_pixels, successor_goal, successor_history = p0._prepare(
                    preprocessor, advanced["frames"], advanced["root_frame"], advanced["goal"],
                    np.asarray(advanced["history_action_blocks"], dtype=np.float32), device,
                )
                successor_audit = _jacobian_and_fd(
                    model, successor_pixels, successor_goal, successor_history,
                    history_len=args.history_len, future_horizon=args.future_horizon,
                    probe_dims=4, fd_pairs=4, epsilon=1.0e-3,
                    action_lower=lower, action_upper=upper, seed=seed + 7000, device=device,
                )
                progress, progress_receipt = _bounded_single_step_dls_program(
                    model, successor_pixels, successor_goal, successor_history,
                    future_horizon=args.future_horizon, action_lower=lower,
                    action_upper=upper, damping=args.damping,
                )
                progress_raw, progress_normalized = p0._to_raw_actions(progress, mean, std, lower, upper, env.action_space.low, env.action_space.high)
                physical, successor_identity = p0._execute(
                    env, seed, progress_raw, expected_frame=advanced["root_frame"], expected_goal=advanced["goal"], execution_snapshot=advanced["execution_snapshot"],
                )
                row.update({
                    "reconfiguration_action_witness_normalized": normalized.tolist(),
                    "reconfiguration_action_witness_raw": raw.tolist(),
                    "initial_execution_identity": initial_identity,
                    "observed_successor_context": {key: advanced[key] for key in ("context_mode", "parent_context_mode", "root_rgb_sha256", "goal_rgb_sha256", "advanced_action_sha256")},
                    "successor_jacobian": successor_audit,
                    "recompiled_progress_action_witness_normalized": progress_normalized.tolist(),
                    "recompiled_progress_action_witness_raw": progress_raw.tolist(),
                    "recompiled_progress_receipt": progress_receipt,
                    "successor_execution_identity": successor_identity,
                    "two_stage_physical": physical,
                })
            rows.append(row)
            print(json.dumps({"seed": seed, "non_greedy_witness_found": compilation["non_greedy_witness_found"]}, sort_keys=True))
    finally:
        env.close()
    hash_after = _parameter_hash(model)
    completed = [row for row in rows if row["reconfiguration"]["non_greedy_witness_found"]]
    for row in completed:
        row["reconfiguration"]["declared_future_score_improvement"] = _declared_future_score_improvement(
            row["reconfiguration"], args.selection_metric,
        )
    successor_audits = [row["successor_jacobian"] for row in completed]
    gates = {
        "cuda_only": True,
        "upstream_p0_admission_verified": upstream["p1_unlocked"],
        "frozen_parameter_hash_unchanged": hash_before == hash_after,
        "at_least_declared_model_selected_non_greedy_reconfiguration_witnesses": (
            len(completed) >= args.min_model_witness_roots
        ),
        "every_selected_chunk_improves_declared_recompiled_future_score": bool(completed) and all(
            row["reconfiguration"]["declared_future_score_improvement"] > 0.0 for row in completed
        ),
        "every_selected_chunk_is_immediately_non_greedy_in_frozen_goal_cost": bool(completed) and all(row["reconfiguration"]["non_greedy_model_margin"] > 0.0 for row in completed),
        "observed_successor_recompiled_with_nonzero_fd_validated_jacobian": bool(successor_audits) and all(item["frobenius_norm"] > 1.0e-10 and item["median_relative_error"] <= 0.10 for item in successor_audits),
        "root_and_successor_execution_rgb_identity_verified": bool(completed) and all(row["initial_execution_identity"] and row["successor_execution_identity"] for row in completed),
        "all_raw_actions_within_environment_box": bool(completed) and all(
            np.all(np.asarray(row["reconfiguration_action_witness_raw"]) >= np.asarray(env.action_space.low) - 1.0e-6)
            and np.all(np.asarray(row["reconfiguration_action_witness_raw"]) <= np.asarray(env.action_space.high) + 1.0e-6)
            and np.all(np.asarray(row["recompiled_progress_action_witness_raw"]) >= np.asarray(env.action_space.low) - 1.0e-6)
            and np.all(np.asarray(row["recompiled_progress_action_witness_raw"]) <= np.asarray(env.action_space.high) + 1.0e-6)
            for row in completed
        ),
    }
    report = {
        "schema": "d2p_reacher_p1_reconfiguration_preflight.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "CUDA-only plumbing admission for a reconfiguration program; no calibration, outcome comparison, or P1 efficacy conclusion",
        "cuda_host_preflight": {"device": "cuda", "name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version()},
        "upstream_p0_evidence": upstream,
        "protocol": {
            "runtime_input": "observed RGB/action history and rendered goal only; simulator state is evaluator-only",
            "reconfiguration_selection": "the declared future score after the predicted chunk; immediate frozen goal cost is only a non-greedy admissibility constraint",
            "history_len": args.history_len,
            "min_model_witness_roots": args.min_model_witness_roots,
            "reconfiguration_horizon": args.reconfiguration_horizon,
            "recompiled_progress_horizon": args.future_horizon,
            "total_executed_action_tokens": args.reconfiguration_horizon + args.future_horizon,
            "action_block": args.action_block,
            "checkpoint_action_block": checkpoint_block,
            "context_mode": "executed_warmup then observed_successor_recompile",
            "warmup_policy": WARMUP_POLICY,
            "target_delta": list(args.target_delta),
            "damping": args.damping,
            "proposals": args.proposals,
            "proposal_distribution": args.proposal_distribution,
            "selection_metric": args.selection_metric,
            "max_relative_reachability_improvement": args.max_relative_reachability_improvement,
            "min_second_to_first_singular_ratio": args.min_second_to_first_singular_ratio,
            "recompiled_suffix_solver": "bounded_single_step_dls; identical solver family to the bounded selector score",
        },
        "inputs": {"checkpoint": str(args.policy.resolve()), "weights_sha256": _sha256_file(args.policy / "weights.pt"), "adapter_artifact": str(args.adapter_artifact.resolve()), "adapter_sha256": _sha256_file(args.adapter_artifact), "runner_sha256": _sha256_file(Path(__file__))},
        "parameter_hash_before": hash_before,
        "parameter_hash_after": hash_after,
        "gates": gates,
        "eligibility": {
            "model_witness_root_count": len(completed),
            "model_ineligible_root_count": len(rows) - len(completed),
            "minimum_model_witness_roots": args.min_model_witness_roots,
        },
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "claim_boundary": "Passing only rules out this stage's context/root/action/Jacobian plumbing failures. It does not show physical reconfiguration benefit, surpass a no-reconfiguration program, or establish a planning-runtime claim.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
