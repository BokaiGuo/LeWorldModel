"""CUDA-only physical runner for the Reacher D2P P1 reconfiguration test.

P1 is deliberately a two-stage closed-loop protocol.  Its reconfiguration
chunk is chosen from frozen RGB dynamics by post-chunk reachability, then is
physically executed.  The progress chunk is compiled from the *observed*
successor, never a simulated successor.  Greedy two-stage D2P and full-horizon
CEM execute the same three action tokens, from the same evaluator-only root.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

# Support the direct-script CUDA service entry point as well as ``-m``.
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
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import _normalized_action_bounds, _parameter_hash, _sha256_file
from experiments.spectral_reachability.reacher_d2p_context import WARMUP_POLICY, advance_executed_context, collect_executed_context, restore_execution_snapshot
from experiments.spectral_reachability.build_reacher_d2p_training_action_stats import validate_training_semantic_receipt

MECHANISM_OBSERVABILITIES = (
    "qpos_distance",
    "observed_encoded_goal_cost",
)


def _resolve_local_cli_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """Make explicitly supplied local evidence independent of the shell CWD.

    ``load_compatible_pretrained`` intentionally interprets relative names as
    cache checkpoint identifiers.  The P1 CLI, however, documents ``--policy``
    (and the evidence arguments) as filesystem paths.  Resolve them at the
    boundary so a normal repository-relative invocation cannot silently change
    which checkpoint is requested.
    """

    for attribute in (
        "policy",
        "adapter_artifact",
        "p0_evidence",
        "p1_preflight_evidence",
        "calibration_evidence",
        "margin_band_selection_evidence",
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            setattr(args, attribute, Path(value).expanduser().resolve())
    # This option is repeatable on the CLI.  Treating its default empty list
    # as a single Path caused every ordinary P1 run to fail before CUDA work.
    # Resolve each supplied manifest at the boundary while preserving an empty
    # evidence set as an empty list.
    manifests = getattr(args, "training_source_manifest", None)
    if manifests is not None:
        setattr(args, "training_source_manifest", [
            Path(value).expanduser().resolve() for value in manifests
        ])
    return args


def _reserve_immutable_output(path: Path) -> Path:
    """Atomically claim an output path before expensive CUDA work begins.

    Testing ``exists()`` at startup but creating the directory only after all
    roots have run permits two launchers to evaluate the same panel and race
    while publishing their receipt.  The directory creation itself is the
    cross-process reservation; a concurrent launcher now fails before it can
    consume GPU time or execute overlapping roots.
    """

    path.mkdir(parents=True, exist_ok=False)
    return path


def _bind_simulator_training_sources(paths: list[Path], *, run_seeds: set[int]) -> list[dict[str, Any]]:
    """Reject planning roots that appeared in a simulator-generated training pool.

    Public archive training data has no Reacher reset-seed namespace.  In
    contrast, a counterfactual simulator pool does.  A repaired checkpoint
    trained from such a pool must never be evaluated on one of its pool roots,
    even if the downstream run is called a calibration rather than an outcome.
    """

    bindings: list[dict[str, Any]] = []
    for supplied in paths:
        manifest_path = Path(supplied).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = manifest_path
        if payload.get("schema") == "reacher_d2p_counterfactual_calibration_repair.v1":
            source_path = Path(payload.get("counterfactual_pool_manifest", "")).resolve()
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "reacher_d2p_counterfactual_pair_pool.v1" or payload.get("status") != "completed":
            raise ValueError("training-source manifest must be a completed counterfactual pool or its repair manifest")
        seed_range = payload.get("seed_range")
        if not isinstance(seed_range, list) or len(seed_range) != 2:
            raise ValueError("counterfactual training source is missing its seed range")
        lower, upper = (int(seed_range[0]), int(seed_range[1]))
        if upper < lower:
            raise ValueError("counterfactual training source has an invalid seed range")
        overlap = sorted(run_seeds & set(range(lower, upper + 1)))
        if overlap:
            raise ValueError(f"planning roots overlap simulator training roots: {overlap[:8]}")
        bindings.append({
            "supplied_manifest": str(manifest_path), "supplied_sha256": _sha256_file(manifest_path),
            "pool_manifest": str(source_path), "pool_manifest_sha256": _sha256_file(source_path),
            "reserved_training_seed_range": [lower, upper],
        })
    return bindings


def _physically_non_greedy(method: dict[str, Any], *, tolerance: float = 1.0e-8) -> bool:
    """Check the P1 mechanism against the same-length zero-prefix control.

    Model eligibility compares a reconfiguration prefix to the frozen
    zero-prefix.  The evaluator must preserve that counterfactual: a warmup
    root can carry velocity, so comparing the candidate's successor directly
    with the pre-action root would confound reconfiguration with passive
    dynamics during the five environment steps.
    """

    physical = method.get("physical", {})
    candidate_after = float(physical["after_first_chunk_qpos_distance"])
    zero_after = float(physical["after_zero_prefix_qpos_distance"])
    return bool(candidate_after >= zero_after - float(tolerance))


def _mechanism_non_greedy(
    method: dict[str, Any], *, observability: str, tolerance: float = 1.0e-8,
) -> bool:
    """Check a preregistered P1 first-chunk mechanism coordinate.

    The original coordinate is evaluator-only qpos distance.  The observable
    coordinate instead compares frozen encoder cost to the rendered goal on
    the two actual successor RGB images.  Neither coordinate enters proposal
    generation or model-side ranking.
    """

    if observability == "qpos_distance":
        return _physically_non_greedy(method, tolerance=tolerance)
    if observability == "observed_encoded_goal_cost":
        observed = method["observed_first_chunk"]
        candidate = float(observed["candidate_encoded_goal_cost"])
        zero = float(observed["zero_prefix_encoded_goal_cost"])
        return bool(candidate >= zero - float(tolerance))
    raise ValueError(f"unknown P1 mechanism observability: {observability}")


def _mechanism_gate_name(observability: str) -> str:
    if observability == "qpos_distance":
        return "every_model_eligible_reconfiguration_is_physically_non_greedy_relative_to_zero_prefix"
    if observability == "observed_encoded_goal_cost":
        return "every_model_eligible_reconfiguration_is_observably_non_greedy_relative_to_zero_prefix"
    raise ValueError(f"unknown P1 mechanism observability: {observability}")


def _validate_preflight(path: Path, *, args: argparse.Namespace) -> dict[str, Any]:
    """Require a preflight for this exact frozen planning contract.

    A successful CUDA plumbing receipt is not portable across checkpoints,
    action normalizers, target renderings, or planner geometries.  Binding it
    here prevents a later P1 panel from borrowing a preflight whose observed
    successor recurrence was tested under different dynamics.
    """

    report_path = p1.resolve_preflight_report(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "d2p_reacher_p1_reconfiguration_preflight.v1" or report.get("status") != "pass":
        raise ValueError("P1 physical runner requires a passing P1 CUDA plumbing preflight")
    if not all(report.get("gates", {}).values()):
        raise ValueError("P1 plumbing preflight contains a failed gate")
    inputs = report.get("inputs", {})
    expected_inputs = {
        "weights_sha256": _sha256_file(args.policy / "weights.pt"),
        "adapter_sha256": _sha256_file(args.adapter_artifact),
        "runner_sha256": _sha256_file(Path(p1.__file__)),
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise ValueError(f"P1 plumbing preflight is incompatible on {key}")
    if report.get("upstream_p0_evidence", {}).get("sha256") != _sha256_file(args.p0_evidence):
        raise ValueError("P1 plumbing preflight is bound to a different P0 evidence receipt")
    protocol = report.get("protocol", {})
    expected_protocol = {
        "history_len": 3,
        "min_model_witness_roots": p1.DEFAULT_MIN_MODEL_WITNESS_ROOTS,
        "reconfiguration_horizon": args.reconfiguration_horizon,
        "recompiled_progress_horizon": args.progress_horizon,
        "action_block": args.action_block,
        "proposals": args.proposals,
        "proposal_distribution": args.proposal_distribution,
        "selection_metric": args.selection_metric,
        "damping": args.damping,
        "max_relative_reachability_improvement": args.max_relative_reachability_improvement,
        "min_second_to_first_singular_ratio": args.min_second_to_first_singular_ratio,
        "warmup_policy": WARMUP_POLICY,
        "target_delta": list(args.target_delta),
    }
    for key, expected in expected_protocol.items():
        actual = protocol.get(key)
        if isinstance(expected, float):
            if actual is None or not np.isclose(float(actual), expected, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"P1 plumbing preflight is incompatible on {key}")
        elif actual != expected:
            raise ValueError(f"P1 plumbing preflight is incompatible on {key}")
    return {"path": str(report_path.resolve()), "sha256": _sha256_file(report_path)}


def _validate_margin_band_selection(path: Path, *, args: argparse.Namespace) -> dict[str, Any]:
    """Bind a bounded calibration to a sealed calibration-only selector."""

    selection = json.loads(path.read_text(encoding="utf-8"))
    schema = selection.get("schema")
    allowed = {"d2p_reacher_p1_margin_band_guard_selection.v1", "d2p_reacher_p1_task_critical_margin_band_selection.v1"}
    if schema not in allowed or selection.get("status") != "pass":
        raise ValueError("P1 margin-band calibration requires a passing margin-band selection receipt")
    if schema == "d2p_reacher_p1_task_critical_margin_band_selection.v1" and args.mechanism_observability != "qpos_distance":
        raise ValueError("task-critical margin-band selection requires qpos_distance observability")
    selected = selection.get("selected", {})
    expected_bounds = {
        "min_non_greedy_margin": args.non_greedy_margin,
        "max_non_greedy_margin": args.max_non_greedy_margin,
    }
    for key, value in expected_bounds.items():
        if value is None or not np.isclose(float(selected.get(key, float("nan"))), float(value), rtol=0.0, atol=1.0e-12):
            raise ValueError("P1 margin-band bounds differ from their selection receipt")
    binding = selection.get("protocol_binding", {})
    expected_protocol = {
        "checkpoint_weights_sha256": _sha256_file(args.policy / "weights.pt"),
        "adapter_sha256": _sha256_file(args.adapter_artifact),
        "runner_sha256": _sha256_file(Path(__file__)),
        "mechanism_observability": args.mechanism_observability,
        "reconfiguration_horizon": args.reconfiguration_horizon,
        "recompiled_progress_horizon": args.progress_horizon,
        "proposals": args.proposals,
        "proposal_distribution": args.proposal_distribution,
        "selection_metric": args.selection_metric,
        "damping": args.damping,
        "action_block": args.action_block,
        "max_relative_reachability_improvement": args.max_relative_reachability_improvement,
        "min_second_to_first_singular_ratio": args.min_second_to_first_singular_ratio,
        "warmup_policy": WARMUP_POLICY,
        "target_delta": list(args.target_delta),
    }
    for key, value in expected_protocol.items():
        actual = binding.get(key)
        if isinstance(value, float):
            if actual is None or not np.isclose(float(actual), value, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"P1 margin-band selection is incompatible on {key}")
        elif actual != value:
            raise ValueError(f"P1 margin-band selection is incompatible on {key}")
    inputs = selection.get("inputs", {})
    sources: dict[str, dict[str, str]] = {}
    required_sources = ("report",) if schema == "d2p_reacher_p1_task_critical_margin_band_selection.v1" else ("report", "fidelity")
    for label in required_sources:
        source = Path(inputs.get(label, ""))
        source_sha = inputs.get(f"{label}_sha256")
        if not source.is_file() or not isinstance(source_sha, str) or _sha256_file(source) != source_sha:
            raise ValueError(f"P1 margin-band selection source {label} is missing or has changed")
        sources[label] = {"path": str(source.resolve()), "sha256": source_sha}
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "selected_bounds": expected_bounds,
        "source_runner_sha256": binding.get("runner_sha256"),
        "source_receipts": sources,
    }


def _validate_outcome_calibration(path: Path, *, args: argparse.Namespace, outcome_seeds: set[int]) -> dict[str, Any]:
    """Bind an efficacy panel to one compatible, root-disjoint calibration receipt."""

    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "d2p_reacher_p1_reconfiguration.v1" or report.get("status") != "calibration_only":
        raise ValueError("outcome-only requires a passing P1 calibration-only receipt")
    if not all(report.get("gates", {}).values()) or not report.get("protocol", {}).get("calibration_only"):
        raise ValueError("P1 calibration receipt has a failed gate or is not calibration-only")
    if report.get("inputs", {}).get("runner_sha256") != _sha256_file(Path(__file__)):
        raise ValueError("P1 outcome runner differs from the calibration runner")
    protocol = report["protocol"]
    expected = {
        "reconfiguration_horizon": args.reconfiguration_horizon,
        "recompiled_progress_horizon": args.progress_horizon,
        "action_block": args.action_block,
        "proposals": args.proposals,
        "proposal_distribution": args.proposal_distribution,
        "selection_metric": args.selection_metric,
        "damping": args.damping,
        "non_greedy_margin": args.non_greedy_margin,
        "max_non_greedy_margin": args.max_non_greedy_margin,
        "max_relative_reachability_improvement": args.max_relative_reachability_improvement,
        "min_second_to_first_singular_ratio": args.min_second_to_first_singular_ratio,
        "min_eligible_calibration_roots": args.min_eligible_calibration_roots,
        "warmup_policy": WARMUP_POLICY,
        "mechanism_observability": getattr(args, "mechanism_observability", "qpos_distance"),
        "target_delta": list(args.target_delta),
    }
    if any(
        (protocol.get(key, "qpos_distance") if key == "mechanism_observability" else protocol.get(key)) != value
        for key, value in expected.items()
    ):
        raise ValueError("P1 outcome configuration differs from calibration")
    if args.max_non_greedy_margin is not None and not protocol.get("margin_band_selection_binding"):
        raise ValueError("margin-bounded P1 outcome requires a calibration receipt bound to its selection evidence")
    calibration_seeds = {int(row["seed"]) for row in report.get("rows", [])}
    if len(calibration_seeds) < 2 or calibration_seeds & outcome_seeds:
        raise ValueError("P1 calibration and outcome roots must be disjoint, with at least two calibration roots")
    return {"path": str(path.resolve()), "sha256": _sha256_file(path), "calibration_root_seeds": sorted(calibration_seeds)}


def _base_distance(env: gym.Env, *, seed: int, context: dict[str, Any]) -> tuple[float, dict[str, str]]:
    root, goal = restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
    identity = p0._verify_execution_identity(root, goal, context["root_frame"], context["goal"])
    raw = env.unwrapped
    return p0._qpos_distance(np.asarray(raw.env.physics.data.qpos), np.asarray(raw.env.task.target_qpos)), identity


def _two_stage(
    *,
    name: str,
    first_program: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, dict[str, Any]]],
    env: gym.Env,
    seed: int,
    context: dict[str, Any],
    preprocessor: _Preprocessor,
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    lower: torch.Tensor,
    upper: torch.Tensor,
    action_block: int,
    progress_horizon: int,
    damping: float,
    device: torch.device,
) -> dict[str, Any]:
    """Execute a first chunk, observe its root, and compile the progress suffix."""

    pixels, goal, history = p0._prepare(
        preprocessor, context["frames"], context["root_frame"], context["goal"],
        np.asarray(context["history_action_blocks"], dtype=np.float32), device,
    )
    started = time.perf_counter()
    first, first_receipt = first_program(pixels, goal, history, seed)
    first_seconds = time.perf_counter() - started
    first_raw, first_normalized = p0._to_raw_actions(first, mean, std, lower, upper, env.action_space.low, env.action_space.high)
    zero_params = torch.zeros_like(first)
    zero_raw, zero_normalized = p0._to_raw_actions(
        zero_params, mean, std, lower, upper, env.action_space.low, env.action_space.high,
    )
    initial_distance, initial_identity = _base_distance(env, seed=seed, context=context)
    advanced = advance_executed_context(env, context=context, raw_action_block=first_raw, action_block=action_block)
    intermediate_raw = env.unwrapped
    intermediate_distance = p0._qpos_distance(
        np.asarray(intermediate_raw.env.physics.data.qpos), np.asarray(intermediate_raw.env.task.target_qpos),
    )
    # Restore the identical evaluator-only root before executing the physical
    # counterpart of proposal zero. This is a gate-only control; qpos never
    # enters the frozen candidate rank or suffix compiler.
    zero_root, zero_goal = restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
    zero_identity = p0._verify_execution_identity(zero_root, zero_goal, context["root_frame"], context["goal"])
    zero_advanced = advance_executed_context(env, context=context, raw_action_block=zero_raw, action_block=action_block)
    zero_raw_env = env.unwrapped
    zero_distance = p0._qpos_distance(
        np.asarray(zero_raw_env.env.physics.data.qpos), np.asarray(zero_raw_env.env.task.target_qpos),
    )
    successor_pixels, successor_goal, successor_history = p0._prepare(
        preprocessor, advanced["frames"], advanced["root_frame"], advanced["goal"],
        np.asarray(advanced["history_action_blocks"], dtype=np.float32), device,
    )
    zero_successor_pixels, _, zero_successor_history = p0._prepare(
        preprocessor, zero_advanced["frames"], zero_advanced["root_frame"], zero_advanced["goal"],
        np.asarray(zero_advanced["history_action_blocks"], dtype=np.float32), device,
    )
    goal_latent = p0._goal_latent(model, successor_goal)
    candidate_observed_latent = p0._latent_context(model, successor_pixels, successor_history)[0, -1]
    zero_observed_latent = p0._latent_context(model, zero_successor_pixels, zero_successor_history)[0, -1]
    candidate_observed_goal_cost = float((candidate_observed_latent - goal_latent).square().mean().detach())
    zero_observed_goal_cost = float((zero_observed_latent - goal_latent).square().mean().detach())
    started = time.perf_counter()
    suffix, suffix_receipt = p1._bounded_single_step_dls_program(
        model, successor_pixels, successor_goal, successor_history,
        future_horizon=progress_horizon, action_lower=lower, action_upper=upper,
        damping=damping,
    )
    suffix_seconds = time.perf_counter() - started
    suffix_raw, suffix_normalized = p0._to_raw_actions(suffix, mean, std, lower, upper, env.action_space.low, env.action_space.high)
    suffix_physical, successor_identity = p0._execute(
        env, seed, suffix_raw, expected_frame=advanced["root_frame"], expected_goal=advanced["goal"], execution_snapshot=advanced["execution_snapshot"],
    )
    if abs(float(suffix_physical["initial_qpos_distance"]) - float(intermediate_distance)) > 1.0e-8:
        raise RuntimeError(f"{name}: successor snapshot qpos does not match observed reconfiguration root")
    return {
        "action_witness_raw": np.concatenate([first_raw, suffix_raw], axis=0).tolist(),
        "action_witness_normalized": np.concatenate([first_normalized, suffix_normalized], axis=0).tolist(),
        "first_action_witness_raw": first_raw.tolist(),
        "first_action_witness_normalized": first_normalized.tolist(),
        "zero_prefix_action_witness_raw": zero_raw.tolist(),
        "zero_prefix_action_witness_normalized": zero_normalized.tolist(),
        "first_receipt": first_receipt,
        "suffix_receipt": suffix_receipt,
        "planner_wall_seconds": first_seconds + suffix_seconds,
        "execution_identity": {"initial": initial_identity, "zero_prefix": zero_identity, "successor": successor_identity},
        "observed_successor_context": {key: advanced[key] for key in ("context_mode", "root_rgb_sha256", "goal_rgb_sha256", "advanced_action_sha256")},
        "zero_prefix_successor_context": {key: zero_advanced[key] for key in ("context_mode", "root_rgb_sha256", "goal_rgb_sha256", "advanced_action_sha256")},
        "observed_first_chunk": {
            "coordinate": "frozen_encoder_goal_cost_on_actual_successor_rgb",
            "candidate_encoded_goal_cost": candidate_observed_goal_cost,
            "zero_prefix_encoded_goal_cost": zero_observed_goal_cost,
        },
        "physical": {
            "initial_qpos_distance": initial_distance,
            "after_first_chunk_qpos_distance": intermediate_distance,
            "after_zero_prefix_qpos_distance": zero_distance,
            "final_qpos_distance": suffix_physical["final_qpos_distance"],
            "min_qpos_distance_after_first_chunk": suffix_physical["min_qpos_distance"],
            "physical_progress": initial_distance - suffix_physical["final_qpos_distance"],
        },
    }


def _paired(rows: list[dict[str, Any]], method: str, reference: str, seed: int) -> dict[str, Any]:
    delta = np.asarray([
        row["methods"][method]["physical"]["final_qpos_distance"] - row["methods"][reference]["physical"]["final_qpos_distance"]
        for row in rows
    ], dtype=np.float64)
    return {
        "metric": f"final_qpos_distance: {method} minus {reference}; negative favors {method}",
        "n": int(delta.size), "mean": float(delta.mean()), "ci95": p0._bootstrap_mean(delta, seed),
        "method_only_wins": int(np.sum(delta < 0.0)), "reference_only_wins": int(np.sum(delta > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p0-evidence", type=Path, required=True)
    parser.add_argument("--p1-preflight-evidence", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm"))
    parser.add_argument("--adapter-artifact", type=Path, default=Path("results/live_cross_environment_prediction_20261015/reacher_cuda/cross_environment_prediction.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20262800)
    parser.add_argument("--roots", type=int, default=2)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--outcome-only", action="store_true")
    parser.add_argument("--calibration-evidence", type=Path)
    parser.add_argument(
        "--margin-band-selection-evidence", type=Path,
        help="Immutable calibration-only selection receipt required for a bounded non-greedy margin band.",
    )
    parser.add_argument(
        "--training-source-manifest", type=Path, action="append", default=[],
        help="Counterfactual simulator pool or repair manifest; its reset seeds are forbidden for this panel.",
    )
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--reconfiguration-horizon", type=int, default=1)
    parser.add_argument("--progress-horizon", type=int, default=2)
    parser.add_argument("--proposals", type=int, default=p1.DEFAULT_RECONFIGURATION_PROPOSALS)
    parser.add_argument(
        "--proposal-distribution", choices=p1.RECONFIGURATION_PROPOSAL_DISTRIBUTIONS,
        default=p1.DEFAULT_RECONFIGURATION_PROPOSAL_DISTRIBUTION,
        help="Frozen reconfiguration-bank measure; defaults to physical uniform actions.",
    )
    parser.add_argument(
        "--selection-metric", choices=p1.RECONFIGURATION_SELECTION_METRICS,
        default=p1.DEFAULT_RECONFIGURATION_SELECTION_METRIC,
        help="Frozen future prefix score; bounded DLS cost is the default physical-action-aware metric.",
    )
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument(
        "--non-greedy-margin", type=float, default=1.0e-6,
        help=(
            "Minimum frozen immediate-goal-cost increase over the zero prefix. "
            "This is a calibration-selected, model-only admissibility threshold; "
            "it is never selected from outcome-root qpos."
        ),
    )
    parser.add_argument(
        "--max-non-greedy-margin", type=float, default=None,
        help=(
            "Optional upper bound on frozen immediate-goal-cost increase. "
            "Together with --non-greedy-margin this defines a model-only "
            "calibration-frozen admissibility band."
        ),
    )
    parser.add_argument(
        "--max-relative-reachability-improvement", type=float, default=None,
        help="Optional calibration-frozen upper guard on model-only relative V_reach improvement.",
    )
    parser.add_argument(
        "--min-second-to-first-singular-ratio", type=float, default=None,
        help="Optional calibration-frozen lower guard on post-prefix Jacobian spectral balance.",
    )
    parser.add_argument("--cem-samples", type=int, default=64)
    parser.add_argument("--cem-steps", type=int, default=8)
    parser.add_argument("--cem-elite", type=int, default=8)
    parser.add_argument(
        "--min-eligible-calibration-roots", type=int, default=4,
        help="Calibration must physically validate every eligible prefix across at least this many roots.",
    )
    parser.add_argument("--min-eligible-outcome-roots", type=int, default=12)
    parser.add_argument(
        "--mechanism-observability", choices=MECHANISM_OBSERVABILITIES, default="qpos_distance",
        help=(
            "Calibration-only first-chunk non-greediness coordinate. qpos and observed RGB-encoder "
            "criteria are separate protocol branches; neither affects model-side ranking."
        ),
    )
    parser.add_argument("--target-delta", type=float, nargs=2, default=(0.65, -0.65))
    args = parser.parse_args()
    _resolve_local_cli_inputs(args)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable P1 output: {args.output}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Reacher D2P-P1 requires host CUDA exactly; CPU fallback is forbidden")
    if min(args.roots, args.action_block, args.reconfiguration_horizon, args.progress_horizon, args.proposals, args.cem_samples, args.cem_steps, args.cem_elite) < 1 or args.proposals < 2 or args.cem_elite >= args.cem_samples:
        raise ValueError("invalid P1 protocol dimensions")
    if args.calibration_only and args.outcome_only:
        raise ValueError("calibration-only and outcome-only are mutually exclusive")
    if args.margin_band_selection_evidence is not None and not args.calibration_only:
        raise ValueError("margin-band selection evidence is accepted only for a fresh calibration-only run")
    if args.calibration_only and args.roots < args.min_eligible_calibration_roots:
        raise ValueError("calibration-only requires at least the declared number of calibration roots")
    if args.outcome_only and (args.roots < args.min_eligible_outcome_roots or args.calibration_evidence is None):
        raise ValueError("outcome-only requires a calibration receipt and at least the declared minimum number of fresh roots")
    if args.min_eligible_calibration_roots < 2 or args.min_eligible_outcome_roots < 2:
        raise ValueError("P1 requires at least two eligible roots in every analysis split")
    if args.damping <= 0.0 or args.non_greedy_margin < 0.0:
        raise ValueError("P1 damping must be positive and non-greedy margin non-negative")
    if args.max_non_greedy_margin is not None and args.max_non_greedy_margin < args.non_greedy_margin:
        raise ValueError("P1 maximum non-greedy margin must not be below its minimum")
    if args.max_relative_reachability_improvement is not None and not 0.0 < args.max_relative_reachability_improvement <= 1.0:
        raise ValueError("P1 maximum relative reachability improvement must lie in (0, 1]")
    if args.min_second_to_first_singular_ratio is not None and not 0.0 < args.min_second_to_first_singular_ratio <= 1.0:
        raise ValueError("P1 minimum second-to-first singular ratio must lie in (0, 1]")
    p0_evidence = p1._validate_upstream_p0(args.p0_evidence)
    preflight_evidence = _validate_preflight(args.p1_preflight_evidence, args=args)
    outcome_seed_set = {args.seed + offset for offset in range(args.roots)}
    training_source_seed_binding = _bind_simulator_training_sources(
        args.training_source_manifest, run_seeds=outcome_seed_set,
    )
    calibration_binding = _validate_outcome_calibration(
        args.calibration_evidence, args=args, outcome_seeds=outcome_seed_set,
    ) if args.outcome_only else None
    margin_band_selection_binding = _validate_margin_band_selection(
        args.margin_band_selection_evidence, args=args,
    ) if args.margin_band_selection_evidence is not None else None
    _reserve_immutable_output(args.output)
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
            seed = args.seed + offset
            context = collect_executed_context(env, seed=seed, history_len=3, action_block=args.action_block)

            def reconfiguration_first(pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor, root_seed: int):
                params, receipt = p1._compile_reconfiguration(
                    model, pixels, goal, history, reconfiguration_horizon=args.reconfiguration_horizon,
                    future_horizon=args.progress_horizon, action_lower=lower, action_upper=upper,
                    damping=args.damping, proposals=args.proposals, non_greedy_margin=args.non_greedy_margin,
                    max_non_greedy_margin=args.max_non_greedy_margin,
                    max_relative_reachability_improvement=args.max_relative_reachability_improvement,
                    min_second_to_first_singular_ratio=args.min_second_to_first_singular_ratio,
                    reachability_margin=1.0e-6, seed=root_seed,
                    proposal_distribution=args.proposal_distribution,
                    selection_metric=args.selection_metric,
                )
                if params is None:
                    # Eligibility is determined solely by frozen RGB dynamics
                    # before physical execution.  An ineligible root still
                    # receives the same one-token neutral prefix and observed
                    # successor recompile so the runner has no root-dependent
                    # execution budget.  It is excluded from the primary
                    # reconfiguration mechanism comparison, never selected by
                    # qpos or another physical outcome.
                    neutral = torch.zeros(
                        (1, args.reconfiguration_horizon, history.shape[-1]),
                        device=history.device, dtype=history.dtype,
                    )
                    receipt = dict(receipt)
                    receipt["fallback_neutral_prefix"] = True
                    return neutral, receipt
                receipt = dict(receipt)
                receipt["fallback_neutral_prefix"] = False
                return params, receipt

            def greedy_first(pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor, root_seed: int):
                return p0._d2p_program(
                    model, pixels, goal, history, future_horizon=args.reconfiguration_horizon,
                    action_lower=lower, action_upper=upper, damping=args.damping,
                    relinearization_steps=p0.DEFAULT_RELINEARIZATION_STEPS, restarts=1,
                    initialization="screened_gaussian_proposals", proposal_samples=128, proposal_elites=4, seed=root_seed + 3000,
                )

            def zero_first(pixels: torch.Tensor, goal: torch.Tensor, history: torch.Tensor, root_seed: int):
                """Physical no-reconfiguration control with the identical suffix compiler."""

                del pixels, goal, root_seed
                zero = torch.zeros(
                    (1, args.reconfiguration_horizon, history.shape[-1]),
                    device=history.device, dtype=history.dtype,
                )
                return zero, {
                    "program_type": "physical_zero_prefix_control",
                    "selection_rule": "fixed model-zero prefix; suffix recompiles only from its observed successor",
                    "forward_rollout_evaluations": 0,
                    "backward_vjp_rows": 0,
                }

            recon = _two_stage(name="reconfiguration", first_program=reconfiguration_first, env=env, seed=seed, context=context, preprocessor=preprocessor, model=model, mean=mean, std=std, lower=lower, upper=upper, action_block=args.action_block, progress_horizon=args.progress_horizon, damping=args.damping, device=device)
            zero = _two_stage(name="zero_prefix_two_stage_d2p", first_program=zero_first, env=env, seed=seed, context=context, preprocessor=preprocessor, model=model, mean=mean, std=std, lower=lower, upper=upper, action_block=args.action_block, progress_horizon=args.progress_horizon, damping=args.damping, device=device)
            greedy = _two_stage(name="greedy_two_stage_d2p", first_program=greedy_first, env=env, seed=seed, context=context, preprocessor=preprocessor, model=model, mean=mean, std=std, lower=lower, upper=upper, action_block=args.action_block, progress_horizon=args.progress_horizon, damping=args.damping, device=device)
            pixels, goal, history = p0._prepare(preprocessor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], dtype=np.float32), device)
            started_cem = time.perf_counter()
            cem_params, cem_receipt = p0._cem(model, pixels, goal, history, future_horizon=args.reconfiguration_horizon + args.progress_horizon, action_lower=lower, action_upper=upper, samples=args.cem_samples, steps=args.cem_steps, elite=args.cem_elite, seed=seed + 6000)
            cem_receipt["planner_wall_seconds"] = time.perf_counter() - started_cem
            cem_raw, cem_normalized = p0._to_raw_actions(cem_params, mean, std, lower, upper, env.action_space.low, env.action_space.high)
            cem_physical, cem_identity = p0._execute(env, seed, cem_raw, expected_frame=context["root_frame"], expected_goal=context["goal"], execution_snapshot=context["execution_snapshot"])
            rows.append({"seed": seed, "split": "calibration" if args.calibration_only else "outcome", "executed_context": {key: context[key] for key in ("context_mode", "warmup_policy", "root_rgb_sha256", "goal_rgb_sha256", "warmup_actions_sha256")}, "methods": {"reconfiguration": recon, "zero_prefix_two_stage_d2p": zero, "greedy_two_stage_d2p": greedy, "cem_full_horizon": {"action_witness_raw": cem_raw.tolist(), "action_witness_normalized": cem_normalized.tolist(), "receipt": cem_receipt, "execution_identity": cem_identity, "physical": cem_physical}}})
            print(json.dumps({"seed": seed, "reconfiguration_final_qpos": recon["physical"]["final_qpos_distance"], "zero_prefix_final_qpos": zero["physical"]["final_qpos_distance"], "greedy_final_qpos": greedy["physical"]["final_qpos_distance"], "cem_final_qpos": cem_physical["final_qpos_distance"]}, sort_keys=True))
    finally:
        env.close()
    hash_after = _parameter_hash(model)
    eligible_rows = [
        row for row in rows
        if row["methods"]["reconfiguration"]["first_receipt"]["non_greedy_witness_found"]
    ]
    comparisons = {} if args.calibration_only or len(eligible_rows) < 2 else {
        name: _paired(eligible_rows, "reconfiguration", name, args.seed + index)
        for index, name in enumerate(("zero_prefix_two_stage_d2p", "greedy_two_stage_d2p", "cem_full_horizon"), start=1)
    }
    all_methods = [method for row in rows for method in row["methods"].values()]
    mechanism_non_greedy = [
        row for row in eligible_rows
        if _mechanism_non_greedy(
            row["methods"]["reconfiguration"], observability=args.mechanism_observability,
        )
    ]
    mechanism_gate = _mechanism_gate_name(args.mechanism_observability)
    # Keep the compact legacy report construction below readable; the emitted
    # receipt immediately renames this field to the actual mechanism branch.
    physical_non_greedy = mechanism_non_greedy
    gates = {
        "cuda_only": True,
        "upstream_p0_admission_verified": p0_evidence["p1_unlocked"],
        "p1_plumbing_preflight_verified": True,
        "frozen_parameter_hash_unchanged": hash_before == hash_after,
        "at_least_one_model_eligible_reconfiguration_root": bool(eligible_rows),
        "at_least_declared_eligible_calibration_roots": (
            not args.calibration_only or len(eligible_rows) >= args.min_eligible_calibration_roots
        ),
        "every_eligible_reconfiguration_is_model_non_greedy": all(row["methods"]["reconfiguration"]["first_receipt"]["non_greedy_model_margin"] > 0.0 for row in eligible_rows),
        "every_eligible_reconfiguration_improves_declared_future_score": all(
            p1._declared_future_score_improvement(
                row["methods"]["reconfiguration"]["first_receipt"], args.selection_metric,
            ) > 0.0
            for row in eligible_rows
        ),
        mechanism_gate: len(mechanism_non_greedy) == len(eligible_rows),
        "every_ineligible_root_uses_declared_neutral_prefix": all(
            method["first_receipt"].get("fallback_neutral_prefix") is (not method["first_receipt"]["non_greedy_witness_found"])
            for row in rows for method in (row["methods"]["reconfiguration"],)
        ),
        "all_methods_execute_exactly_three_action_tokens": all(len(method["action_witness_raw"]) == 3 * args.action_block for method in all_methods),
        "all_raw_actions_within_environment_box": all(np.all(np.asarray(method["action_witness_raw"]) >= np.asarray(env.action_space.low) - 1.0e-6) and np.all(np.asarray(method["action_witness_raw"]) <= np.asarray(env.action_space.high) + 1.0e-6) for method in all_methods),
        "all_physical_outcomes_finite": all(np.isfinite(value) for method in all_methods for value in method["physical"].values()),
        "all_execution_root_identities_verified": all(bool(method["execution_identity"]) for method in all_methods),
    }
    p2_unlocked = bool(
        not args.calibration_only and len(eligible_rows) >= args.min_eligible_outcome_roots and comparisons
        and all(float(item["ci95"][1]) <= 0.0 for item in comparisons.values()) and all(gates.values())
    )
    report = {"schema": "d2p_reacher_p1_reconfiguration.v1", "status": "calibration_only" if args.calibration_only else "completed_without_efficacy_verdict", "scope": "fresh root-paired physical P1; P1 efficacy requires a separate root-disjoint outcome receipt", "cuda_host": {"name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version()}, "protocol": {"calibration_only": bool(args.calibration_only), "outcome_only": bool(args.outcome_only), "reconfiguration_horizon": args.reconfiguration_horizon, "recompiled_progress_horizon": args.progress_horizon, "total_executed_action_tokens": args.reconfiguration_horizon + args.progress_horizon, "action_block": args.action_block, "checkpoint_action_block": checkpoint_block, "warmup_policy": WARMUP_POLICY, "target_delta": list(args.target_delta), "proposals": args.proposals, "proposal_distribution": args.proposal_distribution, "selection_metric": args.selection_metric, "damping": args.damping, "non_greedy_margin": args.non_greedy_margin, "max_non_greedy_margin": args.max_non_greedy_margin, "min_eligible_calibration_roots": args.min_eligible_calibration_roots, "min_eligible_outcome_roots": args.min_eligible_outcome_roots, "methods": ["reconfiguration", "zero_prefix_two_stage_d2p", "greedy_two_stage_d2p", "cem_full_horizon"], "runtime_input": "RGB/action history and rendered goal only; qpos is evaluator-only", "reconfiguration_selection": "bounded nonlinear DLS suffix terminal cost plus a calibration-frozen non-greedy immediate-cost margin band", "physical_mechanism_requirement": "every model-eligible first chunk must be no more goal-greedy than the same-length physical zero-prefix control before its later qpos outcome can support a reconfiguration claim", "eligibility": "a root is primary-analysis eligible only when frozen RGB dynamics produce a non-greedy reachability witness before physical execution; otherwise it executes an explicit neutral-prefix fallback with the same token budget", "calibration_binding": calibration_binding}, "evidence": {"p0": p0_evidence, "p1_preflight": preflight_evidence}, "inputs": {"checkpoint": str(args.policy.resolve()), "weights_sha256": _sha256_file(args.policy / "weights.pt"), "adapter_artifact": str(args.adapter_artifact.resolve()), "adapter_sha256": _sha256_file(args.adapter_artifact), "runner_sha256": _sha256_file(Path(__file__))}, "parameter_hash_before": hash_before, "parameter_hash_after": hash_after, "gates": gates, "eligibility": {"eligible_root_count": len(eligible_rows), "physically_non_greedy_eligible_root_count": len(physical_non_greedy), "ineligible_root_count": len(rows) - len(eligible_rows), "eligible_root_seeds": [row["seed"] for row in eligible_rows], "minimum_required_for_calibration": args.min_eligible_calibration_roots, "minimum_required_for_outcome": args.min_eligible_outcome_roots}, "comparisons_outcome_only": comparisons, "stage_gate": {"criterion": "P2 opens only if a root-disjoint P1 outcome panel has the predeclared minimum number of model-eligible roots, every model-eligible first chunk is no more goal-greedy than its same-length physical zero-prefix control, and reconfiguration-minus-zero-prefix, reconfiguration-minus-greedy, and reconfiguration-minus-CEM paired 95% intervals all favor reconfiguration", "p2_unlocked": p2_unlocked}, "rows": rows, "elapsed_seconds": time.time() - started, "claim_boundary": "This runner makes a physical P1 comparison possible. Calibration receipts have no efficacy claim; an outcome receipt needs root-disjoint calibration binding before any mechanism conclusion."}
    report["protocol"]["max_relative_reachability_improvement"] = args.max_relative_reachability_improvement
    report["protocol"]["min_second_to_first_singular_ratio"] = args.min_second_to_first_singular_ratio
    report["protocol"]["mechanism_observability"] = args.mechanism_observability
    report["protocol"]["margin_band_selection_binding"] = margin_band_selection_binding
    report["inputs"]["training_source_seed_binding"] = training_source_seed_binding
    report["protocol"].pop("physical_mechanism_requirement")
    report["protocol"]["mechanism_requirement"] = (
        "every model-eligible first chunk must be no more goal-greedy than the same-length zero-prefix "
        f"control under the preregistered {args.mechanism_observability} coordinate before a later outcome claim"
    )
    report["eligibility"]["mechanism_non_greedy_eligible_root_count"] = report["eligibility"].pop(
        "physically_non_greedy_eligible_root_count"
    )
    report["stage_gate"]["criterion"] = (
        "P2 opens only if a root-disjoint P1 outcome panel has the predeclared minimum number of model-eligible roots, "
        "every model-eligible first chunk is no more goal-greedy than its same-length zero-prefix control under "
        f"{args.mechanism_observability}, and reconfiguration-minus-zero-prefix, reconfiguration-minus-greedy, and "
        "reconfiguration-minus-CEM paired 95% intervals all favor reconfiguration"
    )
    report["protocol"]["reconfiguration_selection"] = (
        f"{args.selection_metric} plus calibration-frozen immediate-cost, relative-reachability-improvement, "
        "and spectral-balance bands"
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "comparisons": comparisons, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
