"""Export one immutable record for real-checkpoint planning selection.

The checkpoint is never updated.  Simulator state is used only to create the
offline diagnostic split and score the held-out evaluator; at runtime the
planner receives rendered RGB observations and a goal image.  The resulting
record is consumed by ``run_real_checkpoint_selection_benchmark.py``.

This runner currently instantiates PushT because that is the only local
intervention environment with a compatible visual model.  It is intentionally
environment-scoped; cross-environment generality requires records produced by
additional environment adapters under the same schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import scipy.linalg
import stable_worldmodel  # noqa: F401: registers PushT
import torch

try:
    from .task_metric_diagnostics import metric_geometry, pusht_task_metric, raw_chart_jacobian, task_metric_metadata
except ImportError:  # direct script execution
    from task_metric_diagnostics import metric_geometry, pusht_task_metric, raw_chart_jacobian, task_metric_metadata  # type: ignore[no-redef]

try:
    from .evaluate_causal_visual_checkpoint import checkpoint_adapter_architecture, checkpoint_utility_mode
    from .run_causal_visual_pusht_world_model import (
        PUSHT_RENDER_SIZE,
        Normalization,
        VisualCausalWorldModel,
        balanced_source_cases,
        evaluate_visual_planning,
        make_counterfactual_dataset,
        observable_pose_vector,
        raw_from_normalized,
        reset_with_case,
        tensor_images,
        visual_plan_action,
    )


except ImportError:  # direct script execution
    from evaluate_causal_visual_checkpoint import checkpoint_adapter_architecture, checkpoint_utility_mode  # type: ignore[no-redef]
    from run_causal_visual_pusht_world_model import (  # type: ignore[no-redef]
        PUSHT_RENDER_SIZE,
        Normalization,
        VisualCausalWorldModel,
        balanced_source_cases,
        evaluate_visual_planning,
        make_counterfactual_dataset,
        observable_pose_vector,
        raw_from_normalized,
        reset_with_case,
        tensor_images,
        visual_plan_action,
    )


# These variants are frozen before any held-out planning result is read.  The
# small epsilon entries preserve a positive-definite metric while making the
# named component effectively absent; this avoids silently switching to a
# singular pseudo-metric with a different invariance definition.
PRE_REGISTERED_TASK_METRIC_VARIANTS = {
    "default_task_space": (0.25, 1.0, 4.0),
    "agent_weight_low": (0.05, 1.0, 4.0),
    "agent_weight_high": (1.0, 1.0, 4.0),
    "block_xy_only": (1e-6, 1.0, 1e-6),
    "orientation_only": (1e-6, 1e-6, 4.0),
    "block_xy_plus_orientation": (1e-6, 1.0, 4.0),
    # Follow-up controls for the condition-number confound in the near-zero
    # component ablations.  The floor is fixed before rerunning the follow-up
    # and is not selected from held-out planning.
    "stable_block_xy_focus": (0.05, 1.0, 0.05),
    "stable_orientation_focus": (0.05, 0.05, 4.0),
    "stable_block_xy_plus_orientation": (0.05, 1.0, 4.0),
}


def _action_effect_metric(data: dict[str, np.ndarray], *, ridge: float = 1e-3) -> np.ndarray:
    """Build a non-degenerate external metric from diagnostic interventions."""
    effects = np.asarray(data["next_state"], dtype=np.float64) - np.asarray(data["state"], dtype=np.float64)
    actions = np.asarray(data["action"], dtype=np.float64)
    if effects.ndim != 2 or actions.ndim != 2 or effects.shape[0] != actions.shape[0]:
        raise ValueError("action-effect metric requires aligned state effects and actions")
    # Fit only on the diagnostic interventions; the action-to-effect Gram
    # matrix is external to latent covariance and remains positive definite
    # after a fixed trace-scaled ridge.
    design = np.column_stack([actions, np.ones(actions.shape[0], dtype=np.float64)])
    coefficients = np.linalg.lstsq(design, effects, rcond=None)[0]
    linear_effect = coefficients[: actions.shape[1]].T
    metric = linear_effect @ linear_effect.T
    metric = 0.5 * (metric + metric.T)
    trace_scale = max(float(np.trace(metric)) / max(metric.shape[0], 1), 1e-12)
    metric = metric / trace_scale
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("action-effect metric ridge must be positive and finite")
    metric += float(ridge) * np.eye(metric.shape[0], dtype=np.float64)
    return metric


def _task_metric_variants(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    variants = {
        name: pusht_task_metric(
            agent_weight=weights[0],
            block_position_weight=weights[1],
            orientation_weight=weights[2],
        )
        for name, weights in PRE_REGISTERED_TASK_METRIC_VARIANTS.items()
    }
    variants["action_effect"] = _action_effect_metric(data)
    variants["stable_action_effect"] = _action_effect_metric(data, ridge=0.05)
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--group", required=True, help="e.g. PushT/seed-0/horizon-6")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diagnostic-roots", type=int, default=240)
    parser.add_argument("--diagnostic-root-rollout-steps", type=int, default=8)
    parser.add_argument("--diagnostic-seed", type=int, default=74001)
    parser.add_argument("--planning-episodes", type=int, default=24)
    parser.add_argument("--planning-steps", type=int, default=100)
    parser.add_argument("--planning-horizon", type=int, default=6)
    parser.add_argument("--cem-samples", type=int, default=192)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--contact-servo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direct-utility", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task-space-objective", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_checkpoint(path: Path, device: str) -> tuple[VisualCausalWorldModel, Normalization, SimpleNamespace]:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved = payload["args"]
    utility_mode = checkpoint_utility_mode(payload)
    adapter_architecture = checkpoint_adapter_architecture(payload)
    model = VisualCausalWorldModel(
        int(saved["latent_dim"]), str(saved["vision_encoder"]), bool(saved["temporal_context"]), utility_mode,
        adapter_architecture, bool(saved.get("structured_readout", False)),
        bool(saved.get("contact_frame_dynamics", False)),
        bool(saved.get("mode_conditioned_dynamics", False)),
        bool(saved.get("current_only_readout", False)),
        bool(saved.get("torque_features", False)),
        bool(saved.get("temporal_dynamics", False)),
        bool(saved.get("previous_action_dynamics", False)),
        bool(saved.get("explicit_angle_effect", False)),
        bool(saved.get("explicit_angle_readout", False)),
        bool(saved.get("explicit_goal_angle_readout", False)),
        bool(saved.get("rich_temporal_dynamics", False)),
        invariant_angle_readout=bool(saved.get("invariant_angle_readout", False)),
        contact_interface_dynamics=bool(saved.get("contact_interface_dynamics", False)),
        angle_readout_current_only=bool(saved.get("angle_readout_current_only", False)),
    ).to(device)
    model.load_state_dict(payload["model"])
    normalizer = Normalization(**payload["normalization"])
    mean, std = normalizer.tensors(device)
    model.set_state_normalization(mean, std)
    runtime = dict(saved)
    # New checkpoints persist these fields; old ones infer them from immutable
    # state-dict structure.  Assignment (rather than duplicate kwargs) makes
    # both contracts load through the same path.
    runtime["utility_mode"] = utility_mode
    runtime["adapter_architecture"] = adapter_architecture
    return model.eval(), normalizer, SimpleNamespace(**runtime)


def _operator_diagnostics(
    model: VisualCausalWorldModel, data: dict[str, np.ndarray], normalizer: Normalization, device: str,
    transient_horizon: int,
) -> dict[str, float]:
    """Evaluate covariance--predictor geometry in a fixed raw pose chart."""
    mean, std = normalizer.tensors(device)
    image = tensor_images(data["image"], device)
    previous = tensor_images(data["previous_image"], device)
    action = torch.as_tensor(data["action"], dtype=torch.float32, device=device)
    with torch.inference_mode():
        decoded, predicted = model(image, action, previous)
        decoded_raw = (decoded * std + mean).detach()
        predicted_raw = (predicted * std + mean).detach()
    # The average local Jacobian is a *diagnostic summary*, not a global
    # linearization.  It is computed in the decoded, raw observable-pose chart
    # to hold the coordinate gauge fixed across checkpoints.
    sample_count = min(128, decoded.shape[0])
    state = decoded[:sample_count].detach().clone().requires_grad_(True)
    zero_action = torch.zeros(sample_count, 2, dtype=state.dtype, device=device)
    successor = model.predict_next(state, zero_action)
    rows = []
    for output_dimension in range(successor.shape[1]):
        gradient = torch.autograd.grad(successor[:, output_dimension].sum(), state, retain_graph=True)[0]
        rows.append(gradient.mean(dim=0))
    normalized_jacobian = torch.stack(rows).detach().cpu().numpy().astype(np.float64)
    # ``state`` and ``successor`` are normalized model coordinates.  The
    # covariance below is intentionally in the raw observable-pose chart, so
    # compare it with the chain-rule transformed Jacobian rather than mixing
    # two coordinate systems.  This also makes the declared task metric
    # meaningful and auditable.
    jacobian = raw_chart_jacobian(normalized_jacobian, std.detach().cpu().numpy())
    states = decoded_raw.cpu().numpy().astype(np.float64)
    covariance = np.cov(states, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    task_metrics = _task_metric_variants(data)
    task_geometry = metric_geometry(
        covariance, jacobian, task_metrics["default_task_space"],
        rank=min(3, covariance.shape[0]), horizons=range(1, int(transient_horizon) + 1),
    )
    task_metric_variants = {
        name: metric_geometry(
            covariance, jacobian, metric,
            rank=min(3, covariance.shape[0]), horizons=range(1, int(transient_horizon) + 1),
        )
        for name, metric in task_metrics.items()
    }
    commutator = covariance @ jacobian - jacobian @ covariance
    denominator = max(float(np.linalg.norm(covariance, "fro") * np.linalg.norm(jacobian, "fro")), 1e-12)
    # scipy returns (T, Z); keep the triangular factor distinct from the
    # orthogonal Schur basis in the exported diagnostic definition.
    triangular, _basis = scipy.linalg.schur(jacobian, output="real")
    upper = np.triu(triangular, k=1)
    rank = min(3, covariance.shape[0])
    covariance_vectors = np.linalg.eigh(covariance)[1][:, -rank:]
    predictor_vectors = np.linalg.svd(jacobian, full_matrices=False)[0][:, :rank]
    principal_cosines = np.linalg.svd(covariance_vectors.T @ predictor_vectors, compute_uv=False)
    alignment = float(np.sqrt(max(0.0, 1.0 - float(np.mean(np.square(principal_cosines))))))
    transient = max(
        float(np.linalg.svd(np.linalg.matrix_power(jacobian, horizon), compute_uv=False)[0])
        for horizon in range(1, int(transient_horizon) + 1)
    )
    prediction_mse = float(np.mean(np.square(predicted_raw.cpu().numpy() - data["next_state"])))
    # Resolvent norm on a fixed unit-circle grid is a finite, reproducible
    # pseudospectral-sensitivity proxy.  It does not claim a full continuum
    # pseudospectrum calculation.
    grid = np.exp(1j * np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False))
    identity = np.eye(jacobian.shape[0])
    pseudospectral = max(
        1.0 / max(float(np.linalg.svd(value * identity - jacobian, compute_uv=False)[-1]), 1e-8)
        for value in grid
    )
    predictor_spectrum = np.linalg.eigvals(jacobian)
    predictor_spectrum = predictor_spectrum[np.lexsort((predictor_spectrum.imag, predictor_spectrum.real))]
    return {
        # This one-step counterfactual number is retained as a descriptive
        # diagnostic.  It is not the quality criterion used for v2 model
        # selection because its action law is not the controller's action law.
        "one_step_counterfactual_mse": prediction_mse,
        "alignment": alignment,
        "normalized_commutator": float(np.linalg.norm(commutator, "fro") / denominator),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(jacobian)))),
        "latent_variance": float(np.trace(covariance) / covariance.shape[0]),
        "schur_offdiagonal_energy": float(np.linalg.norm(upper, "fro") / max(np.linalg.norm(triangular, "fro"), 1e-12)),
        "departure_from_normality": float(
            np.linalg.norm(jacobian.T @ jacobian - jacobian @ jacobian.T, "fro")
            / max(np.linalg.norm(jacobian, "fro") ** 2, 1e-12)
        ),
        "transient_amplification": transient,
        "pseudospectral_sensitivity": float(pseudospectral),
        # Keep the full local-Jacobian spectrum as an audit field.  Scalar
        # spectral_radius is insufficient for a fixed-spectrum control.
        "predictor_spectrum": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in predictor_spectrum
        ],
        # These are deliberately separate from the legacy Euclidean fields.
        # The task metric is external and fixed before held-out planning.
        **task_geometry,
        "task_metric_variants": task_metric_variants,
        "task_metric_variant_metadata": {
            name: task_metric_metadata(metric) for name, metric in task_metrics.items()
        },
    }


def _root_hash(roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]]) -> str:
    """Hash diagnostic source states without serialising rendered images."""
    digest = hashlib.sha256()
    for state, goal, *_rest in roots:
        digest.update(np.asarray(state, dtype=np.float32).tobytes())
        digest.update(np.asarray(goal, dtype=np.float32).tobytes())
    return digest.hexdigest()


def planner_matched_rollout_mse(
    env: gym.Env,
    model: VisualCausalWorldModel,
    normalizer: Normalization,
    runtime: SimpleNamespace,
    roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]],
    horizon: int,
    action_repeat: int,
    seed: int,
) -> float:
    """Evaluate open-loop predictor error under the frozen planner's actions.

    The diagnostic roots are disjoint from planning test episodes.  At each
    diagnostic root the same RGB-only controller that is used at test time
    produces the action sequence.  We then roll the immutable learned
    predictor open-loop over that realised sequence and compare it with the
    simulator trajectory.  This makes the prediction-quality selector match
    both the controller's action distribution and its effective model horizon.
    Simulator state remains an offline scoring target only.
    """
    if horizon < 1:
        raise ValueError("planner-matched diagnostic horizon must be positive")
    mean, std = normalizer.tensors(runtime.device)
    generator = torch.Generator(device=runtime.device).manual_seed(seed)
    squared_errors: list[float] = []
    # ``balanced_source_cases`` carries previous_action and contact-root
    # metadata in addition to the images required by this open-loop audit.
    # Keep this consumer forward-compatible with that immutable root schema;
    # neither metadata field is used by the visual runtime scorer here.
    for state, goal, image, previous_image, goal_image, *_metadata in roots:
        runtime._orientation_phase_latched = False
        observation, _ = reset_with_case(env, state, goal)
        current_image = np.asarray(image, dtype=np.uint8).copy()
        previous = np.asarray(previous_image, dtype=np.uint8).copy()
        goal_render = np.asarray(goal_image, dtype=np.uint8).copy()
        with torch.inference_mode():
            predicted = model.encode_state(
                tensor_images(current_image[None], runtime.device),
                tensor_images(previous[None], runtime.device),
            )
        for _step in range(horizon):
            action = visual_plan_action(model, current_image, previous, goal_render, normalizer, runtime, generator)
            terminated = truncated = False
            for _ in range(action_repeat):
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            with torch.inference_mode():
                action_tensor = torch.as_tensor(action[None], dtype=torch.float32, device=runtime.device)
                predicted = model.predict_next(predicted, action_tensor)
                prediction_raw = raw_from_normalized(predicted, mean, std).squeeze(0).cpu().numpy()
            truth = observable_pose_vector(np.asarray(observation["state"], dtype=np.float32))
            squared_errors.append(float(np.mean(np.square(prediction_raw - truth))))
            if terminated or truncated:
                break
            previous = current_image
            current_image = env.render().copy()
    if not squared_errors:
        raise RuntimeError("planner-matched diagnostic produced no valid transitions")
    return float(np.mean(squared_errors))


def planning_loss(episodes: list[dict[str, Any]]) -> tuple[float, dict[str, float | int]]:
    success = np.asarray([row["strict_success"] for row in episodes], dtype=np.float64)
    xy = np.asarray([row["block_xy_distance"] for row in episodes], dtype=np.float64)
    angle = np.asarray([row["angle_error"] for row in episodes], dtype=np.float64)
    # This scalar is pre-declared solely to make selection regret comparable:
    # strict failure dominates, while geometric residuals break failures ties.
    loss = float((1.0 - success.mean()) + xy.mean() / 256.0 + angle.mean() / np.pi)
    return loss, {
        "strict_success_count": int(success.sum()),
        "strict_success_rate": float(success.mean()),
        "mean_block_xy_distance": float(xy.mean()),
        "mean_angle_error": float(angle.mean()),
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    model, normalizer, saved = load_checkpoint(args.checkpoint, args.device)
    runtime = SimpleNamespace(**vars(saved))
    for key in (
        "planning_episodes", "planning_steps", "planning_horizon", "cem_samples", "cem_iterations",
        "action_repeat", "contact_servo", "direct_utility", "task_space_objective", "device",
    ):
        setattr(runtime, key, getattr(args, key))
    runtime.seed = int(args.diagnostic_seed)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=args.planning_steps * args.action_repeat)
    try:
        roots = balanced_source_cases(
            env, args.diagnostic_roots, args.diagnostic_root_rollout_steps, args.diagnostic_seed,
            float(saved.contact_root_fraction), int(saved.contact_approach_steps),
        )
        diagnostic = make_counterfactual_dataset(
            env, roots, int(saved.image_size), args.diagnostic_seed + 1, args.action_repeat, 0,
        )
        diagnostics = _operator_diagnostics(model, diagnostic, normalizer, args.device, args.planning_horizon)
        # Direct-utility control consumes one action at a time; rollout MPC
        # consumes ``planning_horizon`` actions.  This is the effective
        # predictor horizon, rather than the maximum episode length.
        effective_horizon = 1 if args.direct_utility else int(args.planning_horizon)
        diagnostics["prediction_mse"] = planner_matched_rollout_mse(
            env, model, normalizer, runtime, roots, effective_horizon, args.action_repeat,
            args.diagnostic_seed + 2,
        )
        episodes = evaluate_visual_planning(env, model, normalizer, runtime)
    finally:
        env.close()
    loss, heldout = planning_loss(episodes)
    payload = {
        "record_schema": "real_checkpoint_selection.v1",
        "protocol": {
            "environment": "swm/PushT-v1",
            "diagnostic_split": "frozen before checkpoint selection",
            "heldout_planning": "disjoint from diagnostic split",
            "coordinate_gauge": "decoded_observable_pose_raw",
            "render_coordinate_scale": float(PUSHT_RENDER_SIZE),
            "task_metric": task_metric_metadata(pusht_task_metric()),
            "task_metric_variants": {
                "pre_registered": list(PRE_REGISTERED_TASK_METRIC_VARIANTS) + ["action_effect"],
                "fit_split": "diagnostic split only",
                "selection_split": "diagnostic split only",
                "heldout_outcome": "disjoint planning episodes",
                "action_effect_construction": "action-to-observed-effect Gram metric with fixed trace-scaled ridges 1e-3 and 0.05",
            },
            "runtime_input": "previous/current rendered image and goal image only" if saved.temporal_context else "current rendered image and goal image only",
            "runtime_state_access": False,
            "runtime_successor_verification": False,
            "checkpoint_immutable": True,
            "prediction_quality": {
                "metric": "planner_matched_open_loop_rollout_mse",
                "diagnostic_roots_disjoint_from_planning_test": True,
                "action_distribution": "frozen_rgb_only_controller_actions_on_diagnostic_roots",
                "effective_horizon": effective_horizon,
                "root_count": len(roots),
                "root_state_goal_sha256": _root_hash(roots),
                "one_step_counterfactual_mse_retained_as": "diagnostics.one_step_counterfactual_mse",
            },
            "interface": {
                "contact_servo": bool(args.contact_servo),
                "direct_utility": bool(args.direct_utility),
                "task_space_objective": bool(args.task_space_objective),
                "utility_mode": str(saved.utility_mode),
                "adapter_architecture": str(saved.adapter_architecture),
                "planning_horizon": int(args.planning_horizon),
                "cem_samples": int(args.cem_samples),
                "cem_iterations": int(args.cem_iterations),
                "action_repeat": int(args.action_repeat),
                "planning_steps": int(args.planning_steps),
            },
        },
        "record": {
            "group": args.group,
            "checkpoint": args.checkpoint_label,
            "model_factors": {
                "latent_dimension": int(saved.latent_dim),
                "training_stage": int(next(
                    (token.split("_")[-1] for token in str(args.checkpoint_label).split("/") if token.startswith("epoch_")),
                    "-1",
                )),
                "action_distribution": "frozen_rgb_only_controller_actions_on_diagnostic_roots",
                "predictor_spectrum_source": "mean_zero_action_local_jacobian_in_decoded_raw_chart",
            },
            "diagnostics": diagnostics,
            "heldout": {"planning_loss": loss, **heldout},
        },
        "episodes": episodes,
        "claim_boundary": "One real frozen checkpoint under a fixed PushT visual-only interface. It must be combined with independently trained checkpoints and other environments before claiming model-selection generality.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoint_record.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["record"], indent=2))


if __name__ == "__main__":
    main()
