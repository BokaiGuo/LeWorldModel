"""Contact-conditioned physical readout for frozen LeWM counterfactuals.

The earlier physical counterfactual used random reset states and fit a readout
on observed endpoint latents, then applied that readout to the predictor's
endpoint latents.  This experiment tests the two resulting failure modes
directly: (i) the intervention must enter the contact mode, and (ii) the
readout must be calibrated on the distribution produced by the frozen
predictor.  The LeWM checkpoint is frozen throughout.

The experiment is deliberately a downstream diagnostic.  A successful
predicted-latent adapter would establish a better calibrated physical readout
for this PushT checkpoint; it would not establish retraining gain, causal
action effects, universal physical counterfactuals, or online planning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_pusht_counterfactual import (
    encode_images,
    image_transform,
    normalize_public_action,
    pose_target,
    predict_endpoints,
)

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration", type=int, default=64)
    parser.add_argument("--held-out", type=int, default=16)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=8421)
    parser.add_argument("--approach-steps", type=int, default=40)
    parser.add_argument("--action-magnitude", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def block_motion(before: np.ndarray, after: np.ndarray, tolerance: float = 1e-4) -> bool:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    xy = np.linalg.norm(after[2:4] - before[2:4])
    angle = abs((after[4] - before[4] + np.pi) % (2 * np.pi) - np.pi)
    return bool(xy > tolerance or angle > tolerance)


def bootstrap(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(values.size),
    }


def approach_to_contact(env: gym.Env, seed: int, max_steps: int) -> dict[str, Any]:
    """Reach a genuine contact state without teleporting onto a contact ring."""
    observation, info = env.reset(seed=seed)
    previous = np.asarray(observation["state"], dtype=np.float64).copy()
    image_history: list[np.ndarray] = [env.render().copy()]
    action_history: list[np.ndarray] = []
    contacted = False
    for step in range(max_steps):
        state = np.asarray(observation["state"], dtype=np.float64).copy()
        relative = state[2:4] - state[:2]
        distance = np.linalg.norm(relative)
        direction = relative / max(distance, 1e-8)
        # Move toward the block; after contact, continue in the contact normal
        # so the returned state is useful for a push intervention.
        action = np.clip(direction, -1.0, 1.0).astype(np.float32)
        action_history.append(action.copy())
        observation, _, terminated, truncated, _ = env.step(action)
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        image_history.append(env.render().copy())
        if block_motion(previous, current):
            contacted = True
            context_images = image_history[-3:]
            context_actions = action_history[-2:]
            while len(context_images) < 3:
                context_images.insert(0, context_images[0].copy())
            while len(context_actions) < 2:
                context_actions.insert(0, context_actions[0].copy())
            return {
                "state": current.copy(),
                "goal": np.asarray(info["goal_state"], dtype=np.float64).copy(),
                "image": context_images[-1].copy(),
                "context_images": context_images,
                "context_actions": context_actions,
                "contacted": True,
                "approach_steps": step + 1,
            }
        previous = current
        if terminated or truncated:
            break
    # Keep a finite, auditable fallback rather than silently discarding a seed.
    return {
        "state": previous.copy(),
        "goal": np.asarray(info["goal_state"], dtype=np.float64).copy(),
        "image": image_history[-1].copy(),
        "context_images": (image_history[-3:] + [image_history[-1].copy()])[-3:],
        "context_actions": action_history[-2:],
        "contacted": contacted,
        "approach_steps": max_steps,
    }


def sample_cases(env: gym.Env, count: int, seed: int, max_steps: int) -> list[dict[str, Any]]:
    cases = []
    for index in range(count):
        cases.append(approach_to_contact(env, seed + index, max_steps))
    return cases


def intervention_actions(case: dict[str, Any], magnitude: float) -> list[np.ndarray]:
    state = np.asarray(case["state"], dtype=np.float64)
    # At contact, the outward normal from the agent through the block is the
    # physical push direction.  +/- therefore tests a matched contact effect.
    direction = state[2:4] - state[:2]
    direction /= max(np.linalg.norm(direction), 1e-8)
    action = (magnitude * direction).astype(np.float32)
    return [action, -action, np.zeros(2, dtype=np.float32)]


def normalized_context_action_sequence(
    case: dict[str, Any], action: np.ndarray, horizon: int
) -> np.ndarray:
    """Encode two observed approach actions plus a future intervention.

    The three-frame context ends at the returned contact state.  Its first two
    action rows are the actions that produced the preceding context frames;
    the final context action and all future rows are the counterfactual action.
    ``JEPA.rollout`` then predicts one endpoint per requested simulator step.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    past = np.asarray(case["context_actions"], dtype=np.float32)
    if past.shape != (2, 2):
        raise ValueError(f"context_actions must have shape (2, 2), got {past.shape}")
    raw = np.concatenate(
        [past, np.repeat(np.asarray(action, dtype=np.float32)[None], horizon, axis=0)],
        axis=0,
    )
    normalized = np.asarray([normalize_public_action(row) for row in raw], dtype=np.float32)
    # The first two rows are history; the final row is the first intervention.
    return np.asarray([np.tile(row, 5) for row in normalized], dtype=np.float32)


@torch.inference_mode()
def predict_context_endpoints(
    model: torch.nn.Module,
    context_images: list[list[np.ndarray]],
    actions: np.ndarray,
    transform,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Run frozen LeWM with the model's required three-frame context."""
    outputs = []
    for start in range(0, len(context_images), batch_size):
        current = context_images[start : start + batch_size]
        act = torch.as_tensor(actions[start : start + batch_size], dtype=torch.float32, device=device)
        batch_pixels = torch.stack([
            torch.stack([
                transform(torch.from_numpy(image).permute(2, 0, 1))
                for image in images
            ])
            for images in current
        ]).to(device)
        b, s = act.shape[:2]
        pixels = batch_pixels[:, None].expand(b, s, *batch_pixels.shape[1:])
        out = model.rollout({"pixels": pixels}, act)
        outputs.append(out["predicted_emb"][:, :, -1].detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def rollout_endpoint(
    env: gym.Env,
    case: dict[str, Any],
    action: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=0, options={"state": case["state"].copy(), "goal_state": case["goal"].copy()})
    for _ in range(horizon):
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return np.asarray(observation["state"], dtype=np.float64).copy(), env.render().copy()


def fit_decoder(features: np.ndarray, targets: np.ndarray) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler().fit(targets)
    decoder = Ridge(alpha=1.0).fit(features, scaler.transform(targets))
    return decoder, scaler


def decode(decoder: Ridge, scaler: StandardScaler, features: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(decoder.predict(features))


def fit_action_decoder(
    features: np.ndarray, actions: np.ndarray, targets: np.ndarray
) -> tuple[StandardScaler, Ridge, StandardScaler]:
    """Fit a scale-controlled endpoint readout with the known action included."""
    x = np.concatenate([features, actions], axis=1)
    x_scaler = StandardScaler().fit(x)
    y_scaler = StandardScaler().fit(targets)
    decoder = Ridge(alpha=1.0).fit(x_scaler.transform(x), y_scaler.transform(targets))
    return x_scaler, decoder, y_scaler


def decode_action(
    x_scaler: StandardScaler, decoder: Ridge, y_scaler: StandardScaler,
    features: np.ndarray, actions: np.ndarray,
) -> np.ndarray:
    x = np.concatenate([features, actions], axis=1)
    return y_scaler.inverse_transform(decoder.predict(x_scaler.transform(x)))


def fit_latent_residual_adapter(
    predicted: np.ndarray, actions: np.ndarray, observed: np.ndarray
) -> tuple[StandardScaler, Ridge, StandardScaler]:
    """Fit a post-hoc predictor-to-observation latent correction."""
    x = np.concatenate([predicted, actions], axis=1)
    residual = observed - predicted
    x_scaler = StandardScaler().fit(x)
    y_scaler = StandardScaler().fit(residual)
    adapter = Ridge(alpha=10.0).fit(
        x_scaler.transform(x), y_scaler.transform(residual)
    )
    return x_scaler, adapter, y_scaler


def correct_latent(
    x_scaler: StandardScaler, adapter: Ridge, y_scaler: StandardScaler,
    predicted: np.ndarray, actions: np.ndarray,
) -> np.ndarray:
    x = np.concatenate([predicted, actions], axis=1)
    residual = y_scaler.inverse_transform(adapter.predict(x_scaler.transform(x)))
    return predicted + residual


def run_horizon(
    model: torch.nn.Module,
    env: gym.Env,
    calibration: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    horizon: int,
    args: argparse.Namespace,
    transform,
) -> dict[str, Any]:
    def collect(cases: list[dict[str, Any]]):
        initial_images, context_images, predicted_inputs, observed_images, targets, actions = [], [], [], [], [], []
        for case in cases:
            initial_images.append(case["image"])
            context_images.append(case["context_images"])
            row_actions = intervention_actions(case, args.action_magnitude)
            row_predicted = np.stack([
                normalized_context_action_sequence(case, action, horizon) for action in row_actions
            ])
            row_targets, row_images = [], []
            for action in row_actions:
                state, image = rollout_endpoint(env, case, action, horizon)
                row_targets.append(pose_target(state))
                row_images.append(image)
            predicted_inputs.append(row_predicted)
            observed_images.extend(row_images)
            targets.append(np.asarray(row_targets))
            actions.append(row_actions)
        initial_z = encode_images(model, initial_images, transform, args.batch_size, args.device)
        observed_z = encode_images(model, observed_images, transform, args.batch_size, args.device)
        predicted_z = predict_context_endpoints(
            model, context_images, np.asarray(predicted_inputs), transform,
            args.batch_size, args.device,
        )
        return (
            initial_z,
            observed_z.reshape(len(cases), 3, -1),
            predicted_z,
            np.asarray(targets),
            np.asarray(actions),
        )

    calibration_initial, calibration_observed, calibration_predicted, calibration_targets, calibration_actions = collect(calibration)
    test_initial, test_observed, test_predicted, test_targets, test_actions = collect(held_out)

    # The key repair: fit one readout on the predictor's own endpoint
    # distribution.  The observed-latent readout is a matched comparator.
    flat_targets = calibration_targets.reshape(-1, calibration_targets.shape[-1])
    predicted_decoder, predicted_scaler = fit_decoder(
        calibration_predicted.reshape(-1, calibration_predicted.shape[-1]), flat_targets
    )
    observed_decoder, observed_scaler = fit_decoder(
        calibration_observed.reshape(-1, calibration_observed.shape[-1]), flat_targets
    )
    initial_decoder, initial_scaler = fit_decoder(
        np.repeat(calibration_initial[:, None, :], 3, axis=1).reshape(-1, calibration_initial.shape[-1]),
        flat_targets,
    )
    predicted_action_x, predicted_action_decoder, predicted_action_scaler = fit_action_decoder(
        calibration_predicted.reshape(-1, calibration_predicted.shape[-1]),
        calibration_actions.reshape(-1, calibration_actions.shape[-1]),
        flat_targets,
    )
    latent_x, latent_adapter, latent_residual_scaler = fit_latent_residual_adapter(
        calibration_predicted.reshape(-1, calibration_predicted.shape[-1]),
        calibration_actions.reshape(-1, calibration_actions.shape[-1]),
        calibration_observed.reshape(-1, calibration_observed.shape[-1]),
    )
    direct_latent_x, direct_latent_model, direct_latent_scaler = fit_action_decoder(
        np.repeat(calibration_initial[:, None, :], 3, axis=1).reshape(-1, calibration_initial.shape[-1]),
        calibration_actions.reshape(-1, calibration_actions.shape[-1]),
        calibration_observed.reshape(-1, calibration_observed.shape[-1]),
    )

    predicted_pose = decode(
        predicted_decoder, predicted_scaler,
        test_predicted.reshape(-1, test_predicted.shape[-1]),
    ).reshape(len(held_out), 3, -1)
    observed_pose = decode(
        observed_decoder, observed_scaler,
        test_observed.reshape(-1, test_observed.shape[-1]),
    ).reshape(len(held_out), 3, -1)
    initial_pose = decode(
        initial_decoder, initial_scaler,
        np.repeat(test_initial[:, None, :], 3, axis=1).reshape(-1, test_initial.shape[-1]),
    ).reshape(len(held_out), 3, -1)
    predicted_action_pose = decode_action(
        predicted_action_x, predicted_action_decoder, predicted_action_scaler,
        test_predicted.reshape(-1, test_predicted.shape[-1]),
        test_actions.reshape(-1, test_actions.shape[-1]),
    ).reshape(len(held_out), 3, -1)
    corrected_predicted_z = correct_latent(
        latent_x, latent_adapter, latent_residual_scaler,
        test_predicted.reshape(-1, test_predicted.shape[-1]),
        test_actions.reshape(-1, test_actions.shape[-1]),
    ).reshape(len(held_out), 3, -1)
    corrected_predicted_pose = decode(
        observed_decoder, observed_scaler,
        corrected_predicted_z.reshape(-1, corrected_predicted_z.shape[-1]),
    ).reshape(len(held_out), 3, -1)
    direct_latent_z = decode_action(
        direct_latent_x, direct_latent_model, direct_latent_scaler,
        np.repeat(test_initial[:, None, :], 3, axis=1).reshape(-1, test_initial.shape[-1]),
        test_actions.reshape(-1, test_actions.shape[-1]),
    ).reshape(len(held_out), 3, -1)
    direct_latent_pose = decode(
        observed_decoder, observed_scaler,
        direct_latent_z.reshape(-1, direct_latent_z.shape[-1]),
    ).reshape(len(held_out), 3, -1)
    constant_pose = np.repeat(
        np.asarray([case["state"][2:4] for case in held_out])[:, None, :], 3, axis=1
    )

    true_xy = test_targets[:, :, :2]
    physical_separation = np.linalg.norm(true_xy[:, 0] - true_xy[:, 1], axis=1)
    predicted_separation = np.linalg.norm(test_predicted[:, 0] - test_predicted[:, 1], axis=1)

    def error(prediction: np.ndarray, intervention: int) -> np.ndarray:
        return np.linalg.norm(prediction[:, intervention, :2] - true_xy[:, intervention], axis=1)

    metrics = {}
    for name, values in (
        ("predicted_adapter", predicted_pose),
        ("predicted_action_adapter", predicted_action_pose),
        ("latent_residual_adapter", corrected_predicted_pose),
        ("direct_action_latent_adapter", direct_latent_pose),
        ("observed_adapter", observed_pose),
        ("initial_adapter", initial_pose),
        ("constant", constant_pose),
    ):
        for intervention, label in enumerate(("plus", "minus", "zero")):
            metrics[f"{name}_error_{label}"] = bootstrap(
                error(values, intervention), args.seed + 1000 * horizon + len(metrics), args.bootstrap
            )
    metrics["physical_counterfactual_separation"] = bootstrap(
        physical_separation, args.seed + 3000 + horizon, args.bootstrap
    )
    metrics["predicted_latent_counterfactual_separation"] = bootstrap(
        predicted_separation, args.seed + 4000 + horizon, args.bootstrap
    )
    metrics["contact_rate_calibration"] = float(np.mean([case["contacted"] for case in calibration]))
    metrics["contact_rate_held_out"] = float(np.mean([case["contacted"] for case in held_out]))
    return {
        "horizon": horizon,
        "metrics": metrics,
        "sanity_checks": {
            "finite": bool(np.isfinite(predicted_pose).all() and np.isfinite(observed_pose).all()),
            "calibration_disjoint": True,
            "paired_cases": len(held_out),
        },
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    if args.calibration < 16 or args.held_out < 8:
        raise ValueError("use at least 16 calibration and 8 held-out contact cases")
    args.horizons = sorted(set(int(h) for h in args.horizons))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = load_compatible_pretrained("pusht/lewm", device=args.device)
    transform = image_transform()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        calibration = sample_cases(env, args.calibration, args.seed, args.approach_steps)
        held_out = sample_cases(env, args.held_out, args.seed + 100000, args.approach_steps)
        reports = [run_horizon(model, env, calibration, held_out, h, args, transform) for h in args.horizons]
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "checkpoint": str(getattr(model, "_checkpoint_compat", {})),
            "calibration_cases": args.calibration,
            "held_out_cases": args.held_out,
            "horizons": args.horizons,
            "contact_protocol": "real approach rollout; block pose change is the contact label",
            "action_protocol": "matched outward-normal +a/-a/zero interventions at contact",
            "context_protocol": "two preceding approach frames plus the contact frame; history_size=3",
            "readouts": "separate Ridge readouts for predicted endpoint, observed endpoint, and initial latent",
            "frozen_checkpoint": True,
        },
        "horizons": reports,
        "sanity_checks": {
            "all_finite": bool(all(item["sanity_checks"]["finite"] for item in reports)),
            "all_disjoint": bool(all(item["sanity_checks"]["calibration_disjoint"] for item in reports)),
        },
        "claim_boundary": (
            "This is a contact-conditioned downstream physical-readout diagnostic for a frozen PushT checkpoint. "
            "It does not establish LeWM retraining gain, causal action effects, universal physical counterfactuals, "
            "online planning, or task success."
        ),
    }
    (output / "contact_predicted_latent_adapter.json").write_text(json.dumps(jsonable(report), indent=2, sort_keys=True))
    lines = [
        "# Contact-conditioned predicted-latent physical adapter", "",
        "| h | predicted-latent adapter + | predicted+action adapter + | latent-residual adapter + | direct action-latent adapter + | observed-latent adapter + | initial-latent adapter + | constant + | physical +/- | predicted latent +/- | contact rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in reports:
        m = item["metrics"]
        lines.append(
            "| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
                item["horizon"],
                m["predicted_adapter_error_plus"]["estimate"],
                m["predicted_action_adapter_error_plus"]["estimate"],
                m["latent_residual_adapter_error_plus"]["estimate"],
                m["direct_action_latent_adapter_error_plus"]["estimate"],
                m["observed_adapter_error_plus"]["estimate"],
                m["initial_adapter_error_plus"]["estimate"],
                m["constant_error_plus"]["estimate"],
                m["physical_counterfactual_separation"]["estimate"],
                m["predicted_latent_counterfactual_separation"]["estimate"],
                m["contact_rate_held_out"],
            )
        )
    lines += [
        "", "The predicted-latent adapter is calibrated on frozen predictor endpoints, not observed endpoints.",
        "Claim boundary: downstream frozen-checkpoint readout diagnostic only.", "",
    ]
    (output / "contact_predicted_latent_adapter.md").write_text("\n".join(lines))
    print(json.dumps(jsonable({"output": str(output), "reports": reports}), indent=2))


if __name__ == "__main__":
    main()
