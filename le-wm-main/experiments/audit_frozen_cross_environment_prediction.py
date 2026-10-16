"""Audit one frozen LeWM checkpoint across live, distinct visual environments.

This is an external *prediction* experiment, deliberately not a planning
proxy.  A single immutable checkpoint is applied without adapter training to
RGB/action windows collected online from each requested simulator.  Per-domain
action standardisation is estimated only from disjoint calibration episodes;
model weights, architecture, action encoder, and diagnostic definitions stay
fixed.  The report therefore supports only a scoped action-conditioned
prediction statement, never universal planning or universal world-model claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401: registers environments
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.dmcontrol_compat import compatibility_status, install_missing_field_compat
from experiments.empirical_spectral_audit.run_lewm_external_pusht_temporal_audit import _predict_horizons, _relative_commutator
from utils import get_img_preprocessor


CONTEXT = 3


def _bootstrap(values: np.ndarray, seed: int, draws: int = 2000) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("episode bootstrap requires at least two values")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return {
        "n_episodes": int(values.size),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "bootstrap_seed": int(seed),
    }


def _sample_action(space: gym.spaces.Box, rng: np.random.Generator) -> np.ndarray:
    if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
        raise ValueError("cross-environment audit currently requires a one-dimensional continuous Box action space")
    if not (np.isfinite(space.low).all() and np.isfinite(space.high).all()):
        raise ValueError("cross-environment audit requires finite action bounds")
    return rng.uniform(space.low, space.high).astype(np.float32)


def _collect_episode(
    env: gym.Env,
    *,
    seed: int,
    blocks: int,
    action_block: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Collect rendered observations and the intervening physical action blocks."""
    _observation, _info = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 20_000)
    frames: list[np.ndarray] = [np.asarray(env.render(), dtype=np.uint8).copy()]
    actions: list[np.ndarray] = []
    for _ in range(blocks):
        block: list[np.ndarray] = []
        terminated = truncated = False
        for _ in range(action_block):
            action = _sample_action(env.action_space, rng)
            _observation, _reward, terminated, truncated, _info = env.step(action)
            block.append(action)
            if terminated or truncated:
                break
        if len(block) != action_block:
            return None
        actions.append(np.concatenate(block, axis=0))
        frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
        if terminated or truncated:
            return None
    # The final action embedding is not consumed for any evaluated successor,
    # but model.encode requires an action row per frame.
    actions.append(np.zeros_like(actions[-1]))
    return np.stack(frames), np.stack(actions)


def _collect_split(
    env_id: str,
    *,
    start_seed: int,
    episodes: int,
    blocks: int,
    action_block: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    env = gym.make(env_id, render_mode="rgb_array")
    frames, actions = [], []
    try:
        # Allow extra resets because an environment may terminate before the
        # fixed temporal window.  Accepted episode IDs are recorded below.
        for offset in range(episodes * 4):
            if len(frames) >= episodes:
                break
            item = _collect_episode(env, seed=start_seed + offset, blocks=blocks, action_block=action_block)
            if item is not None:
                image, action = item
                frames.append(image)
                actions.append(action)
    finally:
        env.close()
    if len(frames) != episodes:
        raise RuntimeError(f"{env_id} yielded only {len(frames)}/{episodes} full temporal windows")
    return frames, actions


def _action_stats(action_rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate([row[:-1] for row in action_rows], axis=0).astype(np.float32)
    return flat.mean(axis=0), flat.std(axis=0).clip(1e-6)


def _windows(length: int, maximum_horizon: int) -> list[int]:
    needed = CONTEXT + maximum_horizon
    if length < needed:
        return []
    return sorted({0, length - needed})


def _collect_prediction_vectors(
    model: torch.nn.Module,
    frame_rows: list[np.ndarray],
    action_rows: list[np.ndarray],
    action_mean: np.ndarray,
    action_std: np.ndarray,
    *,
    horizons: list[int],
    maximum_horizon: int,
    device: torch.device,
    action_matrix: np.ndarray | None = None,
    action_bias: np.ndarray | None = None,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Collect raw predictor/target vectors without fitting on the split."""
    preprocess = get_img_preprocessor("pixels", "pixels", 224)
    vectors: dict[str, dict[str, list[np.ndarray]]] = {}
    for episode, (frames, actions) in enumerate(zip(frame_rows, action_rows)):
        pixel_sequence = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        pixels = preprocess({"pixels": pixel_sequence})["pixels"].unsqueeze(0).to(device)
        standardized = (actions - action_mean) / action_std
        zero_physical = (np.zeros_like(actions) - action_mean) / action_std
        if action_matrix is not None or action_bias is not None:
            if action_matrix is None or action_bias is None:
                raise ValueError("action_matrix and action_bias must be supplied together")
            raw_dim = action_matrix.shape[0]
            if standardized.shape[1] % raw_dim != 0:
                raise ValueError("action adapter raw dimension does not divide flattened action width")
            action_block = standardized.shape[1] // raw_dim
            matrix = np.asarray(action_matrix, dtype=np.float32)
            bias = np.asarray(action_bias, dtype=np.float32)
            if matrix.shape != (raw_dim, raw_dim) or bias.shape != (raw_dim,):
                raise ValueError("invalid action adapter shape")
            def adapt(values: np.ndarray) -> np.ndarray:
                blocks = values.reshape(values.shape[0], action_block, raw_dim)
                mapped = blocks @ matrix.T + bias
                return mapped.reshape(values.shape[0], -1).astype(np.float32)
            standardized = adapt(standardized)
            zero_physical = adapt(zero_physical)
        for mode, blocks_array in (("observed", standardized), ("zero_physical", zero_physical)):
            blocks_tensor = torch.from_numpy(blocks_array).unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                encoded = model.encode({"pixels": pixels, "action": blocks_tensor})
                predictions = _predict_horizons(model, encoded["emb"], encoded["act_emb"], horizons, CONTEXT)
            for start in _windows(frames.shape[0], maximum_horizon):
                if start:
                    with torch.inference_mode():
                        window = model.encode({
                            "pixels": pixels[:, start : start + CONTEXT + maximum_horizon],
                            "action": blocks_tensor[:, start : start + CONTEXT + maximum_horizon],
                        })
                        predictions = _predict_horizons(model, window["emb"], window["act_emb"], horizons, CONTEXT)
                    embeddings = window["emb"]
                else:
                    embeddings = encoded["emb"]
                for horizon in horizons:
                    key = f"{mode}/h{horizon}"
                    entry = vectors.setdefault(key, {"prediction": [], "target": [], "episode": []})
                    entry["target"].append(embeddings[:, CONTEXT - 1 + horizon].detach().cpu().numpy()[0])
                    entry["prediction"].append(predictions[horizon].detach().cpu().numpy()[0])
                    entry["episode"].append(np.asarray(episode))
    return vectors


def _fit_action_interface_adapter(
    model: torch.nn.Module,
    frame_rows: list[np.ndarray],
    action_rows: list[np.ndarray],
    action_mean: np.ndarray,
    action_std: np.ndarray,
    *,
    action_block: int,
    raw_dim: int,
    device: torch.device,
    steps: int,
    learning_rate: float,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit a small target-action -> checkpoint-action map on calibration only.

    The LeWM weights remain frozen.  The adapter is deliberately shared over
    the five frame-skip rows, initialized at identity, and regularized toward
    identity so that a small calibration split cannot invent an arbitrary
    action policy.  The returned map is evaluated only on the disjoint test
    split by the caller.
    """
    preprocess = get_img_preprocessor("pixels", "pixels", 224)
    maximum_horizon = 2
    pixel_batches: list[torch.Tensor] = []
    action_batches: list[torch.Tensor] = []
    for frames, actions in zip(frame_rows, action_rows):
        pixels = preprocess({
            "pixels": torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        })["pixels"]
        standardized = (actions - action_mean) / action_std
        for start in _windows(frames.shape[0], maximum_horizon):
            sl = slice(start, start + CONTEXT + maximum_horizon)
            pixel_batches.append(pixels[sl])
            action_batches.append(torch.from_numpy(standardized[sl]).float())
    if len(pixel_batches) < 2:
        raise ValueError("action adapter requires at least two calibration windows")
    pixels = torch.stack(pixel_batches).to(device)
    actions = torch.stack(action_batches).to(device)
    identity = torch.eye(raw_dim, device=device)
    matrix = torch.nn.Parameter(identity.clone())
    bias = torch.nn.Parameter(torch.zeros(raw_dim, device=device))
    optimizer = torch.optim.Adam([matrix, bias], lr=learning_rate)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    def differentiable_predictions(
        embedding: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        context_emb = embedding[:, :CONTEXT].clone()
        context_act = action_embedding[:, :CONTEXT].clone()
        predictions: dict[int, torch.Tensor] = {}
        for horizon in range(1, maximum_horizon + 1):
            pred = model.predict(context_emb, context_act)[:, -1]
            predictions[horizon] = pred
            if horizon < maximum_horizon:
                next_action = action_embedding[:, CONTEXT - 1 + horizon : CONTEXT + horizon]
                context_emb = torch.cat((context_emb, pred[:, None]), dim=1)[:, -CONTEXT:]
                context_act = torch.cat((context_act, next_action), dim=1)[:, -CONTEXT:]
        return predictions

    initial_loss = float("nan")
    final_loss = float("nan")
    for step in range(max(1, steps)):
        blocks = actions.reshape(actions.shape[0], actions.shape[1], action_block, raw_dim)
        mapped = (blocks @ matrix.T + bias).reshape(actions.shape[0], actions.shape[1], -1)
        encoded = model.encode({"pixels": pixels, "action": mapped})
        predictions = differentiable_predictions(encoded["emb"], encoded["act_emb"])
        target = encoded["emb"][:, CONTEXT : CONTEXT + 2].detach()
        prediction = torch.stack([predictions[1], predictions[2]], dim=1)
        fit_loss = (prediction - target).pow(2).mean()
        penalty = regularization * ((matrix - identity).pow(2).mean() + bias.pow(2).mean())
        loss = fit_loss + penalty
        if step == 0:
            initial_loss = float(fit_loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(fit_loss.detach().cpu())
    return (
        matrix.detach().cpu().numpy().astype(np.float32),
        bias.detach().cpu().numpy().astype(np.float32),
        {
            "calibration_windows": float(len(pixel_batches)),
            "initial_fit_mse": initial_loss,
            "final_fit_mse": final_loss,
            "steps": float(max(1, steps)),
            "learning_rate": float(learning_rate),
            "regularization": float(regularization),
        },
    )


def _fit_diagonal_affine(
    calibration_vectors: dict[str, dict[str, list[np.ndarray]]],
    horizons: list[int],
    ridge: float,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Fit a calibration-only diagonal affine map prediction -> target."""
    maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for horizon in horizons:
        source = np.stack(calibration_vectors[f"observed/h{horizon}"]["prediction"])
        target = np.stack(calibration_vectors[f"observed/h{horizon}"]["target"])
        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)
        centered_source = source - source_mean
        centered_target = target - target_mean
        variance = np.mean(centered_source * centered_source, axis=0)
        covariance = np.mean(centered_source * centered_target, axis=0)
        scale = covariance / (variance + float(ridge))
        intercept = target_mean - scale * source_mean
        maps[horizon] = (scale.astype(np.float32), intercept.astype(np.float32))
    return maps


def _summarize_vectors(
    vectors: dict[str, dict[str, list[np.ndarray]]],
    horizons: list[int],
    test_episodes: int,
    seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"by_mode_horizon": {}, "paired_action_effect": {}}
    for horizon in horizons:
        episode_values: dict[str, np.ndarray] = {}
        for mode in ("observed", "zero_physical"):
            entry = vectors[f"{mode}/h{horizon}"]
            prediction = np.stack(entry["prediction"])
            target = np.stack(entry["target"])
            per_row = np.mean(np.square(prediction - target), axis=1)
            episodes = np.asarray(entry["episode"], dtype=np.int64)
            values = np.asarray([
                per_row[episodes == episode].mean() for episode in range(test_episodes)
            ])
            episode_values[mode] = values
            summary["by_mode_horizon"][f"{mode}/h{horizon}"] = {
                "mse": _bootstrap(values, seed + horizon + (0 if mode == "observed" else 100)),
                "relative_commutator": _relative_commutator(target, prediction),
            }
        difference = episode_values["zero_physical"] - episode_values["observed"]
        summary["paired_action_effect"][f"h{horizon}"] = {
            "zero_minus_observed_mse": _bootstrap(difference, seed + 200 + horizon),
            "fraction_zero_worse": float(np.mean(difference > 0.0)),
        }
    return summary


def _environment_report(
    model: torch.nn.Module,
    *,
    environment: str,
    calibration_episodes: int,
    test_episodes: int,
    blocks: int,
    action_block: int,
    horizons: list[int],
    seed: int,
    device: torch.device,
    affine_calibration: bool,
    affine_ridge: float,
    action_adapter: bool,
    action_adapter_steps: int,
    action_adapter_learning_rate: float,
    action_adapter_regularization: float,
) -> dict[str, Any]:
    probe = gym.make(environment, render_mode="rgb_array")
    try:
        if not isinstance(probe.action_space, gym.spaces.Box) or len(probe.action_space.shape) != 1:
            return {"environment": environment, "status": "incompatible_action_space", "action_space": repr(probe.action_space)}
        raw_dim = int(probe.action_space.shape[0])
    finally:
        probe.close()
    expected_dim = int(model.action_encoder.input_dim)
    if raw_dim * action_block != expected_dim:
        return {
            "environment": environment,
            "status": "incompatible_action_encoding",
            "raw_action_dim": raw_dim,
            "action_block": action_block,
            "checkpoint_action_input_dim": expected_dim,
        }
    maximum = max(horizons)
    if blocks + 1 < CONTEXT + maximum:
        raise ValueError("blocks must provide at least context + maximum horizon observations")
    calibration_frames, calibration_actions = _collect_split(
        environment, start_seed=seed, episodes=calibration_episodes, blocks=blocks, action_block=action_block,
    )
    test_frames, test_actions = _collect_split(
        environment, start_seed=seed + 1_000_000, episodes=test_episodes, blocks=blocks, action_block=action_block,
    )
    action_mean, action_std = _action_stats(calibration_actions)
    calibration_vectors = _collect_prediction_vectors(
        model, calibration_frames, calibration_actions, action_mean, action_std,
        horizons=horizons, maximum_horizon=maximum, device=device,
    )
    test_vectors = _collect_prediction_vectors(
        model, test_frames, test_actions, action_mean, action_std,
        horizons=horizons, maximum_horizon=maximum, device=device,
    )
    summary = _summarize_vectors(test_vectors, horizons, test_episodes, seed)
    calibration_maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if affine_calibration:
        calibration_maps = _fit_diagonal_affine(calibration_vectors, horizons, affine_ridge)
        calibrated_vectors = {
            key: {name: list(values) for name, values in entry.items()}
            for key, entry in test_vectors.items()
        }
        for horizon, (scale, intercept) in calibration_maps.items():
            for mode in ("observed", "zero_physical"):
                entry = calibrated_vectors[f"{mode}/h{horizon}"]
                entry["prediction"] = [prediction * scale + intercept for prediction in entry["prediction"]]
        calibrated_summary = _summarize_vectors(calibrated_vectors, horizons, test_episodes, seed + 1000)
        summary["affine_calibrated"] = calibrated_summary
        summary["affine_calibration_protocol"] = {
            "fit_split": "calibration_episodes_observed_action_only",
            "ridge": float(affine_ridge),
            "calibration_windows": len(calibration_vectors["observed/h1"]["prediction"]),
        }
    if action_adapter:
        matrix, bias, adapter_fit = _fit_action_interface_adapter(
            model, calibration_frames, calibration_actions, action_mean, action_std,
            action_block=action_block, raw_dim=raw_dim, device=device,
            steps=action_adapter_steps, learning_rate=action_adapter_learning_rate,
            regularization=action_adapter_regularization,
        )
        adapted_vectors = _collect_prediction_vectors(
            model, test_frames, test_actions, action_mean, action_std,
            horizons=horizons, maximum_horizon=maximum, device=device,
            action_matrix=matrix, action_bias=bias,
        )
        summary["action_adapter"] = _summarize_vectors(
            adapted_vectors, horizons, test_episodes, seed + 2000,
        )
        summary["action_adapter_protocol"] = {
            "fit_split": "calibration_episodes_observed_action_only",
            "weights_frozen": True,
            "shared_over_frame_skip": True,
            "raw_action_dim": raw_dim,
            "action_block": action_block,
            "fit": adapter_fit,
            "matrix": matrix.tolist(),
            "bias": bias.tolist(),
        }
    return {
        "environment": environment,
        "status": "completed",
        "raw_action_dim": raw_dim,
        "action_block": action_block,
        "calibration_action_mean": action_mean.tolist(),
        "calibration_action_std": action_std.tolist(),
        "affine_calibration": bool(affine_calibration),
        "action_adapter": bool(action_adapter),
        "test_episode_count": test_episodes,
        "windows_per_episode": len(_windows(blocks + 1, maximum)),
        "summary": summary,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--environment", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--affine-calibration",
        action="store_true",
        help="fit a diagonal prediction-to-target affine map on disjoint calibration episodes",
    )
    parser.add_argument("--affine-ridge", type=float, default=1e-3)
    parser.add_argument("--action-adapter", action="store_true")
    parser.add_argument("--action-adapter-steps", type=int, default=80)
    parser.add_argument("--action-adapter-learning-rate", type=float, default=5e-2)
    parser.add_argument("--action-adapter-regularization", type=float, default=1e-2)
    args = parser.parse_args()
    horizons = sorted(set(args.horizons))
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must be positive")
    if args.calibration_episodes < 2 or args.test_episodes < 2:
        raise ValueError("calibration and test each require at least two episodes")
    if args.affine_ridge < 0.0:
        raise ValueError("affine-ridge must be non-negative")
    if args.action_adapter_steps < 1:
        raise ValueError("action-adapter-steps must be positive")
    if args.action_adapter_learning_rate <= 0.0 or args.action_adapter_regularization < 0.0:
        raise ValueError("action adapter learning rate must be positive and regularization non-negative")
    device = torch.device(args.device)
    install_missing_field_compat()
    model = load_compatible_pretrained(args.policy, device=device).eval()
    weights = Path(model._checkpoint_compat["weights"])
    reports = [
        _environment_report(
            model, environment=environment, calibration_episodes=args.calibration_episodes,
            test_episodes=args.test_episodes, blocks=args.blocks, action_block=args.action_block,
            horizons=horizons, seed=args.seed + index * 100_000, device=device,
            affine_calibration=args.affine_calibration, affine_ridge=args.affine_ridge,
            action_adapter=args.action_adapter,
            action_adapter_steps=args.action_adapter_steps,
            action_adapter_learning_rate=args.action_adapter_learning_rate,
            action_adapter_regularization=args.action_adapter_regularization,
        )
        for index, environment in enumerate(args.environment)
    ]
    payload = {
        "schema": "frozen_cross_environment_prediction.v1",
        "protocol": {
            "checkpoint": args.policy,
            "checkpoint_weights": str(weights),
            "checkpoint_sha256": _sha256(weights),
            "checkpoint_frozen": True,
            "adapter_training": False,
            "runtime_input": "live rendered RGB observations and environment actions only",
            "runtime_state_access": False,
            "runtime_successor_verification": False,
            "calibration_test_disjoint": True,
            "calibration_role": "per-environment action z-score only; no learned target-domain adapter",
            "context_length": CONTEXT,
            "horizons": horizons,
            "seed": args.seed,
            "device": str(device),
            "render_backend": os.environ.get("MUJOCO_GL"),
            "dmcontrol_missing_field_compat": compatibility_status(),
        },
        "reports": reports,
        "claim_boundary": (
            "This is a same-checkpoint, cross-environment action-conditioned latent-prediction audit. "
            "It does not evaluate goal-conditioned closed-loop planning, physical causal observability, or universal world-model capability."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cross_environment_prediction.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed": [r["environment"] for r in reports if r["status"] == "completed"], "reports": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
