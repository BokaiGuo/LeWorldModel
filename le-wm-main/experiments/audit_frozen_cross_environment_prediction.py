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
    _calibration_frames, calibration_actions = _collect_split(
        environment, start_seed=seed, episodes=calibration_episodes, blocks=blocks, action_block=action_block,
    )
    test_frames, test_actions = _collect_split(
        environment, start_seed=seed + 1_000_000, episodes=test_episodes, blocks=blocks, action_block=action_block,
    )
    action_mean, action_std = _action_stats(calibration_actions)
    preprocess = get_img_preprocessor("pixels", "pixels", 224)
    rows: list[dict[str, Any]] = []
    vector_store: dict[str, list[np.ndarray]] = {}
    for episode, (frames, actions) in enumerate(zip(test_frames, test_actions)):
        pixel_sequence = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        pixels = preprocess({"pixels": pixel_sequence})["pixels"].unsqueeze(0).to(device)
        standardized = (actions - action_mean) / action_std
        zero_physical = (np.zeros_like(actions) - action_mean) / action_std
        for mode, blocks_array in (("observed", standardized), ("zero_physical", zero_physical)):
            blocks_tensor = torch.from_numpy(blocks_array).unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                encoded = model.encode({"pixels": pixels, "action": blocks_tensor})
                predictions = _predict_horizons(model, encoded["emb"], encoded["act_emb"], horizons, CONTEXT)
            for start in _windows(frames.shape[0], maximum):
                # _predict_horizons assumes context begins at zero.  Slice a
                # legal window to avoid evaluating only initial states.
                if start:
                    with torch.inference_mode():
                        window = model.encode({
                            "pixels": pixels[:, start : start + CONTEXT + maximum],
                            "action": blocks_tensor[:, start : start + CONTEXT + maximum],
                        })
                        predictions = _predict_horizons(model, window["emb"], window["act_emb"], horizons, CONTEXT)
                    embeddings = window["emb"]
                else:
                    embeddings = encoded["emb"]
                for horizon in horizons:
                    target = embeddings[:, CONTEXT - 1 + horizon].detach().cpu().numpy()[0]
                    prediction = predictions[horizon].numpy()[0]
                    key = f"{mode}/h{horizon}"
                    vector_store.setdefault(f"{key}/target", []).append(target)
                    vector_store.setdefault(f"{key}/prediction", []).append(prediction)
                    rows.append({
                        "episode": episode, "window_start": start, "mode": mode, "horizon": horizon,
                        "mse": float(np.mean(np.square(prediction - target))),
                    })
    summary: dict[str, Any] = {"by_mode_horizon": {}, "paired_action_effect": {}}
    for horizon in horizons:
        observed = np.asarray([row["mse"] for row in rows if row["mode"] == "observed" and row["horizon"] == horizon])
        zero = np.asarray([row["mse"] for row in rows if row["mode"] == "zero_physical" and row["horizon"] == horizon])
        # Average windows within episode before uncertainty estimation.
        obs_episode = np.asarray([observed[np.asarray([r["episode"] for r in rows if r["mode"] == "observed" and r["horizon"] == horizon]) == ep].mean() for ep in range(test_episodes)])
        zero_episode = np.asarray([zero[np.asarray([r["episode"] for r in rows if r["mode"] == "zero_physical" and r["horizon"] == horizon]) == ep].mean() for ep in range(test_episodes)])
        for mode, episode_values in (("observed", obs_episode), ("zero_physical", zero_episode)):
            key = f"{mode}/h{horizon}"
            target = np.stack(vector_store[f"{key}/target"])
            prediction = np.stack(vector_store[f"{key}/prediction"])
            summary["by_mode_horizon"][key] = {
                "mse": _bootstrap(episode_values, seed + horizon + (0 if mode == "observed" else 100)),
                "relative_commutator": _relative_commutator(target, prediction),
            }
        summary["paired_action_effect"][f"h{horizon}"] = {
            "zero_minus_observed_mse": _bootstrap(zero_episode - obs_episode, seed + 200 + horizon),
            "fraction_zero_worse": float(np.mean(zero_episode > obs_episode)),
        }
    return {
        "environment": environment,
        "status": "completed",
        "raw_action_dim": raw_dim,
        "action_block": action_block,
        "calibration_action_mean": action_mean.tolist(),
        "calibration_action_std": action_std.tolist(),
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
    args = parser.parse_args()
    horizons = sorted(set(args.horizons))
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must be positive")
    if args.calibration_episodes < 2 or args.test_episodes < 2:
        raise ValueError("calibration and test each require at least two episodes")
    device = torch.device(args.device)
    model = load_compatible_pretrained(args.policy, device=device).eval()
    weights = Path(model._checkpoint_compat["weights"])
    reports = [
        _environment_report(
            model, environment=environment, calibration_episodes=args.calibration_episodes,
            test_episodes=args.test_episodes, blocks=args.blocks, action_block=args.action_block,
            horizons=horizons, seed=args.seed + index * 100_000, device=device,
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
