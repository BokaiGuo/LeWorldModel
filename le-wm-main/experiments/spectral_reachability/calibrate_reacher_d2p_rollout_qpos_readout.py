"""Calibrate a frozen qpos chart for one-step *predicted* Reacher latents.

The RGB encoder and the rollout predictor are distinct frozen maps.  This
script deliberately fits only a closed-form readout from the latter's one-step
outputs to source qpos labels, on episode-disjoint source data.  It never
updates the world model and refuses a CPU execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.calibrate_reacher_d2p_qpos_readout import _metrics, fit_ridge_angle_readout
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import _normalized_action_bounds
from experiments.spectral_reachability.reacher_d2p_packet_support import receipt_source
from experiments.spectral_reachability.reacher_d2p_qpos_readout import PREDICTION_READOUT_TASK


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_rows(lengths: np.ndarray, offsets: np.ndarray, episodes: np.ndarray, *, per_episode: int, rng: np.random.Generator) -> np.ndarray:
    """Rows t with three causal RGB anchors and a complete five-step successor."""

    selected: list[np.ndarray] = []
    for episode in np.asarray(episodes, dtype=np.int64):
        start, length = int(offsets[episode]), int(lengths[episode])
        # Need RGB at t-10,t-5,t, the *causal* two preceding action blocks,
        # and the packet t:t+5.  The earlier version began at t-10 and then
        # duplicated the predicted packet inside the history, shifting the
        # predictor's action slots by one block.
        candidates = np.arange(start + 15, start + length - 5, dtype=np.int64)
        if len(candidates) < per_episode:
            raise ValueError("selected official Reacher episode is too short for rollout-readout calibration")
        selected.append(np.sort(rng.choice(candidates, size=per_episode, replace=False)))
    return np.concatenate(selected)


def _raw_samples(handle: h5py.File, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load causal RGB/action windows and labels without crossing episodes."""

    pixels: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    future_actions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for row in np.asarray(rows, dtype=np.int64):
        pixels.append(np.asarray(handle["pixels"][[row - 10, row - 5, row]], dtype=np.uint8))
        # Runtime planning observes frames t-10,t-5,t and retains three
        # historical blocks ending at t.  Its terminal adapter then discards
        # the oldest one and appends the proposed packet a[t:t+5].
        actions.append(np.asarray(handle["action"][row - 15:row], dtype=np.float32).reshape(3, -1))
        future_actions.append(np.asarray(handle["action"][row:row + 5], dtype=np.float32).reshape(-1))
        targets.append(np.asarray(handle["qpos"][row + 5], dtype=np.float64)[:2])
    return np.stack(pixels), np.stack(actions), np.stack(future_actions), np.stack(targets)


def _params_from_raw(
    raw: np.ndarray, *, mean: np.ndarray, std: np.ndarray, lower: torch.Tensor, upper: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    values = torch.as_tensor(raw, device=device, dtype=torch.float32)
    normalized = (values - torch.as_tensor(mean, device=device)) / torch.as_tensor(std, device=device)
    coordinate = (2.0 * normalized - (upper + lower)) / (upper - lower)
    if not torch.isfinite(coordinate).all() or torch.any(coordinate <= -1.0) or torch.any(coordinate >= 1.0):
        raise ValueError("official source successor packet is outside the deployed bounded action chart")
    return torch.atanh(coordinate)


def _predicted_latents(
    model: torch.nn.Module, preprocessor: _Preprocessor, pixels: np.ndarray, history_raw: np.ndarray, future_raw: np.ndarray,
    *, mean: np.ndarray, std: np.ndarray, lower: torch.Tensor, upper: torch.Tensor, device: torch.device, batch_size: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, len(pixels), batch_size):
        stop = min(start + batch_size, len(pixels))
        packed = preprocessor._prepare_info({
            "pixels": np.ascontiguousarray(pixels[start:stop]),
            "action": np.ascontiguousarray(history_raw[start:stop]),
        })
        image = packed["pixels"].to(device)
        history = packed["action"].to(device)
        params = _params_from_raw(future_raw[start:stop, None], mean=mean, std=std, lower=lower, upper=upper, device=device)
        # ``p0._terminal_batch`` deliberately represents one observed root and
        # a bank of candidate actions.  Calibration has the opposite layout:
        # independent source roots paired one-to-one with their recorded next
        # packet.  Keep those roots independent instead of accidentally
        # treating batch members as competing candidates.
        with torch.no_grad():
            encoded = p0._latent_context(model, image, history)
            future = p0._bounded_normalized_actions(params, lower, upper)
            action_context = torch.cat((history[:, 1:], future), dim=1)
            terminal = model.predict(encoded, model.action_encoder(action_context))[:, -1]
        values.append(terminal.detach().cpu().numpy().astype(np.float64, copy=False))
    return np.concatenate(values, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026102580)
    parser.add_argument("--train-episodes", type=int, default=96)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--samples-per-episode", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Reacher rollout qpos-readout calibration requires host CUDA exactly; CPU fallback is forbidden")
    if min(args.train_episodes, args.validation_episodes, args.samples_per_episode, args.batch_size) < 1 or args.ridge_alpha < 0.0:
        raise ValueError("rollout qpos-readout sizes must be positive and ridge alpha nonnegative")
    output, policy, adapter = args.output.resolve(), args.policy.resolve(), args.adapter_artifact.resolve()
    if output.exists():
        raise FileExistsError("rollout qpos-readout output directory must be new")
    dataset, dataset_sha256 = receipt_source(adapter)
    install_missing_field_compat(); device = torch.device("cuda"); torch.empty(0, device=device)
    mean, std = _load_action_stats(adapter, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(policy, device=device).eval(); model.requires_grad_(False); model.interpolate_pos_encoding = True
    # Reacher's raw environment Box is [-1, 1]^2; only its flattened,
    # normalized five-step tokens enter the frozen predictor.
    lower_np, upper_np = _normalized_action_bounds(mean, std, np.full(2, -1.0, dtype=np.float32), np.full(2, 1.0, dtype=np.float32))
    lower, upper = torch.as_tensor(lower_np, device=device), torch.as_tensor(upper_np, device=device)
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    rng = np.random.default_rng(args.seed)
    with h5py.File(dataset, "r", swmr=True) as handle:
        lengths, offsets = np.asarray(handle["ep_len"]), np.asarray(handle["ep_offset"])
        count = int(args.train_episodes + args.validation_episodes)
        if len(lengths) < count:
            raise ValueError("official Reacher source has too few episodes for an episode-disjoint rollout split")
        episodes = rng.choice(len(lengths), size=count, replace=False)
        train_episodes, validation_episodes = np.sort(episodes[:args.train_episodes]), np.sort(episodes[args.train_episodes:])
        train_rows = _episode_rows(lengths, offsets, train_episodes, per_episode=args.samples_per_episode, rng=rng)
        validation_rows = _episode_rows(lengths, offsets, validation_episodes, per_episode=args.samples_per_episode, rng=rng)
        train_pixels, train_history, train_future, train_qpos = _raw_samples(handle, train_rows)
        validation_pixels, validation_history, validation_future, validation_qpos = _raw_samples(handle, validation_rows)
    train_latents = _predicted_latents(model, preprocessor, train_pixels, train_history, train_future, mean=mean, std=std, lower=lower, upper=upper, device=device, batch_size=args.batch_size)
    validation_latents = _predicted_latents(model, preprocessor, validation_pixels, validation_history, validation_future, mean=mean, std=std, lower=lower, upper=upper, device=device, batch_size=args.batch_size)
    artifact = fit_ridge_angle_readout(train_latents, train_qpos, alpha=args.ridge_alpha)
    from experiments.spectral_reachability.calibrate_reacher_d2p_qpos_readout import decode_qpos
    validation = _metrics(decode_qpos(artifact, validation_latents), validation_qpos)
    gate = bool(validation["mean_joint_wrapped_abs_error"] <= 0.35 and validation["qpos_distance_rmse"] <= 0.55)
    output.mkdir(parents=True)
    artifact_path = output / "predicted_qpos_readout.npz"
    np.savez_compressed(artifact_path, **artifact, task=np.asarray([PREDICTION_READOUT_TASK]), dataset_sha256=np.asarray([dataset_sha256]))
    report = {
        "schema": "reacher_d2p_frozen_rollout_qpos_readout.v1", "status": "completed",
        "scope": "episode-disjoint official-source calibration of a one-step frozen predictor latent coordinate; no world-model parameter is updated",
        "inputs": {"policy": str(policy), "weights_sha256": _sha256(policy / "weights.pt"), "adapter_artifact": str(adapter), "adapter_sha256": _sha256(adapter), "official_dataset": str(dataset), "official_dataset_sha256": dataset_sha256},
        "protocol": {"device": str(device), "seed": args.seed, "train_episode_count": int(len(train_episodes)), "validation_episode_count": int(len(validation_episodes)), "samples_per_episode": int(args.samples_per_episode), "ridge_alpha": args.ridge_alpha, "target": "[sin(qpos_0), sin(qpos_1), cos(qpos_0), cos(qpos_1)]", "prediction_horizon": 1, "episode_disjoint": True, "declared_validation_gate": {"mean_joint_wrapped_abs_error_max": 0.35, "qpos_distance_rmse_max": 0.55}},
        "split": {"train_episode_ids_sha256": hashlib.sha256(train_episodes.tobytes()).hexdigest(), "validation_episode_ids_sha256": hashlib.sha256(validation_episodes.tobytes()).hexdigest(), "train_row_count": int(len(train_rows)), "validation_row_count": int(len(validation_rows))},
        "validation": {**validation, "gate_passed": gate}, "artifact": {"path": str(artifact_path), "sha256": _sha256(artifact_path)},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["validation"], sort_keys=True))


if __name__ == "__main__":
    main()
