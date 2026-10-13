"""Held-out physical counterfactual test for the frozen LeWM checkpoint.

The simulator supplies the ground-truth endpoint for the same initial state
under multiple action sequences.  A frozen LeWM predicts endpoint embeddings;
a state decoder is fit only on disjoint calibration rollouts and is evaluated
on held-out intervention pairs.  The report contains both latent prediction
error and decoded physical-pose error, plus a no-action/constant-state
baseline.  This is a PushT checkpoint-and-simulator result, not a universal
world-model claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_pretraining as spt
import stable_worldmodel  # noqa: F401
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.checkpoint_compat import load_compatible_pretrained

PUBLIC_ACTION_MEAN = np.asarray([-0.0087, 0.0068], dtype=np.float32)
PUBLIC_ACTION_STD = np.asarray([0.2019, 0.2002], dtype=np.float32)


def normalize_public_action(action: np.ndarray) -> np.ndarray:
    # PushT receives a normalized relative action but internally scales it by
    # ``action_scale=100`` before applying the position controller.  The
    # released LeWM loader standardizes the corresponding raw action after
    # dividing by 100.  Counterfactual experiments must therefore apply the
    # same conversion before the public mean/std normalization.
    scaled = np.asarray(action, dtype=np.float32) / 100.0
    return (scaled - PUBLIC_ACTION_MEAN) / PUBLIC_ACTION_STD


def normalized_action_sequence(action: np.ndarray, horizon: int) -> np.ndarray:
    """Encode one live action as the model's repeated 5-frame public input."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return np.tile(np.tile(normalize_public_action(action), 5), (horizon, 1)).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--calibration", type=int, default=32)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--seed", type=int, default=4072)
    p.add_argument("--action-magnitude", type=float, default=0.55)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def pose_target(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    return np.asarray([state[2], state[3], np.sin(state[4]), np.cos(state[4])], dtype=np.float64)


def reset_state(env, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs, info = env.reset(seed=seed)
    return np.asarray(obs["state"], dtype=np.float64).copy(), np.asarray(info["goal_state"], dtype=np.float64).copy(), env.render().copy()


def rollout(env, state: np.ndarray, goal: np.ndarray, action: np.ndarray, horizon: int, seed: int) -> dict:
    _, info = env.reset(seed=seed, options={"state": state.copy(), "goal_state": goal.copy()})
    initial_image = env.render().copy()
    last_info = info
    for _ in range(horizon):
        _, _, terminated, truncated, last_info = env.step(action.astype(np.float32, copy=False))
        if terminated or truncated:
            break
    final_state = np.asarray(last_info["goal_state"], dtype=np.float64).copy()
    # PushT exposes the actual current state in the raw observation, not info.
    final_state = np.asarray(env.unwrapped._get_obs(), dtype=np.float64).copy()
    return {"initial_image": initial_image, "final_image": env.render().copy(), "final_state": final_state}


def image_transform():
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])


@torch.inference_mode()
def encode_images(model, images: list[np.ndarray], transform, batch_size: int, device: str) -> np.ndarray:
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([
            transform(torch.from_numpy(image).permute(2, 0, 1)) for image in images[start : start + batch_size]
        ]).unsqueeze(1).to(device)
        outputs.append(model.encode({"pixels": batch})["emb"][:, 0].detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch.inference_mode()
def predict_endpoints(model, images: list[np.ndarray], actions: np.ndarray, transform, batch_size: int, device: str) -> np.ndarray:
    outputs = []
    for start in range(0, len(images), batch_size):
        current = images[start : start + batch_size]
        act = torch.as_tensor(actions[start : start + batch_size], dtype=torch.float32, device=device)
        b, s = act.shape[:2]
        pixels = torch.stack([
            transform(torch.from_numpy(image).permute(2, 0, 1)) for image in current
        ]).to(device)
        pixels = pixels[:, None, None].expand(b, s, 1, *pixels.shape[1:])
        out = model.rollout({"pixels": pixels}, act)
        outputs.append(out["predicted_emb"][:, :, -1].detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def bootstrap(values: np.ndarray, seed: int, draws: int = 2000) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(1)
    return {"estimate": float(values.mean()), "ci95_low": float(np.quantile(means, .025)), "ci95_high": float(np.quantile(means, .975)), "n": int(len(values))}


def main() -> None:
    args = parse_args()
    if args.pairs < 4 or args.calibration < 8:
        raise ValueError("use at least 4 held-out pairs and 8 calibration rollouts")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    transform = image_transform()
    model = load_compatible_pretrained("pusht/lewm", device=args.device)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    calibration_images, calibration_targets = [], []
    test_initial_images, test_final_images, test_actions, test_targets, test_initial_targets = [], [], [], [], []
    try:
        for i in range(args.calibration):
            seed = args.seed + i
            state, goal, _ = reset_state(env, seed)
            direction = rng.normal(size=2); direction /= max(np.linalg.norm(direction), 1e-12)
            action = (args.action_magnitude * direction).astype(np.float32)
            result = rollout(env, state, goal, action, args.horizon, seed)
            calibration_images.append(result["final_image"])
            calibration_targets.append(pose_target(result["final_state"]))
        for i in range(args.pairs):
            seed = args.seed + 10000 + i
            state, goal, initial_image = reset_state(env, seed)
            direction = rng.normal(size=2); direction /= max(np.linalg.norm(direction), 1e-12)
            action = (args.action_magnitude * direction).astype(np.float32)
            pair_actions = np.stack([
                normalized_action_sequence(action, args.horizon),
                normalized_action_sequence(-action, args.horizon),
                normalized_action_sequence(np.zeros(2, dtype=np.float32), args.horizon),
            ], axis=0)
            endpoints = []
            for intervention in (action, -action, np.zeros(2, dtype=np.float32)):
                endpoints.append(rollout(env, state, goal, intervention, args.horizon, seed))
            test_initial_images.append(initial_image)
            test_final_images.extend([x["final_image"] for x in endpoints])
            test_actions.append(pair_actions)
            test_targets.append(np.stack([pose_target(x["final_state"]) for x in endpoints]))
            test_initial_targets.append(pose_target(state))
    finally:
        env.close()

    calibration_z = encode_images(model, calibration_images, transform, args.batch_size, args.device)
    test_initial_z = encode_images(model, test_initial_images, transform, args.batch_size, args.device)
    predicted_z = predict_endpoints(model, test_initial_images, np.stack(test_actions), transform, args.batch_size, args.device)
    actual_z = encode_images(model, test_final_images, transform, args.batch_size, args.device).reshape(args.pairs, 3, -1)

    target_scaler = StandardScaler().fit(np.asarray(calibration_targets))
    # Decode predicted embeddings with a separate linear readout fitted on calibration embeddings.
    decoder = Ridge(alpha=1.0).fit(calibration_z, target_scaler.transform(calibration_targets))
    pred_pose_scaled = decoder.predict(predicted_z.reshape(-1, predicted_z.shape[-1]))
    pred_pose = target_scaler.inverse_transform(pred_pose_scaled).reshape(args.pairs, 3, 4)
    true_pose = np.asarray(test_targets)
    baseline_pose = np.asarray(test_initial_targets)[:, None, :]
    pred_position_error = np.linalg.norm((pred_pose[:, :, :2] - true_pose[:, :, :2]), axis=-1)
    baseline_position_error = np.linalg.norm((baseline_pose[:, :, :2] - true_pose[:, :, :2]), axis=-1)
    pred_latent_error = np.linalg.norm(predicted_z - actual_z, axis=-1)
    latent_cf_distance = np.linalg.norm(predicted_z[:, 0] - predicted_z[:, 1], axis=-1)
    physical_cf_distance = np.linalg.norm(true_pose[:, 0, :2] - true_pose[:, 1, :2], axis=-1)

    def summarize(values):
        return bootstrap(np.asarray(values), args.seed + 791)

    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "checkpoint": str(getattr(model, "_checkpoint_compat", {})),
            "calibration_rollouts": args.calibration,
            "held_out_pairs": args.pairs,
            "horizon": args.horizon,
            "action_protocol": "same seeded simulator variation/state/goal; +a, -a, and zero action sequence",
            "action_normalization": {"raw_action_divisor": 100.0, "mean": PUBLIC_ACTION_MEAN.tolist(), "std": PUBLIC_ACTION_STD.tolist(), "source": "public PushT loader constants; live env action is divided by PushT action_scale=100 before standardization"},
            "decoder": "Ridge fit on disjoint calibration endpoint images; held-out test only",
        },
        "metrics": {
            "latent_counterfactual_separation": summarize(latent_cf_distance),
            "physical_counterfactual_xy_separation": summarize(physical_cf_distance),
            "latent_endpoint_prediction_error": summarize(pred_latent_error.reshape(-1)),
            "decoded_position_error_plus": summarize(pred_position_error[:, 0]),
            "decoded_position_error_minus": summarize(pred_position_error[:, 1]),
            "decoded_position_error_zero": summarize(pred_position_error[:, 2]),
            "constant_initial_pose_error_plus": summarize(baseline_position_error[:, 0]),
            "constant_initial_pose_error_minus": summarize(baseline_position_error[:, 1]),
            "constant_initial_pose_error_zero": summarize(baseline_position_error[:, 2]),
            "decoded_minus_baseline_plus": summarize(pred_position_error[:, 0] - baseline_position_error[:, 0]),
            "decoded_minus_baseline_minus": summarize(pred_position_error[:, 1] - baseline_position_error[:, 1]),
            "decoded_minus_baseline_zero": summarize(pred_position_error[:, 2] - baseline_position_error[:, 2]),
        },
        "sanity_checks": {
            "finite_predictions": bool(np.isfinite(predicted_z).all() and np.isfinite(pred_pose).all()),
            "calibration_disjoint_from_test": True,
            "paired_rows": len(test_targets) == args.pairs,
            "checkpoint_key_remap_count": int(getattr(model, "_checkpoint_compat", {}).get("remapped_keys", -1)),
        },
    }
    (out / "counterfactual.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
    (out / "counterfactual.md").write_text("# PushT held-out physical counterfactual\n\n" + json.dumps(report["metrics"], indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "metrics": report["metrics"], "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
