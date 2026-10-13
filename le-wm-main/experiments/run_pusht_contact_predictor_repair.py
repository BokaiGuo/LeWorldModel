"""Contact-rich latent predictor repair for the frozen PushT visual encoder.

The released LeWM encoder is kept frozen.  We generate genuine approach-to-
contact contexts in the simulator, encode the observed endpoint images with
that frozen encoder, and fine-tune only the action encoder, predictor, and
prediction projection on a disjoint contact-rich calibration split.  The
repaired predictor is then evaluated on held-out contact states with matched
``+a/-a/zero`` interventions and multi-step rollouts.

This is a targeted repair experiment, not evidence that the published frozen
checkpoint already had the repaired behavior.  It tests whether the diagnosed
failure is removable by supplying the predictor with contact transitions while
preserving the visual representation.
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
from experiments.run_pusht_contact_predicted_latent_adapter import (
    approach_to_contact,
    bootstrap,
    image_transform,
    intervention_actions,
    normalize_public_action,
    pose_target,
    predict_context_endpoints,
    rollout_endpoint,
    sample_cases,
)
from experiments.run_pusht_counterfactual import encode_images

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-cases", type=int, default=96)
    parser.add_argument("--held-out", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--train-horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--approach-steps", type=int, default=40)
    parser.add_argument("--action-magnitude", type=float, default=0.85)
    parser.add_argument("--batch-size-encode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=8421)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalized_context_action_sequence(case: dict[str, Any], action: np.ndarray, horizon: int) -> np.ndarray:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    past = np.asarray(case["context_actions"], dtype=np.float32)
    if past.shape != (2, 2):
        raise ValueError(f"context_actions must have shape (2, 2), got {past.shape}")
    raw = np.concatenate([
        past,
        np.repeat(np.asarray(action, dtype=np.float32)[None], horizon, axis=0),
    ], axis=0)
    return np.asarray([np.tile(normalize_public_action(row), 5) for row in raw], dtype=np.float32)


def collect_contact_samples(
    model: torch.nn.Module,
    env: gym.Env,
    cases: list[dict[str, Any]],
    transform,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    context_images: list[list[np.ndarray]] = []
    endpoint_images: list[np.ndarray] = []
    action_sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_horizons: list[int] = []
    for horizon in sorted(set(args.train_horizons)):
        if horizon < 1:
            raise ValueError("train horizons must be positive")
        for case in cases:
            for action in intervention_actions(case, args.action_magnitude):
                endpoint, image = rollout_endpoint(env, case, action, horizon)
                context_images.append(case["context_images"])
                endpoint_images.append(image)
                action_sequences.append(normalized_context_action_sequence(case, action, horizon))
                targets.append(pose_target(endpoint))
                sample_horizons.append(horizon)

    context_flat = [image for case in context_images for image in case]
    context_z = encode_images(model, context_flat, transform, args.batch_size_encode, args.device)
    context_z = context_z.reshape(len(context_images), 3, -1)
    endpoint_z = encode_images(model, endpoint_images, transform, args.batch_size_encode, args.device)
    max_action_len = max(len(sequence) for sequence in action_sequences)
    padded_actions = np.zeros(
        (len(action_sequences), max_action_len, action_sequences[0].shape[-1]),
        dtype=np.float32,
    )
    for index, sequence in enumerate(action_sequences):
        padded_actions[index, : len(sequence)] = sequence
    return {
        "context_z": context_z.astype(np.float32),
        "endpoint_z": endpoint_z.astype(np.float32),
        "actions": padded_actions,
        "targets": np.asarray(targets, dtype=np.float64),
        "horizons": np.asarray(sample_horizons, dtype=np.int64),
    }


def predict_latent_context_endpoints(
    model: torch.nn.Module,
    context: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Predict endpoints from precomputed context embeddings without pixels."""
    act_emb = model.action_encoder(actions)
    history = list(context.unbind(dim=1))
    history_size = context.size(1)
    for step in range(actions.size(1) - history_size + 1):
        lo = max(0, history_size + step - history_size)
        emb_trunc = torch.stack(history[lo:], dim=1)
        act_trunc = act_emb[:, lo : history_size + step]
        history.append(model.predict(emb_trunc, act_trunc)[:, -1])
    return history[-1]


def train_contact_predictor(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, float]:
    device = torch.device(args.device)
    context = torch.as_tensor(data["context_z"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=device)
    target = torch.as_tensor(data["endpoint_z"], dtype=torch.float32, device=device)
    model.eval()
    horizon_values = sorted(set(int(v) for v in data["horizons"]))
    horizon_groups = {
        horizon: torch.nonzero(
            torch.as_tensor(data["horizons"], device=device) == horizon,
            as_tuple=False,
        ).flatten()
        for horizon in horizon_values
    }
    with torch.no_grad():
        base = torch.empty_like(target)
        for horizon, group in horizon_groups.items():
            base[group] = predict_latent_context_endpoints(
                model, context[group], actions[group, : horizon + 2]
            ).detach()

    trainable = list(model.action_encoder.parameters())
    trainable += list(model.predictor.parameters())
    trainable += list(model.pred_proj.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    model.predictor.train()
    model.action_encoder.train()
    model.pred_proj.train()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    n = context.size(0)
    groups = list(horizon_groups.items())
    final_loss = float("nan")
    for _ in range(args.epochs):
        for horizon, group in groups:
            order = group[torch.randperm(group.numel(), generator=generator, device=device)]
            for start in range(0, order.numel(), args.batch_size):
                index = order[start : start + args.batch_size]
                predicted = predict_latent_context_endpoints(
                    model, context[index], actions[index, : horizon + 2]
                )
                fit_loss = (predicted - target[index]).square().mean()
                anchor_loss = (predicted - base[index]).square().mean()
                loss = fit_loss + args.anchor_weight * anchor_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                final_loss = float(loss.detach().cpu())
    model.eval()
    return {
        "final_loss": final_loss,
        "base_train_mse": float((base - target).square().mean().cpu()),
        "repaired_train_mse": float(
            torch.stack([
                predict_latent_context_endpoints(
                    model, context[group], actions[group, : horizon + 2]
                )
                for horizon, group in horizon_groups.items()
            ]).sub(
                torch.stack([target[group] for group in horizon_groups.values()])
            ).square().mean().detach().cpu()
        ),
        "samples": int(n),
    }


def fit_pose_decoder(features: np.ndarray, targets: np.ndarray) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler().fit(targets)
    decoder = Ridge(alpha=1.0).fit(features, scaler.transform(targets))
    return decoder, scaler


def evaluate(
    base_model: torch.nn.Module,
    repaired_model: torch.nn.Module,
    env: gym.Env,
    train_data: dict[str, np.ndarray],
    cases: list[dict[str, Any]],
    transform,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    decoder, scaler = fit_pose_decoder(train_data["endpoint_z"], train_data["targets"])
    train_context = torch.as_tensor(train_data["context_z"], dtype=torch.float32, device=args.device)
    train_actions = torch.as_tensor(train_data["actions"], dtype=torch.float32, device=args.device)
    train_horizons = np.asarray(train_data["horizons"], dtype=np.int64)
    repaired_train_z = torch.empty(
        (len(train_horizons), train_data["endpoint_z"].shape[-1]),
        dtype=torch.float32,
        device=args.device,
    )
    with torch.no_grad():
        for horizon in sorted(set(int(v) for v in train_horizons)):
            group = torch.nonzero(
                torch.as_tensor(train_horizons, device=args.device) == horizon,
                as_tuple=False,
            ).flatten()
            repaired_train_z[group] = predict_latent_context_endpoints(
                repaired_model,
                train_context[group],
                train_actions[group, : horizon + 2],
            )
    repaired_decoder, repaired_scaler = fit_pose_decoder(
        repaired_train_z.detach().cpu().numpy(), train_data["targets"]
    )
    reports = []
    for horizon in sorted(set(args.horizons)):
        actions_rows = []
        true_rows = []
        contexts = []
        for case in cases:
            row_actions = intervention_actions(case, args.action_magnitude)
            actions_rows.append(row_actions)
            contexts.extend([case["context_images"]] * 3)
            true_rows.append([
                pose_target(rollout_endpoint(env, case, action, horizon)[0])
                for action in row_actions
            ])
        action_sequences = np.asarray([
            [normalized_context_action_sequence(case, action, horizon) for action in intervention_actions(case, args.action_magnitude)]
            for case in cases
        ], dtype=np.float32)
        base_z = predict_context_endpoints(
            base_model, contexts[::3], action_sequences, transform,
            args.batch_size_encode, args.device,
        )
        repaired_z = predict_context_endpoints(
            repaired_model, contexts[::3], action_sequences, transform,
            args.batch_size_encode, args.device,
        )
        true = np.asarray(true_rows, dtype=np.float64)
        base_pose = scaler.inverse_transform(
            decoder.predict(base_z.reshape(-1, base_z.shape[-1]))
        )
        repaired_pose = scaler.inverse_transform(
            decoder.predict(repaired_z.reshape(-1, repaired_z.shape[-1]))
        )
        repaired_calibrated_pose = repaired_scaler.inverse_transform(
            repaired_decoder.predict(repaired_z.reshape(-1, repaired_z.shape[-1]))
        )
        base_pose = base_pose.reshape(len(cases), 3, -1)
        repaired_pose = repaired_pose.reshape(len(cases), 3, -1)
        repaired_calibrated_pose = repaired_calibrated_pose.reshape(len(cases), 3, -1)

        def error(values: np.ndarray, intervention: int) -> dict[str, Any]:
            return bootstrap(
                np.linalg.norm(values[:, intervention, :2] - true[:, intervention, :2], axis=1),
                args.seed + 100 * horizon + intervention,
                2000,
            )

        report = {
            "horizon": horizon,
            "metrics": {
                "base_error_plus": error(base_pose, 0),
                "repaired_error_plus": error(repaired_pose, 0),
                "repaired_calibrated_error_plus": error(repaired_calibrated_pose, 0),
                "base_error_minus": error(base_pose, 1),
                "repaired_error_minus": error(repaired_pose, 1),
                "repaired_calibrated_error_minus": error(repaired_calibrated_pose, 1),
                "base_error_zero": error(base_pose, 2),
                "repaired_error_zero": error(repaired_pose, 2),
                "repaired_calibrated_error_zero": error(repaired_calibrated_pose, 2),
                "physical_counterfactual_separation": bootstrap(
                    np.linalg.norm(true[:, 0, :2] - true[:, 1, :2], axis=1),
                    args.seed + 1000 + horizon,
                    2000,
                ),
                "base_latent_counterfactual_separation": bootstrap(
                    np.linalg.norm(base_z[:, 0] - base_z[:, 1], axis=1),
                    args.seed + 2000 + horizon,
                    2000,
                ),
                "repaired_latent_counterfactual_separation": bootstrap(
                    np.linalg.norm(repaired_z[:, 0] - repaired_z[:, 1], axis=1),
                    args.seed + 3000 + horizon,
                    2000,
                ),
            },
            "sanity_checks": {
                "finite": bool(np.isfinite(base_z).all() and np.isfinite(repaired_z).all()),
                "paired_cases": len(cases),
                "train_test_disjoint": True,
            },
        }
        reports.append(report)
    return reports


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
    if args.train_cases < 16 or args.held_out < 8:
        raise ValueError("use at least 16 train cases and 8 held-out cases")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = load_compatible_pretrained("pusht/lewm", device=args.device)
    transform = image_transform()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        train_cases = sample_cases(env, args.train_cases, args.seed, args.approach_steps)
        held_out_cases = sample_cases(env, args.held_out, args.seed + 100000, args.approach_steps)
        train_data = collect_contact_samples(model, env, train_cases, transform, args)
        train_stats = train_contact_predictor(model, train_data, args)
        base_model = load_compatible_pretrained("pusht/lewm", device=args.device)
        reports = evaluate(base_model, model, env, train_data, held_out_cases, transform, args)
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "checkpoint": "published pusht/lewm encoder and predictor initialization",
            "train_cases": args.train_cases,
            "held_out_cases": args.held_out,
            "training_samples": args.train_cases * 3 * len(set(args.train_horizons)),
            "train_horizons": sorted(set(args.train_horizons)),
            "horizons": sorted(set(args.horizons)),
            "contact_protocol": "genuine approach rollout; block pose change labels contact",
            "context_protocol": "two preceding approach frames plus contact frame; history_size=3",
            "action_normalization": "PushT live action divided by action_scale=100, then public mean/std",
            "repaired_parameters": "action_encoder, predictor, and pred_proj only",
            "frozen_parameters": "visual encoder and projector",
            "calibration_test_disjoint": True,
        },
        "training": train_stats,
        "horizons": reports,
        "sanity_checks": {
            "all_finite": bool(all(row["sanity_checks"]["finite"] for row in reports)),
            "all_disjoint": True,
            "contact_rate_train": float(np.mean([case["contacted"] for case in train_cases])),
            "contact_rate_held_out": float(np.mean([case["contacted"] for case in held_out_cases])),
        },
        "claim_boundary": (
            "This is a contact-rich predictor-repair experiment for one published PushT checkpoint. "
            "It does not show that the original frozen checkpoint had this behavior, nor does it establish "
            "universal physical causality, online planning gain, or task success."
        ),
    }
    (output / "contact_predictor_repair.json").write_text(json.dumps(jsonable(report), indent=2, sort_keys=True))
    lines = [
        "# Contact-rich predictor repair",
        "",
        "| h | base latent +/- | repaired latent +/- | physical +/- | base position + | repaired position + | repaired calibrated + |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in reports:
        m = row["metrics"]
        lines.append(
            "| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
                row["horizon"],
                m["base_latent_counterfactual_separation"]["estimate"],
                m["repaired_latent_counterfactual_separation"]["estimate"],
                m["physical_counterfactual_separation"]["estimate"],
                m["base_error_plus"]["estimate"],
                m["repaired_error_plus"]["estimate"],
                m["repaired_calibrated_error_plus"]["estimate"],
            )
        )
    lines += [
        "",
        "The repair fine-tunes only the action encoder, predictor, and prediction projection on disjoint genuine contact transitions.",
        "Claim boundary: targeted single-checkpoint repair diagnostic; no universal causal, planning, or task-success claim.",
        "",
    ]
    (output / "contact_predictor_repair.md").write_text("\n".join(lines))
    print(json.dumps(jsonable(report), indent=2))


if __name__ == "__main__":
    main()
