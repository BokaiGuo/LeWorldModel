"""Temporal covariance/operator diagnostic for the public DINO-WM PushT model.

Unlike the PointMaze replay, the public PushT archive exposes train/val
directories directly.  The evaluator therefore uses the released validation
trajectories without reconstructing episode membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Root containing train/ and val/ directories.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dinov2-source", type=Path, default=Path("/tmp/dinov2"))
    return p.parse_args()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, "/tmp/dino_wm")
    original_hub_load = torch.hub.load
    if args.dinov2_source.exists():
        def hub_load(repo_or_dir, model, *extra, **kwargs):
            if repo_or_dir == "facebookresearch/dinov2":
                return original_hub_load(str(args.dinov2_source), model, source="local", *extra, **kwargs)
            return original_hub_load(repo_or_dir, model, *extra, **kwargs)
        torch.hub.load = hub_load

    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import PushTDataset
    from models.dino import DinoV2Encoder
    from models.visual_world_model import VWorldModel

    dataset = PushTDataset(
        data_path=str(args.dataset_root / "val"),
        transform=default_transform(224),
        normalize_action=True,
        with_velocity=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    model = VWorldModel(
        image_size=224,
        num_hist=3,
        num_pred=1,
        encoder=encoder,
        proprio_encoder=checkpoint["proprio_encoder"],
        action_encoder=checkpoint["action_encoder"],
        decoder=None,
        predictor=checkpoint["predictor"],
        proprio_dim=10,
        action_dim=10,
        concat_dim=1,
        num_action_repeat=1,
        num_proprio_repeat=1,
        train_encoder=False,
        train_predictor=False,
        train_decoder=False,
    )
    model.eval()

    horizons = sorted(set(args.horizons))
    max_h = max(horizons)
    context_len = 3
    per_horizon = {str(h): [] for h in horizons}
    predicted_vectors = {str(h): [] for h in horizons}
    target_vectors = {str(h): [] for h in horizons}
    eligible = [
        i for i in range(len(dataset))
        if int(dataset.get_seq_length(i)) >= (context_len + max_h) * 5
    ]
    with torch.no_grad():
        for offset in range(0, len(eligible), args.batch_size):
            episode_batch = eligible[offset : offset + args.batch_size]
            frame_ids = [5 * i for i in range(context_len + max_h)]
            obs_batch, action_batch = [], []
            for episode in episode_batch:
                obs, _, _, _ = dataset.get_frames(episode, frame_ids)
                obs_batch.append(obs)
                action_batch.append(
                    torch.stack([
                        dataset.actions[episode, i * 5 : (i + 1) * 5].reshape(-1)
                        for i in range(context_len + max_h)
                    ])
                )
            obs = {key: torch.stack([item[key] for item in obs_batch]) for key in obs_batch[0]}
            blocks = torch.stack(action_batch)
            obs0 = {key: value[:, :context_len] for key, value in obs.items()}
            rollout_obs, _ = model.rollout(obs0, blocks)
            encoded = model.encode(obs, blocks)
            for h in horizons:
                pred = rollout_obs["visual"][:, context_len - 1 + h].mean(dim=1)
                target = encoded[:, context_len - 1 + h, :, :384].mean(dim=1)
                errors = torch.mean((pred - target) ** 2, dim=1)
                per_horizon[str(h)].extend(errors.tolist())
                predicted_vectors[str(h)].extend(pred.tolist())
                target_vectors[str(h)].extend(target.tolist())

    commutators, null_means = {}, {}
    for h in horizons:
        p = torch.tensor(predicted_vectors[str(h)])
        t = torch.tensor(target_vectors[str(h)])
        x = t - t.mean(0, keepdim=True)
        y = p - p.mean(0, keepdim=True)
        cov = (x.T @ x) / max(1, len(x) - 1)
        op = torch.linalg.lstsq(x, y).solution
        commutators[str(h)] = float(
            torch.linalg.matrix_norm(cov @ op - op @ cov)
            / (torch.linalg.matrix_norm(cov @ op) + 1e-8)
        )
        vals = []
        for seed in range(32):
            g = torch.Generator().manual_seed(seed)
            q, _ = torch.linalg.qr(torch.randn(cov.shape[0], cov.shape[0], generator=g))
            aq = q @ op @ q.T
            vals.append(float(
                torch.linalg.matrix_norm(cov @ aq - aq @ cov)
                / (torch.linalg.matrix_norm(cov @ aq) + 1e-8)
            ))
        null_means[str(h)] = sum(vals) / len(vals)

    train_dataset = PushTDataset(
        data_path=str(args.dataset_root / "train"),
        normalize_action=True,
        with_velocity=True,
    )
    metrics = {
        "latent_prediction_error_by_horizon": {
            h: {"mean": sum(vals) / len(vals), "episodes": len(vals)}
            for h, vals in per_horizon.items()
            if vals
        },
        "covariance_operator_commutator_by_horizon": commutators,
        "spectrum_preserving_null_by_horizon": null_means,
        "protocol": {
            "context_length": context_len,
            "frame_stride": 5,
            "horizons": horizons,
            "split": "public_val_directory",
            "train_test_split_serialized": True,
            "train_episode_count": len(train_dataset),
            "holdout_episode_count": len(eligible),
            "episodes_used": eligible,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "normalization": "Released PushTDataset constants with normalize_action=True.",
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "dataset_archive_sha256": sha256_file(args.dataset_root.parent.parent / "pusht_noise.zip"),
        },
        "evidence_boundary": (
            "Independent DINO-WM checkpoint evaluated on the explicitly released "
            "PushT validation directory; the diagnostic transfers across model "
            "families but does not imply a universal learned-world-model theorem."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
