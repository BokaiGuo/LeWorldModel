"""Independent action-conditioned temporal diagnostic for the public DINO-WM.

The public archive omits a serialized split manifest.  The released loader does,
however, specify ``split_ratio=0.9`` and uses ``random_split_traj(..., seed=42)``.
This evaluator reproduces that episode-level split from the released dataset
ordering and records the provenance explicitly as *code-reconstructed*, rather
than claiming a separately serialized manifest.
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
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split-ratio", type=float, default=0.9)
    p.add_argument("--split-mode", choices=["code_reconstructed", "tail"], default="code_reconstructed")
    p.add_argument("--holdout-start", type=int, default=1800,
                   help="Only used with --split-mode tail.")
    p.add_argument("--holdout-end", type=int, default=2000,
                   help="Only used with --split-mode tail.")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--dinov2-source", type=Path, default=Path("/tmp/dinov2"))
    p.add_argument("--manifest-only", action="store_true")
    return p.parse_args()


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
    from datasets.point_maze_dset import PointMazeDataset
    from models.dino import DinoV2Encoder
    from models.visual_world_model import VWorldModel

    root = args.repo_root
    dataset = PointMazeDataset(
        data_path=str(args.dataset_root),
        transform=default_transform(224),
        # Match the released DINO-WM train.yaml: normalization is computed by
        # PointMazeDataset before the episode split, over the full archive.
        normalize_action=True,
    )
    if args.split_mode == "code_reconstructed":
        n = len(dataset)
        n_train = int(args.split_ratio * n)
        permutation = torch.randperm(n, generator=torch.Generator().manual_seed(args.split_seed)).tolist()
        train_episode_ids = permutation[:n_train]
        holdout_episode_ids = permutation[n_train:]
        split_description = f"torch.randperm(n={n}, seed={args.split_seed}) with first {n_train} episodes as train"
    else:
        n = len(dataset)
        train_episode_ids = list(range(0, min(args.holdout_start, n)))
        holdout_episode_ids = list(range(min(args.holdout_start, n), min(args.holdout_end, n)))
        split_description = f"tail episodes {holdout_episode_ids[0] if holdout_episode_ids else args.holdout_start}--{holdout_episode_ids[-1] if holdout_episode_ids else args.holdout_end - 1}"

    def sha256_file(path: Path) -> str | None:
        if not path.exists():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    manifest = {
        "dataset": "DINO-WM PointMaze public archive",
        "dataset_root": str(args.dataset_root),
        "num_episodes": len(dataset),
        "split_ratio": args.split_ratio if args.split_mode == "code_reconstructed" else None,
        "split_seed": args.split_seed if args.split_mode == "code_reconstructed" else None,
        "split_algorithm": "torch.randperm(n, generator=torch.Generator().manual_seed(seed)); first floor(ratio*n) train, remainder valid" if args.split_mode == "code_reconstructed" else split_description,
        "train_episode_ids": train_episode_ids,
        "valid_episode_ids": holdout_episode_ids,
        "serialized_manifest_in_archive": False,
        "provenance": "Reconstructed from /tmp/dino_wm/datasets/traj_dset.py and conf/env/point_maze.yaml." if args.split_mode == "code_reconstructed" else "Evaluator-defined tail split.",
        "source_hashes": {
            "traj_dset.py": sha256_file(Path("/tmp/dino_wm/datasets/traj_dset.py")),
            "point_maze.yaml": sha256_file(Path("/tmp/dino_wm/conf/env/point_maze.yaml")),
        },
    }
    if args.manifest_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return

    train_lengths = dataset.seq_lengths[train_episode_ids]

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
    used_episodes = []
    eligible = []
    for episode in holdout_episode_ids:
        if int(dataset.get_seq_length(episode)) >= (context_len + max_h) * 5:
            eligible.append(episode)
    with torch.no_grad():
        for offset in range(0, len(eligible), args.batch_size):
            episode_batch = eligible[offset : offset + args.batch_size]
            frame_ids = [5 * i for i in range(context_len + max_h)]
            obs_batch, action_batch = [], []
            for episode in episode_batch:
                obs, _, _, _ = dataset.get_frames(episode, frame_ids)
                obs_batch.append(obs)
                action_batch.append(
                    torch.stack(
                        [dataset.actions[episode, i * 5 : (i + 1) * 5].reshape(-1) for i in range(context_len + max_h)]
                    )
                )
            obs = {key: torch.stack([item[key] for item in obs_batch]) for key in obs_batch[0]}
            blocks = torch.stack(action_batch)
            obs0 = {key: value[:, :context_len] for key, value in obs.items()}
            rollout_obs, _ = model.rollout(obs0, blocks)
            encoded = model.encode(obs, blocks)
            used_episodes.extend(episode_batch)
            for h in horizons:
                pred = rollout_obs["visual"][:, context_len - 1 + h].mean(dim=1)
                target = encoded[:, context_len - 1 + h, :, :384].mean(dim=1)
                errors = torch.mean((pred - target) ** 2, dim=1)
                per_horizon[str(h)].extend(errors.tolist())
                predicted_vectors[str(h)].extend(pred.tolist())
                target_vectors[str(h)].extend(target.tolist())

    metrics = {
        "latent_prediction_error_by_horizon": {
            h: {"mean": sum(vals) / len(vals), "episodes": len(vals)}
            for h, vals in per_horizon.items()
            if vals
        },
        "protocol": {
            "context_length": context_len,
            "frame_stride": 5,
            "horizons": horizons,
            "split": "dino_wm_loader_reconstructed" if args.split_mode == "code_reconstructed" else f"tail_holdout_{args.holdout_start}_{args.holdout_end - 1}",
            "episodes_used": used_episodes,
            "train_episode_count": len(train_episode_ids),
            "holdout_episode_count": len(holdout_episode_ids),
            "split_seed": args.split_seed if args.split_mode == "code_reconstructed" else None,
            "split_ratio": args.split_ratio if args.split_mode == "code_reconstructed" else None,
            "split_mode": args.split_mode,
            "split_description": split_description,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "original_training_split_certified": False,
            "original_training_split_code_reconstructed": args.split_mode == "code_reconstructed",
            "serialized_split_manifest_present": False,
        "split_note": (
                "Episode membership is reconstructed from the released DINO-WM loader "
                "(split_ratio=0.9, random_split_traj seed=42); no serialized manifest is "
                "included in the public archive."
                if args.split_mode == "code_reconstructed" else
                "DINO-WM archive does not ship original episode membership; train statistics are computed on episodes before holdout."
            ),
            "normalization": "Official PointMazeDataset full-archive action/proprio normalization before splitting (train.yaml normalize_action=True).",
        },
    }
    commutators, null_means = {}, {}
    for h in horizons:
        p = torch.tensor(predicted_vectors[str(h)])
        t = torch.tensor(target_vectors[str(h)])
        x = t - t.mean(0, keepdim=True)
        y = p - p.mean(0, keepdim=True)
        cov = (x.T @ x) / max(1, len(x) - 1)
        op = torch.linalg.lstsq(x, y).solution
        denom = torch.linalg.matrix_norm(cov @ op) + 1e-8
        commutators[str(h)] = float(torch.linalg.matrix_norm(cov @ op - op @ cov) / denom)
        vals = []
        for seed in range(32):
            q, _ = torch.linalg.qr(torch.randn(cov.shape[0], cov.shape[0], generator=torch.Generator().manual_seed(seed)))
            aq = q @ op @ q.T
            vals.append(float(torch.linalg.matrix_norm(cov @ aq - aq @ cov) / (torch.linalg.matrix_norm(cov @ aq) + 1e-8)))
        null_means[str(h)] = sum(vals) / len(vals)
    metrics["covariance_operator_commutator_by_horizon"] = commutators
    metrics["spectrum_preserving_null_by_horizon"] = null_means
    metrics["evidence_boundary"] = (
        "Independent DINO-WM diagnostic on the episode split reconstructed from the "
        "released loader code and public dataset ordering; this is stronger than a "
        "tail replay but remains code-reconstructed because no serialized split manifest "
        "ships with the checkpoint archive."
        if args.split_mode == "code_reconstructed" else
        "Independent DINO-WM diagnostic with reconstructed tail holdout; original checkpoint split membership is uncertified."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.split_mode == "code_reconstructed":
        manifest_path = args.output.parent / "dino_wm_pointmaze_split_manifest_reconstructed.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        metrics["protocol"]["reconstructed_split_manifest"] = str(manifest_path)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
