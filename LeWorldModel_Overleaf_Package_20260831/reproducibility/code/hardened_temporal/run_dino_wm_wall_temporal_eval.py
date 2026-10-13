"""DINO-WM Wall temporal/operator diagnostic on a common holdout split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["code_reconstructed", "tail"], default="code_reconstructed")
    parser.add_argument("--holdout-start", type=int, default=1728)
    parser.add_argument("--holdout-end", type=int, default=1920)
    parser.add_argument("--dinov2-source", type=Path, default=Path("/tmp/dinov2"))
    parser.add_argument("--save-vectors", action="store_true")
    parser.add_argument(
        "--action-mode",
        choices=["observed", "zero", "reverse", "future_zero", "future_reverse", "episode_shuffle", "within_episode_shuffle"],
        default="observed",
    )
    parser.add_argument("--action-permutation-seed", type=int, default=None)
    args = parser.parse_args()
    if args.action_mode == "within_episode_shuffle" and args.action_permutation_seed is None:
        parser.error("--action-permutation-seed is required for within_episode_shuffle")
    return args


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_backed_episode_ids(dataset_root: Path) -> list[int]:
    ids = []
    for path in (dataset_root / "obses").glob("episode_*.pth"):
        try:
            ids.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def choose_episodes(args: argparse.Namespace, dataset_length: int) -> tuple[list[int], str, int]:
    available = image_backed_episode_ids(args.dataset_root)
    if args.split_mode == "tail":
        selected = [episode for episode in available if args.holdout_start <= episode < args.holdout_end]
        episodes = selected[:args.episodes]
        description = f"image_backed_tail_{args.holdout_start}_{args.holdout_end}"
    else:
        permutation = torch.randperm(dataset_length, generator=torch.Generator().manual_seed(args.split_seed)).tolist()
        requested = permutation[-min(args.episodes, len(permutation)):]
        missing = [episode for episode in requested if episode not in set(available)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} requested Wall holdout images are missing; first IDs: {missing[:10]}. "
                "Extract the seed-specific files from wall_single.zip before running the evaluator."
            )
        episodes = requested
        description = f"code_reconstructed_wall_seed{args.split_seed}"
    return episodes, description, len(available)


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
    from datasets.wall_dset import WallDataset
    from models.dino import DinoV2Encoder
    from models.visual_world_model import VWorldModel

    dataset = WallDataset(
        data_path=str(args.dataset_root),
        transform=default_transform(224),
        normalize_action=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    model = VWorldModel(
        image_size=224,
        num_hist=1,
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
    context_len = 1
    episodes, split_description, available_image_count = choose_episodes(args, len(dataset))

    per_horizon = {str(h): [] for h in horizons}
    predicted_vectors = {str(h): [] for h in horizons}
    target_vectors = {str(h): [] for h in horizons}
    action_block_permutations: list[list[int]] = []
    action_permutation = None
    if args.action_mode == "episode_shuffle":
        from action_interventions import derangement_indices
        action_permutation = derangement_indices(len(episodes), args.action_permutation_seed or 0)

    eligible = [
        episode for episode in episodes
        if int(dataset.get_seq_length(episode)) >= (context_len + max_h) * 5
    ]
    with torch.no_grad():
        for offset in range(0, len(eligible), args.batch_size):
            episode_batch = eligible[offset:offset + args.batch_size]
            frame_ids = [5 * index for index in range(context_len + max_h)]
            obs_batch, action_batch = [], []
            for local_index, episode in enumerate(episode_batch):
                obs, _, _, _ = dataset.get_frames(episode, frame_ids)
                obs_batch.append(obs)
                if args.action_mode == "episode_shuffle":
                    action_episode = episodes[action_permutation[offset + local_index]]
                else:
                    action_episode = episode
                actions = torch.stack([
                    dataset.actions[action_episode, index * 5:(index + 1) * 5].reshape(-1)
                    for index in range(context_len + max_h)
                ])
                if args.action_mode == "within_episode_shuffle":
                    from action_interventions import derange_action_blocks
                    actions, permutation = derange_action_blocks(
                        actions, args.action_permutation_seed or 0, episode, preserve_prefix=context_len
                    )
                    action_block_permutations.append(permutation)
                action_batch.append(actions)
            if args.action_mode == "zero":
                action_batch = [actions * 0 for actions in action_batch]
            elif args.action_mode == "reverse":
                action_batch = [actions.flip(0) for actions in action_batch]
            elif args.action_mode in {"future_zero", "future_reverse"}:
                from action_interventions import intervene_future_action_blocks
                action_batch = [
                    intervene_future_action_blocks(actions, args.action_mode, preserve_prefix=context_len)
                    for actions in action_batch
                ]

            obs = {key: torch.stack([item[key] for item in obs_batch]) for key in obs_batch[0]}
            blocks = torch.stack(action_batch)
            obs0 = {key: value[:, :context_len] for key, value in obs.items()}
            rollout_obs, _ = model.rollout(obs0, blocks)
            encoded = model.encode(obs, blocks)
            for horizon in horizons:
                pred = rollout_obs["visual"][:, context_len - 1 + horizon].mean(dim=1)
                target = encoded[:, context_len - 1 + horizon, :, :384].mean(dim=1)
                errors = torch.mean((pred - target) ** 2, dim=1)
                per_horizon[str(horizon)].extend(errors.tolist())
                predicted_vectors[str(horizon)].extend(pred.tolist())
                target_vectors[str(horizon)].extend(target.tolist())

    metrics = {
        "latent_prediction_error_by_horizon": {
            horizon: {"mean": sum(values) / len(values), "episodes": len(values)}
            for horizon, values in per_horizon.items() if values
        },
        "protocol": {
            "environment": "Wall",
            "context_length": context_len,
            "frame_stride": 5,
            "horizons": horizons,
            "split": split_description,
            "split_mode": args.split_mode,
            "split_seed": args.split_seed if args.split_mode == "code_reconstructed" else None,
            "train_episode_count": len(dataset) - len(episodes),
            "holdout_episode_count": len(eligible),
            "episodes_used": eligible,
            "available_image_episode_count": available_image_count,
            "visual_manifest_policy": "explicit image-backed episode IDs; missing requested images are fatal",
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "dataset_archive_sha256": sha256_file(args.dataset_root.parent.parent / "wall_single.zip"),
            "normalization": "Released DINO-WM WallDataset statistics over the full archive.",
            "action_mode": args.action_mode,
            "action_permutation_seed": args.action_permutation_seed,
            "action_block_count": max_h,
            "action_block_preserved_prefix": context_len,
        },
        "evidence_boundary": (
            "DINO-WM Wall fixed-checkpoint temporal replay on an evaluator-defined "
            "holdout; not task performance, causal action effect, or a universal theorem."
        ),
    }
    if args.save_vectors:
        metrics["operator_fit_vectors"] = {
            "predicted": predicted_vectors,
            "target": target_vectors,
            "description": "Per-episode mean pooled latent vectors in evaluator order; for post-hoc cross-fitting only.",
        }
    if args.action_mode == "within_episode_shuffle":
        metrics["protocol"]["action_block_permutations"] = action_block_permutations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
