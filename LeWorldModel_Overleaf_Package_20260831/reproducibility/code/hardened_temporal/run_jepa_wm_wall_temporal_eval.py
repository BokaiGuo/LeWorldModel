"""JEPA-WM Wall temporal/operator diagnostic on a common holdout split.

This is a fixed-checkpoint replay diagnostic.  It deliberately reports
action-intervention sensitivity and held-out operator inputs separately from
task performance or any universal world-model claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["code_reconstructed", "tail"], default="code_reconstructed")
    parser.add_argument("--holdout-start", type=int, default=1728)
    parser.add_argument("--holdout-end", type=int, default=1920)
    parser.add_argument("--dinov2-source", type=Path, default=Path("/tmp/dinov2"))
    parser.add_argument("--save-vectors", action="store_true")
    parser.add_argument(
        "--action-mode",
        choices=["observed", "zero", "reverse", "episode_shuffle", "within_episode_shuffle"],
        default="observed",
    )
    parser.add_argument("--action-permutation-seed", type=int, default=None)
    parser.add_argument("--context-length", type=int, choices=[1, 2, 3], default=2)
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
    root = args.repo_root
    sys.path.insert(0, str(root / "external/jepa-wms"))
    original_hub_load = torch.hub.load
    if args.dinov2_source.exists():
        def hub_load(repo_or_dir, model, *extra, **kwargs):
            if repo_or_dir == "facebookresearch/dinov2":
                return original_hub_load(str(args.dinov2_source), model, source="local", *extra, **kwargs)
            return original_hub_load(repo_or_dir, model, *extra, **kwargs)
        torch.hub.load = hub_load

    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from app.plan_common.datasets.wall_dset import WallDataset
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module
    from tensordict import TensorDict

    config_path = root / (
        "external/jepa-wms/configs/vjepa_wm/wall_sweep/"
        "wall_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text())
    aug = cfg["data_aug"]
    transform = make_transforms(
        img_size=224,
        normalize=aug["normalize"],
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    stats_dataset = WallDataset(data_path=str(args.dataset_root), normalize_action=True)
    dataset = WallDataset(data_path=str(args.dataset_root), transform=transform, normalize_action=False)
    preprocessor = Preprocessor(
        stats_dataset.action_mean,
        stats_dataset.action_std,
        stats_dataset.state_mean,
        stats_dataset.state_std,
        stats_dataset.proprio_mean,
        stats_dataset.proprio_std,
        transform,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = init_module(
        str(args.checkpoint.parent),
        args.checkpoint.name,
        cfg["model"],
        torch.device("cpu"),
        action_dim=2,
        proprio_dim=2,
        preprocessor=preprocessor,
        cfgs_data=cfg["data"],
        wrapper_kwargs={"ctxt_window": 1},
    )
    model.eval()

    horizons = sorted(set(args.horizons))
    max_h = max(horizons)
    context_len = args.context_length
    episodes, split_description, available_image_count = choose_episodes(args, len(dataset))

    per_horizon = {str(h): [] for h in horizons}
    predicted_vectors = {str(h): [] for h in horizons}
    target_vectors = {str(h): [] for h in horizons}
    action_block_permutations: list[list[int]] = []
    action_permutation = None
    if args.action_mode == "episode_shuffle":
        from action_interventions import derangement_indices
        action_permutation = derangement_indices(len(episodes), args.action_permutation_seed or 0)

    with torch.no_grad():
        for offset in range(0, len(episodes), args.batch_size):
            episode_batch = [
                episode for episode in episodes[offset:offset + args.batch_size]
                if int(dataset.get_seq_length(episode)) >= (context_len + max_h) * 5
            ]
            if not episode_batch:
                continue
            frame_ids = [5 * index for index in range(context_len + max_h)]
            observations = []
            for episode in episode_batch:
                obs, _, _, _, _ = dataset.get_frames(episode, frame_ids)
                obs["visual"] = obs["visual"] * 255.0
                observations.append(obs)
            obs = {key: torch.stack([item[key] for item in observations]) for key in observations[0]}
            encoded = model.encode(obs)
            context = TensorDict(
                {"visual": encoded["visual"][:, :context_len], "proprio": encoded["proprio"][:, :context_len]},
                batch_size=[len(episode_batch), context_len],
            )

            action_blocks = []
            for local_index, episode in enumerate(episode_batch):
                if args.action_mode == "episode_shuffle":
                    action_episode = episodes[action_permutation[offset + local_index]]
                else:
                    action_episode = episode
                raw_actions = dataset.actions[action_episode]
                blocks = []
                for horizon_index in range(max_h):
                    start = (horizon_index + context_len - 1) * 5
                    block = (raw_actions[start:start + 5] - stats_dataset.action_mean) / stats_dataset.action_std
                    blocks.append(block.reshape(-1))
                if args.action_mode == "within_episode_shuffle":
                    from action_interventions import derange_action_blocks
                    blocks_tensor, permutation = derange_action_blocks(
                        torch.stack(blocks), args.action_permutation_seed or 0, episode, preserve_prefix=0
                    )
                    blocks = list(blocks_tensor.unbind(0))
                    action_block_permutations.append(permutation)
                action_blocks.append(torch.stack(blocks))
            if args.action_mode == "zero":
                action_blocks = [blocks * 0 for blocks in action_blocks]
            elif args.action_mode == "reverse":
                action_blocks = [blocks.flip(0) for blocks in action_blocks]

            rollout = model.unroll(context, act_suffix=torch.stack(action_blocks).transpose(0, 1))
            for horizon in horizons:
                pred = rollout["visual"][context_len - 1 + horizon].mean(dim=1)
                target = encoded["visual"][:, context_len - 1 + horizon, :, :384].mean(dim=1)
                while pred.ndim > 2:
                    pred = pred.mean(dim=1)
                while target.ndim > 2:
                    target = target.mean(dim=1)
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
            "holdout_episode_count": len(episodes),
            "episodes_used": episodes,
            "available_image_episode_count": available_image_count,
            "visual_manifest_policy": "explicit image-backed episode IDs; missing requested images are fatal",
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "dataset_archive_sha256": sha256_file(args.dataset_root.parent.parent / "wall_single.zip"),
            "normalization": "Official JEPA-WM WallDataset statistics over the full archive.",
            "action_mode": args.action_mode,
            "action_permutation_seed": args.action_permutation_seed,
            "action_block_count": max_h,
        },
        "evidence_boundary": (
            "JEPA-WM Wall fixed-checkpoint temporal replay on an evaluator-defined "
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
