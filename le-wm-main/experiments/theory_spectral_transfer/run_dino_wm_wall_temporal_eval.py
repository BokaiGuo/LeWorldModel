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
    parser.add_argument("--dinov2-source", type=Path, default=Path("external/local_dinov2"))
    parser.add_argument("--save-vectors", action="store_true")
    parser.add_argument("--save-enriched-records", action="store_true",
                        help="Persist the canonical cross-architecture enriched export.")
    parser.add_argument("--architecture", default="dino_wm")
    parser.add_argument("--seed", default="unknown")
    parser.add_argument("--training-stage", type=int, default=None)
    parser.add_argument("--action-distribution", default=None)
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
    root = args.repo_root
    sys.path.insert(0, str(root / "experiments/dino_checkpoint_compat"))
    sys.path.insert(0, str(root / "external/jepa-wms"))
    sys.path.insert(0, str(root))
    original_hub_load = torch.hub.load
    if args.dinov2_source.exists():
        def hub_load(repo_or_dir, model, *extra, **kwargs):
            if repo_or_dir == "facebookresearch/dinov2":
                return original_hub_load(str(args.dinov2_source), model, source="local", *extra, **kwargs)
            return original_hub_load(repo_or_dir, model, *extra, **kwargs)
        torch.hub.load = hub_load

    from datasets.img_transforms import default_transform
    from datasets.wall_dset import WallDataset
    from experiments.empirical_spectral_audit.enriched_record_schema import make_payload
    from models.dino import DinoV2Encoder
    from models.visual_world_model import VWorldModel
    from runtime import load_temporal_components

    dataset = WallDataset(
        data_path=str(args.dataset_root),
        transform=default_transform(224),
        normalize_action=True,
    )
    raw_dataset = WallDataset(
        data_path=str(args.dataset_root),
        transform=None,
        normalize_action=False,
    )
    components = load_temporal_components(args.checkpoint)
    encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    model = VWorldModel(
        image_size=224,
        num_hist=1,
        num_pred=1,
        encoder=encoder,
        proprio_encoder=components["proprio_encoder"],
        action_encoder=components["action_encoder"],
        decoder=None,
        predictor=components["predictor"],
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
    enriched_records: dict[str, dict[str, object]] = {}
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
                obs, _, _, _, _ = dataset.get_frames(episode, frame_ids)
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
                        actions, args.action_permutation_seed or 0, episode, preserve_prefix=context_len - 1
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
                    intervene_future_action_blocks(actions, args.action_mode, preserve_prefix=context_len - 1)
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
                if args.save_enriched_records:
                    for local_index, episode in enumerate(episode_batch):
                        sample_id = f"{episode}:0"
                        record = enriched_records.get(sample_id)
                        if record is None:
                            action_episode = (
                                episodes[action_permutation[offset + local_index]]
                                if args.action_mode == "episode_shuffle"
                                else episode
                            )
                            source_frame = (context_len - 1) * 5
                            normalized_blocks = action_batch[local_index][context_len - 1 : context_len - 1 + max_h].detach().cpu()
                            # Convert exactly the action tensor consumed by the
                            # predictor back to declared raw action units. This
                            # keeps interventions and observed-action exports
                            # on the same canonical contract.
                            raw_blocks = (
                                normalized_blocks.reshape(-1, dataset.action_dim)
                                * dataset.action_std
                                + dataset.action_mean
                            ).reshape_as(normalized_blocks)
                            record = {
                                "sample_id": sample_id,
                                "episode_id": int(episode),
                                "window_start": 0,
                                "action_episode_id": int(action_episode),
                                "source_frame": int(source_frame),
                                "source_state_raw": raw_dataset.states[episode, source_frame].detach().cpu().tolist(),
                                "source_proprio_raw": raw_dataset.proprios[episode, source_frame].detach().cpu().tolist(),
                                "source_latent": encoded[local_index, context_len - 1, :, :384].mean(dim=0).detach().cpu().tolist(),
                                "action_blocks_raw": raw_blocks.tolist(),
                                "action_blocks_normalized": normalized_blocks.tolist(),
                                "latent_by_horizon": {},
                            }
                            enriched_records[sample_id] = record
                        record["latent_by_horizon"][str(horizon)] = {
                            "predicted": pred[local_index].detach().cpu().tolist(),
                            "target": target[local_index].detach().cpu().tolist(),
                            "target_state_raw": raw_dataset.states[episode, (context_len - 1 + horizon) * 5].detach().cpu().tolist(),
                        }

    metrics = {
        "latent_prediction_error_by_horizon": {
            horizon: {"mean": sum(values) / len(values), "episodes": len(values)}
            for horizon, values in per_horizon.items() if values
        },
        "protocol": {
            "environment": "Wall",
            "study_scope": "cross-architecture frozen temporal latent diagnostic",
            "architecture": args.architecture,
            "seed": str(args.seed),
            "training_stage": int(args.training_stage if args.training_stage is not None else -1),
            "action_distribution": args.action_distribution or f"wall/{args.action_mode}/normalized_dataset_stats",
            "runtime": "repo_local_dino_checkpoint_compat.v1",
            "device": str(next(model.parameters()).device),
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
            "checkpoint_epoch": None,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "dataset_archive_sha256": sha256_file(
                root / "data/dino_wm_osf/wall_single.zip"
                if (root / "data/dino_wm_osf/wall_single.zip").exists()
                else args.dataset_root.parent.parent / "wall_single.zip"
            ),
            "normalization": "Released DINO-WM WallDataset statistics over the full archive.",
            "action_mode": args.action_mode,
            "action_permutation_seed": args.action_permutation_seed,
            "action_block_count": max_h,
            "action_block_preserved_prefix": context_len - 1,
            "action_alignment": "source_action_t_predicts_target_frame_t_plus_1",
            "action_source_block_indices": list(range(context_len - 1, context_len - 1 + max_h)),
        },
        "evidence_boundary": (
            "DINO-WM Wall fixed-checkpoint temporal replay on an evaluator-defined "
            "holdout; not task performance, causal action effect, or a universal theorem."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_vectors:
        metrics["operator_fit_vectors"] = {
            "predicted": predicted_vectors,
            "target": target_vectors,
            "description": "Per-episode mean pooled latent vectors in evaluator order; for post-hoc cross-fitting only.",
        }
    if args.action_mode == "within_episode_shuffle":
        metrics["protocol"]["action_block_permutations"] = action_block_permutations
    if args.save_enriched_records:
        records = list(enriched_records.values())
        export = make_payload(
            architecture=args.architecture,
            checkpoint=str(args.checkpoint),
            environment="Wall",
            seed=args.seed,
            training_stage=int(args.training_stage if args.training_stage is not None else -1),
            latent_dimension=len(records[0]["source_latent"]),
            action_distribution=(args.action_distribution or f"wall/{args.action_mode}/normalized_dataset_stats"),
            split={"name": split_description, "episode_ids": [int(x) for x in eligible], "provenance": "DINO-WM evaluator-defined Wall holdout export"},
            horizons=horizons,
            enriched_records=records,
            checkpoint_sha256=sha256_file(args.checkpoint),
            dataset_sha256=metrics["protocol"]["dataset_archive_sha256"],
            protocol=metrics["protocol"],
            study_scope="cross-architecture frozen temporal latent diagnostic",
            claim_boundary=(
                "Frozen DINO-WM Wall temporal replay on an evaluator-defined holdout; "
                "not online planning, causal action-effect identification, or a universal world-model claim."
            ),
            runtime="repo_local_dino_checkpoint_compat.v1",
        )
        (args.output.parent / f"{args.output.stem}_enriched_export.json").write_text(json.dumps(export, indent=2) + "\n")
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
