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
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Optional deterministic cap for smoke runs; omitted means use every eligible holdout episode.")
    p.add_argument("--dinov2-source", type=Path, default=Path("external/local_dinov2"))
    p.add_argument("--dino-source", type=Path, default=None,
                   help="Repo or extracted source tree containing the original DINO-WM models/ and datasets/ packages.")
    p.add_argument("--architecture", default="dino_wm")
    p.add_argument("--seed", default="unknown")
    p.add_argument("--training-stage", type=int, default=None)
    p.add_argument("--action-distribution", default=None)
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--save-vectors", action="store_true",
                   help="Persist per-episode predicted/target latent vectors for post-hoc controls.")
    p.add_argument("--save-enriched-records", action="store_true",
                   help="Persist the canonical cross-architecture enriched export.")
    p.add_argument("--action-mode", choices=["observed", "zero", "reverse", "future_zero", "future_reverse", "episode_shuffle", "within_episode_shuffle"], default="observed",
                   help="Action intervention: observed, zero, reversed, cross-episode, or within-episode future-block shuffle.")
    p.add_argument("--action-permutation-seed", type=int, default=None)
    args = p.parse_args()
    if args.action_mode == "within_episode_shuffle" and args.action_permutation_seed is None:
        p.error("--action-permutation-seed is required for within_episode_shuffle")
    return args


def main() -> None:
    args = parse_args()
    root = args.repo_root
    sys.path.insert(0, str(root / "experiments/dino_checkpoint_compat"))
    sys.path.insert(0, str(root / "external/jepa-wms"))
    sys.path.insert(0, str(root))
    from experiments.empirical_spectral_audit.enriched_record_schema import make_payload, sha256_file
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
    from runtime import load_temporal_components

    dataset = PointMazeDataset(
        data_path=str(args.dataset_root),
        transform=default_transform(224),
        # Match the released DINO-WM train.yaml: normalization is computed by
        # PointMazeDataset before the episode split, over the full archive.
        normalize_action=True,
    )
    raw_dataset = PointMazeDataset(data_path=str(args.dataset_root), normalize_action=False)
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
        "provenance": "Reconstructed from the released DINO-WM PointMaze ordering with the repo-local compatibility runtime." if args.split_mode == "code_reconstructed" else "Evaluator-defined tail split.",
        "source_hashes": {
            "compat_runtime.py": sha256_file(root / "experiments/dino_checkpoint_compat/runtime.py"),
            "point_maze_dset.py": sha256_file(root / "experiments/dino_checkpoint_compat/datasets/point_maze_dset.py"),
        },
    }
    if args.manifest_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return

    train_lengths = dataset.seq_lengths[train_episode_ids]

    components = load_temporal_components(args.checkpoint)
    encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    model = VWorldModel(
        image_size=224,
        num_hist=3,
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
    context_len = 3
    per_horizon = {str(h): [] for h in horizons}
    predicted_vectors = {str(h): [] for h in horizons}
    target_vectors = {str(h): [] for h in horizons}
    enriched_records: dict[str, dict[str, object]] = {}
    action_block_permutations: list[list[int]] = []
    used_episodes = []
    eligible = []
    for episode in holdout_episode_ids:
        if int(dataset.get_seq_length(episode)) >= (context_len + max_h) * 5:
            eligible.append(episode)
    if args.max_episodes is not None:
        eligible = eligible[: args.max_episodes]
    action_permutation = None
    if args.action_mode == "episode_shuffle" and args.action_permutation_seed is not None:
        from action_interventions import derangement_indices
        action_permutation = derangement_indices(len(eligible), args.action_permutation_seed)
    with torch.no_grad():
        for offset in range(0, len(eligible), args.batch_size):
            episode_batch = eligible[offset : offset + args.batch_size]
            frame_ids = [5 * i for i in range(context_len + max_h)]
            obs_batch, action_batch = [], []
            for local_index, episode in enumerate(episode_batch):
                if args.action_mode == "episode_shuffle":
                    if action_permutation is not None:
                        action_episode = eligible[action_permutation[offset + local_index]]
                    else:
                        action_episode = eligible[(offset + local_index + 1) % len(eligible)]
                else:
                    action_episode = episode
                obs, _, _, _, _ = dataset.get_frames(episode, frame_ids)
                obs_batch.append(obs)
                action_tensor = torch.stack(
                    [dataset.actions[action_episode, i * 5 : (i + 1) * 5].reshape(-1) for i in range(context_len + max_h)]
                )
                if args.action_mode == "within_episode_shuffle":
                    from action_interventions import derange_action_blocks
                    action_tensor, permutation = derange_action_blocks(
                        action_tensor, args.action_permutation_seed or 0, episode, preserve_prefix=context_len - 1
                    )
                    action_block_permutations.append(permutation)
                action_batch.append(action_tensor)
            if args.action_mode == "zero":
                action_batch = [torch.zeros_like(actions) for actions in action_batch]
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
            used_episodes.extend(episode_batch)
            for h in horizons:
                pred = rollout_obs["visual"][:, context_len - 1 + h].mean(dim=1)
                target = encoded[:, context_len - 1 + h, :, :384].mean(dim=1)
                errors = torch.mean((pred - target) ** 2, dim=1)
                per_horizon[str(h)].extend(errors.tolist())
                predicted_vectors[str(h)].extend(pred.tolist())
                target_vectors[str(h)].extend(target.tolist())
                if args.save_enriched_records:
                    for local_index, episode in enumerate(episode_batch):
                        sample_id = f"{episode}:0"
                        record = enriched_records.get(sample_id)
                        if record is None:
                            source_index = (context_len - 1) * 5
                            source_latent = encoded[local_index, context_len - 1, :, :384].mean(dim=0)
                            action_episode = episode
                            if args.action_mode == "episode_shuffle":
                                if action_permutation is not None:
                                    action_episode = eligible[action_permutation[offset + local_index]]
                                else:
                                    action_episode = eligible[(offset + local_index + 1) % len(eligible)]
                            raw_blocks = [
                                raw_dataset.actions[action_episode, j * 5 : (j + 1) * 5]
                                .reshape(-1).detach().cpu().tolist()
                                for j in range(context_len - 1, context_len - 1 + max_h)
                            ]
                            normalized_blocks = action_batch[local_index][context_len - 1 : context_len - 1 + max_h].detach().cpu().tolist()
                            record = {
                                "sample_id": sample_id,
                                "episode_id": int(episode),
                                "window_start": 0,
                                "action_episode_id": int(action_episode),
                                "source_frame": int(source_index),
                                "source_state_raw": raw_dataset.states[episode, source_index].detach().cpu().tolist(),
                                "source_proprio_raw": raw_dataset.proprios[episode, source_index].detach().cpu().tolist(),
                                "source_latent": source_latent.detach().cpu().tolist(),
                                "action_blocks_raw": raw_blocks,
                                "action_blocks_normalized": normalized_blocks,
                                "latent_by_horizon": {},
                            }
                            enriched_records[sample_id] = record
                        record["latent_by_horizon"][str(h)] = {
                            "predicted": pred[local_index].detach().cpu().tolist(),
                            "target": target[local_index].detach().cpu().tolist(),
                            "target_state_raw": raw_dataset.states[episode, (context_len - 1 + h) * 5].detach().cpu().tolist(),
                        }

    metrics = {
        "latent_prediction_error_by_horizon": {
            h: {"mean": sum(vals) / len(vals), "episodes": len(vals)}
            for h, vals in per_horizon.items()
            if vals
        },
        "protocol": {
            "study_scope": "cross-architecture frozen temporal latent diagnostic",
            "architecture": args.architecture,
            "seed": str(args.seed),
            "training_stage": int(args.training_stage if args.training_stage is not None else -1),
            "action_distribution": args.action_distribution or f"pointmaze/{args.action_mode}/normalized_dataset_stats",
            "runtime": "repo_local_dino_checkpoint_compat.v1",
            "device": str(next(model.parameters()).device),
            "context_length": context_len,
            "frame_stride": 5,
            "horizons": horizons,
            "split": "dino_wm_loader_reconstructed" if args.split_mode == "code_reconstructed" else f"tail_holdout_{args.holdout_start}_{args.holdout_end}",
            "episodes_used": used_episodes,
            "train_episode_count": len(train_episode_ids),
            "holdout_episode_count": len(holdout_episode_ids),
            "split_seed": args.split_seed if args.split_mode == "code_reconstructed" else None,
            "split_ratio": args.split_ratio if args.split_mode == "code_reconstructed" else None,
            "split_mode": args.split_mode,
            "split_description": split_description,
            "checkpoint_epoch": None,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "dataset_archive_sha256": sha256_file(args.dataset_root.parent.parent / "point_maze.zip"),
            "original_training_split_certified": False,
            "original_training_split_code_reconstructed": args.split_mode == "code_reconstructed",
            "serialized_split_manifest_present": False,
        "split_note": (
                "Episode membership is reconstructed from the released DINO-WM loader "
                f"(split_ratio={args.split_ratio}, random_split_traj seed={args.split_seed}); no serialized manifest is "
                "included in the public archive."
                if args.split_mode == "code_reconstructed" else
                "DINO-WM archive does not ship original episode membership; train statistics are computed on episodes before holdout."
            ),
            "normalization": "Official PointMazeDataset full-archive action/proprio normalization before splitting (train.yaml normalize_action=True).",
            "runtime": "repo_local_dino_checkpoint_compat.v1",
            "action_mode": args.action_mode,
            "action_permutation_seed": args.action_permutation_seed,
            "action_block_count": max_h,
            "action_block_preserved_prefix": context_len - 1,
            "action_alignment": "source_action_t_predicts_target_frame_t_plus_1",
            "action_source_block_indices": list(range(context_len - 1, context_len - 1 + max_h)),
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
    if args.save_vectors:
        metrics["operator_fit_vectors"] = {
            "predicted": predicted_vectors,
            "target": target_vectors,
            "description": "Per-episode mean pooled latent vectors in evaluator order; for post-hoc cross-fitting only.",
        }
    if args.save_enriched_records:
        records = list(enriched_records.values())
        export = make_payload(
            architecture=args.architecture,
            checkpoint=str(args.checkpoint),
            environment="PointMaze",
            seed=args.seed,
            training_stage=int(args.training_stage if args.training_stage is not None else -1),
            latent_dimension=len(records[0]["source_latent"]),
            action_distribution=(args.action_distribution or f"pointmaze/{args.action_mode}/normalized_dataset_stats"),
            split={"name": metrics["protocol"]["split"], "episode_ids": [int(x) for x in used_episodes], "provenance": "DINO-WM loader-defined export"},
            horizons=horizons,
            enriched_records=records,
            checkpoint_sha256=sha256_file(args.checkpoint),
            dataset_sha256=sha256_file(args.dataset_root.parent.parent / "point_maze.zip"),
            protocol=metrics["protocol"],
            study_scope="cross-architecture frozen temporal latent diagnostic",
            claim_boundary=(
                "Frozen DINO-WM temporal replay on a declared holdout split; original checkpoint split "
                "membership is uncertified where the public archive lacks a serialized manifest, and this "
                "does not establish online planning, causal action effects, or a universal world model."
            ),
            runtime="repo_local_dino_checkpoint_compat.v1",
        )
        (args.output.parent / f"{args.output.stem}_enriched_export.json").write_text(json.dumps(export, indent=2) + "\n")
    if args.action_mode == "within_episode_shuffle":
        metrics["protocol"]["action_block_permutations"] = action_block_permutations
    if args.split_mode == "code_reconstructed":
        manifest_path = args.output.parent / "dino_wm_pointmaze_split_manifest_reconstructed.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        metrics["protocol"]["reconstructed_split_manifest"] = str(manifest_path)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
