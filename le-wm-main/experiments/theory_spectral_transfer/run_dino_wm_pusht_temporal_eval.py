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
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Optional deterministic cap for smoke runs; omitted means use every eligible holdout episode.")
    p.add_argument("--dinov2-source", type=Path, default=Path("external/local_dinov2"))
    p.add_argument("--dino-source", type=Path, default=None,
                   help="Repo or extracted source tree containing the original DINO-WM models/ and datasets/ packages.")
    p.add_argument("--architecture", default="dino_wm")
    p.add_argument("--seed", default="unknown")
    p.add_argument("--training-stage", type=int, default=None)
    p.add_argument("--action-distribution", default=None)
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
    from datasets.pusht_dset import PushTDataset
    from models.dino import DinoV2Encoder
    from models.visual_world_model import VWorldModel
    from runtime import load_temporal_components

    dataset = PushTDataset(
        data_path=str(args.dataset_root / "val"),
        transform=default_transform(224),
        normalize_action=True,
        with_velocity=True,
    )
    raw_dataset = PushTDataset(
        data_path=str(args.dataset_root / "val"),
        transform=None,
        normalize_action=False,
        with_velocity=True,
    )
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
    eligible = [
        i for i in range(len(dataset))
        if int(dataset.get_seq_length(i)) >= (context_len + max_h) * 5
    ]
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
                action_tensor = torch.stack([
                    dataset.actions[action_episode, i * 5 : (i + 1) * 5].reshape(-1)
                    for i in range(context_len + max_h)
                ])
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
                            action_episode = episode
                            if args.action_mode == "episode_shuffle":
                                if action_permutation is not None:
                                    action_episode = eligible[action_permutation[offset + local_index]]
                                else:
                                    action_episode = eligible[(offset + local_index + 1) % len(eligible)]
                            source_index = (context_len - 1) * 5
                            record = {
                                "sample_id": sample_id,
                                "episode_id": int(episode),
                                "window_start": 0,
                                "action_episode_id": int(action_episode),
                                "source_frame": int(source_index),
                                "source_state_raw": raw_dataset.states[episode, source_index].detach().cpu().tolist(),
                                "source_proprio_raw": raw_dataset.proprios[episode, source_index].detach().cpu().tolist(),
                                "source_latent": encoded[local_index, context_len - 1, :, :384].mean(dim=0).detach().cpu().tolist(),
                                "action_blocks_raw": [
                                    raw_dataset.actions[action_episode, i * 5 : (i + 1) * 5]
                                    .reshape(-1).detach().cpu().tolist()
                                    for i in range(context_len - 1, context_len - 1 + max_h)
                                ],
                                "action_blocks_normalized": action_batch[local_index][context_len - 1 : context_len - 1 + max_h].detach().cpu().tolist(),
                                "latent_by_horizon": {},
                            }
                            enriched_records[sample_id] = record
                        record["latent_by_horizon"][str(h)] = {
                            "predicted": pred[local_index].detach().cpu().tolist(),
                            "target": target[local_index].detach().cpu().tolist(),
                            "target_state_raw": raw_dataset.states[episode, (context_len - 1 + h) * 5].detach().cpu().tolist(),
                        }

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
            "study_scope": "cross-architecture frozen temporal latent diagnostic",
            "architecture": args.architecture,
            "seed": str(args.seed),
            "training_stage": int(args.training_stage if args.training_stage is not None else -1),
            "action_distribution": args.action_distribution or f"pusht/{args.action_mode}/normalized_dataset_stats",
            "runtime": "repo_local_dino_checkpoint_compat.v1",
            "device": str(next(model.parameters()).device),
            "context_length": context_len,
            "frame_stride": 5,
            "horizons": horizons,
            "split": "public_val_directory",
            "action_mode": args.action_mode,
            "action_permutation_seed": args.action_permutation_seed,
            "action_block_count": max_h,
            "action_block_preserved_prefix": context_len - 1,
            "action_alignment": "source_action_t_predicts_target_frame_t_plus_1",
            "action_source_block_indices": list(range(context_len - 1, context_len - 1 + max_h)),
            "train_test_split_serialized": True,
            "train_episode_count": len(train_dataset),
            "holdout_episode_count": len(eligible),
            "episodes_used": eligible,
            "checkpoint_epoch": None,
            "normalization": "Released PushTDataset constants with normalize_action=True.",
            "runtime": "repo_local_dino_checkpoint_compat.v1",
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
            environment="PushT",
            seed=args.seed,
            training_stage=int(args.training_stage if args.training_stage is not None else -1),
            latent_dimension=len(records[0]["source_latent"]),
            action_distribution=(args.action_distribution or f"pusht/{args.action_mode}/normalized_dataset_stats"),
            split={"name": "public_val_directory", "episode_ids": [int(x) for x in eligible], "provenance": "DINO-WM loader-defined export"},
            horizons=horizons,
            enriched_records=records,
            checkpoint_sha256=sha256_file(args.checkpoint),
            dataset_sha256=sha256_file(args.dataset_root.parent.parent / "pusht_noise.zip"),
            protocol=metrics["protocol"],
            study_scope="cross-architecture frozen temporal latent diagnostic",
            claim_boundary=(
                "Frozen DINO-WM temporal replay with explicit split and source-state/action fields; "
                "not online planning, causal action-effect identification, or a universal world-model claim."
            ),
            runtime="repo_local_dino_checkpoint_compat.v1",
        )
        (args.output.parent / f"{args.output.stem}_enriched_export.json").write_text(json.dumps(export, indent=2) + "\n")
    if args.action_mode == "within_episode_shuffle":
        metrics["protocol"]["action_block_permutations"] = action_block_permutations
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
