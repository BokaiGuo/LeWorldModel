"""Run a small, reproducible held-out JEPA-WM PushT temporal evaluation.

This is an evidence-producing diagnostic, not a claim that all learned JEPAs
or world models satisfy the finite-dimensional theorem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=21)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--dinov2-source", type=Path, default=Path("/tmp/dinov2"))
    p.add_argument("--save-vectors", action="store_true",
                   help="Persist per-episode predicted/target latent vectors for post-hoc controls.")
    p.add_argument("--save-enriched-records", action="store_true",
                   help=("Persist one record per evaluated episode containing the fed action blocks, "
                         "raw state at the source and each target horizon, and latent vectors. "
                         "This is required for external action/task metrics."))
    p.add_argument("--action-mode", choices=["observed", "zero", "reverse", "episode_shuffle", "within_episode_shuffle"], default="observed",
                   help="Action intervention: observed, zero, reversed, cross-episode, or within-episode future-block shuffle.")
    p.add_argument("--context-length", type=int, choices=[1, 2], default=2)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Inference device; auto selects CUDA when available.")
    p.add_argument("--windows-per-episode", type=int, default=1,
                   help="Number of non-overlapping temporal windows exported per episode.")
    p.add_argument("--window-stride", type=int, default=5,
                   help="Stride in frames between enriched windows.")
    p.add_argument("--action-permutation-seed", type=int, default=None)
    p.add_argument("--architecture", default="jepa_wm")
    p.add_argument("--seed", default="unknown")
    p.add_argument("--training-stage", type=int, default=None)
    p.add_argument("--action-distribution", default=None,
                   help="Declared action protocol used for matched-model strata.")
    args = p.parse_args()
    if args.action_mode == "within_episode_shuffle" and args.action_permutation_seed is None:
        p.error("--action-permutation-seed is required for within_episode_shuffle")
    return args


def main() -> None:
    args = parse_args()
    # Create the destination before emitting the optional enriched export.
    # The previous ordering only created the directory after that write, so a
    # valid inference run could be lost at finalization time.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.windows_per_episode < 1 or args.window_stride < 1:
        raise ValueError("windows-per-episode and window-stride must be positive")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cuda" if args.device == "cuda" else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    root = args.repo_root
    sys.path.insert(0, str(root))
    from experiments.empirical_spectral_audit.enriched_record_schema import make_payload, sha256_file
    sys.path.insert(0, str(root / "external/jepa-wms"))
    original_hub_load = torch.hub.load
    if args.dinov2_source.exists():
        def hub_load(repo_or_dir, model, *extra, **kwargs):
            if repo_or_dir == "facebookresearch/dinov2":
                return original_hub_load(str(args.dinov2_source), model, source="local", *extra, **kwargs)
            return original_hub_load(repo_or_dir, model, *extra, **kwargs)
        torch.hub.load = hub_load

    from app.plan_common.datasets.pusht_dset import (
        ACTION_MEAN,
        ACTION_STD,
        PROPRIO_MEAN,
        PROPRIO_STD,
        PushTDataset,
    )
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from tensordict import TensorDict
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

    config_path = root / "external/jepa-wms/configs/vjepa_wm/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml"
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
    preprocessor = Preprocessor(
        ACTION_MEAN,
        ACTION_STD,
        torch.zeros(7),
        torch.ones(7),
        PROPRIO_MEAN,
        PROPRIO_STD,
        transform,
    )
    model = init_module(
        str(args.checkpoint.parent),
        args.checkpoint.name,
        cfg["model"],
        device,
        action_dim=2,
        proprio_dim=4,
        preprocessor=preprocessor,
        cfgs_data=cfg["data"],
        wrapper_kwargs={"ctxt_window": 1},
    )
    dataset = PushTDataset(
        n_rollout=args.episodes,
        data_path=str(args.dataset_root / "val"),
        normalize_action=False,
        with_velocity=True,
    )
    horizons = sorted(set(args.horizons))
    max_h = max(horizons)
    context_len = args.context_length
    per_horizon: dict[str, list[float]] = {str(h): [] for h in horizons}
    predicted_vectors: dict[str, list[list[float]]] = {str(h): [] for h in horizons}
    target_vectors: dict[str, list[list[float]]] = {str(h): [] for h in horizons}
    enriched_records: list[dict[str, object]] = []
    episodes_used: list[int] = []
    action_block_permutations: list[list[int]] = []
    episodes = []
    samples: list[tuple[int, int]] = []
    required_frames = (context_len + max_h - 1) * 5 + 1
    for episode in range(min(args.episodes, len(dataset))):
        seq_length = int(dataset.get_seq_length(episode))
        if seq_length < required_frames:
            continue
        max_start = seq_length - required_frames
        starts = list(range(0, max_start + 1, args.window_stride))[: args.windows_per_episode]
        episodes.append(episode)
        samples.extend((episode, start) for start in starts)
    action_permutation = None
    if args.action_mode == "episode_shuffle" and args.action_permutation_seed is not None:
        from action_interventions import derangement_indices
        action_permutation = derangement_indices(len(episodes), args.action_permutation_seed)

    with torch.no_grad():
        for sample_index, (episode, start_frame) in enumerate(samples):
            episodes_used.append(episode)
            context_len = args.context_length
            frame_ids = [start_frame + 5 * h for h in range(context_len + max_h)]
            obs, _, _, _, _ = dataset.get_frames(episode, frame_ids)
            obs["visual"] = obs["visual"] * 255.0  # wrapper performs /255 once
            encoded = model.encode(
                {"visual": obs["visual"].unsqueeze(0).to(device), "proprio": obs["proprio"].unsqueeze(0).to(device)}
            )
            context = TensorDict(
                {"visual": encoded["visual"][:, :context_len], "proprio": encoded["proprio"][:, :context_len]},
                batch_size=[1, context_len],
            )
            if args.action_mode == "episode_shuffle":
                if action_permutation is not None:
                    action_episode = episodes[action_permutation[episodes.index(episode)]]
                else:
                    action_episode = episodes[(episodes.index(episode) + 1) % len(episodes)]
            else:
                action_episode = episode
            raw = dataset.actions[action_episode]
            blocks = []
            raw_blocks = []
            for h in range(max_h):
                # The first predicted frame follows the last context frame;
                # therefore consume action blocks starting at frame 1.
                action_start = start_frame + (h + context_len - 1) * 5
                block = raw[action_start : action_start + 5] / ACTION_STD
                blocks.append(block.reshape(-1))
                raw_blocks.append(raw[action_start : action_start + 5].reshape(-1))
            if args.action_mode == "zero":
                blocks = [torch.zeros_like(block) for block in blocks]
            elif args.action_mode == "reverse":
                blocks = list(reversed(blocks))
            elif args.action_mode == "within_episode_shuffle":
                from action_interventions import derange_action_blocks
                block_tensor, permutation = derange_action_blocks(
                    torch.stack(blocks), args.action_permutation_seed or 0, episode, preserve_prefix=0
                )
                blocks = list(block_tensor.unbind(0))
                action_block_permutations.append(permutation)
            act_suffix = torch.stack(blocks).unsqueeze(1).to(device)
            rollout = model.unroll(context, act_suffix=act_suffix)
            for h in horizons:
                # `unroll` returns the original context followed by predicted
                # frames.  Horizon one is therefore at index context_len,
                # not index one.
                pred = rollout["visual"][context_len - 1 + h, 0].mean(dim=(0, 1, 2))
                target = encoded["visual"][0, context_len - 1 + h].mean(dim=(0, 1, 2))
                per_horizon[str(h)].append(float(torch.mean((pred - target) ** 2)))
                predicted_vectors[str(h)].append(pred.tolist())
                target_vectors[str(h)].append(target.tolist())
                if args.save_enriched_records:
                    # Keep the source and target states in the dataset's raw
                    # coordinates.  The action is saved both in raw dataset
                    # units and in the normalized units actually fed to the
                    # predictor.  This avoids reconstructing an action/task
                    # metric from pooled latent vectors after the fact.
                    record = next((item for item in enriched_records if item["sample_id"] == f"{episode}:{start_frame}"), None)
                    if record is None:
                        source_index = start_frame + (context_len - 1) * 5
                        source_state = dataset.states[episode, source_index].detach().cpu().tolist()
                        record = {
                            "sample_id": f"{episode}:{start_frame}",
                            "episode_id": int(episode),
                            "window_start": int(start_frame),
                            "action_episode_id": int(action_episode),
                            "source_frame": int(source_index),
                            "source_state_raw": source_state,
                            "source_proprio_raw": dataset.proprios[episode, source_index].detach().cpu().tolist(),
                            "source_latent": encoded["visual"][0, context_len - 1].mean(dim=(0, 1, 2)).detach().cpu().tolist(),
                            "action_blocks_raw": [item.detach().cpu().tolist() for item in raw_blocks],
                            "action_blocks_normalized": [item.detach().cpu().tolist() for item in blocks],
                            "latent_by_horizon": {},
                        }
                        enriched_records.append(record)
                    record["latent_by_horizon"][str(h)] = {
                        "predicted": pred.detach().cpu().tolist(),
                        "target": target.detach().cpu().tolist(),
                        "target_state_raw": dataset.states[episode, start_frame + (context_len - 1 + h) * 5].detach().cpu().tolist(),
                    }

    metrics: dict[str, object] = {
        "latent_prediction_error_by_horizon": {
            h: {"mean": sum(v) / len(v), "episodes": len(v)} for h, v in per_horizon.items() if v
        },
        "protocol": {
                "context_length": args.context_length,
            "frame_stride": 5,
            "horizons": horizons,
                "split": "val",
                "episodes_used": episodes_used,
                "device": str(device),
                "checkpoint_epoch": 50,
                "action_mode": args.action_mode,
                "action_permutation_seed": args.action_permutation_seed,
                "action_block_count": max_h,
            "windows_per_episode": args.windows_per_episode,
            "window_stride": args.window_stride,
                "interface_note": "context-two multimodal AdaLN path with zero-padded historical action token",
        },
    }
    # A covariance/operator diagnostic is only emitted when enough episodes exist.
    # It is intentionally marked diagnostic because this is one released
    # checkpoint under one explicit two-frame protocol.
    commutators: dict[str, float] = {}
    null_means: dict[str, float] = {}
    for h in horizons:
        p = torch.tensor(predicted_vectors[str(h)])
        t = torch.tensor(target_vectors[str(h)])
        if len(p) < 3:
            continue
        x = t - t.mean(0, keepdim=True)
        y = p - p.mean(0, keepdim=True)
        cov = (x.T @ x) / max(1, len(x) - 1)
        op = torch.linalg.lstsq(x, y).solution
        commutators[str(h)] = float(torch.linalg.matrix_norm(cov @ op - op @ cov) / (torch.linalg.matrix_norm(cov @ op) + 1e-8))
        vals = []
        for seed in range(32):
            g = torch.Generator().manual_seed(seed)
            q, _ = torch.linalg.qr(torch.randn(cov.shape[0], cov.shape[0], generator=g))
            vals.append(float(torch.linalg.matrix_norm(cov @ (q @ op @ q.T) - (q @ op @ q.T) @ cov) / (torch.linalg.matrix_norm(cov @ (q @ op @ q.T)) + 1e-8)))
        null_means[str(h)] = sum(vals) / len(vals)
    metrics["covariance_operator_commutator_by_horizon"] = commutators
    metrics["spectrum_preserving_null_by_horizon"] = null_means
    if args.save_vectors:
        metrics["operator_fit_vectors"] = {
            "predicted": predicted_vectors,
            "target": target_vectors,
            "description": "Per-episode mean pooled latent vectors in evaluator order; for post-hoc cross-fitting only.",
        }
    if args.save_enriched_records:
        metrics["enriched_records"] = enriched_records
        metrics["enriched_record_schema"] = {
            "version": "enriched_learned_model.v1",
            "unit": "one record per evaluated episode and action source",
            "raw_action_units": "dataset actions before ACTION_STD normalization; PushT dataset scaling is retained",
            "raw_state_units": "dataset states before model normalization",
            "source_state_index": "frame=start+(context_length-1)*5",
            "target_state_index": "frame=start+(context_length-1+horizon)*5",
            "metric_use": "fit external task/action metric on train-only records; never use held-out planning outcomes",
        }
        export = make_payload(
            architecture=args.architecture,
            checkpoint=str(args.checkpoint),
            environment="PushT",
            seed=args.seed,
            training_stage=(int(args.training_stage) if args.training_stage is not None else 50),
            latent_dimension=len(enriched_records[0]["source_latent"]),
            action_distribution=(args.action_distribution or f"pusht/{args.action_mode}/normalized_ACTION_STD"),
            split={
                "name": "public_val_directory",
                "episode_ids": sorted(set(episodes_used)),
                "windows_per_episode": args.windows_per_episode,
                "window_stride": args.window_stride,
                "provenance": "JEPA-WM evaluator-defined validation export",
            },
            horizons=horizons,
            enriched_records=enriched_records,
            checkpoint_sha256=sha256_file(args.checkpoint),
            dataset_sha256=sha256_file(args.dataset_root.parent.parent / "pusht_noise.zip"),
            protocol=metrics["protocol"],
        )
        (args.output.parent / f"{args.output.stem}_enriched_export.json").write_text(json.dumps(export, indent=2) + "\n")
    if args.action_mode == "within_episode_shuffle":
        metrics["protocol"]["action_block_permutations"] = action_block_permutations
    metrics["evidence_boundary"] = "Diagnostic only; no learned-world-model generality claim is admitted."
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
