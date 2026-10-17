"""JEPA-WM PointMaze temporal/operator diagnostic on the public split.

The planning wrapper's ``model.encode`` method owns RGB preprocessing: it
divides raw 0--255 tensors by 255 and applies the configured image transform.
The observation dataset used here must therefore stay raw. Passing an
already-transformed dataset tensor through ``model.encode`` applies image
normalization twice and creates latents that an online RGB controller cannot
reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--dinov2-source", type=Path, default=Path("/tmp/dinov2"))
    p.add_argument("--split-seed", type=int, default=42,
                   help="Episode-level split seed used to reconstruct the holdout manifest.")
    p.add_argument("--split-mode", choices=["code_reconstructed", "tail"], default="code_reconstructed")
    p.add_argument("--holdout-start", type=int, default=1800,
                   help="Only used with --split-mode tail.")
    p.add_argument("--holdout-end", type=int, default=2000,
                   help="Only used with --split-mode tail.")
    p.add_argument("--save-vectors", action="store_true",
                   help="Persist per-episode predicted/target latent vectors for post-hoc controls.")
    p.add_argument("--action-mode", choices=["observed", "zero", "reverse", "episode_shuffle", "within_episode_shuffle"], default="observed",
                   help="Action intervention: observed, zero, reversed, cross-episode, or within-episode future-block shuffle.")
    p.add_argument("--action-permutation-seed", type=int, default=None)
    p.add_argument("--context-length", type=int, choices=[1, 2], default=2)
    p.add_argument("--windows-per-episode", type=int, default=1,
                   help="Number of causal temporal windows to export per episode.")
    p.add_argument("--window-stride", type=int, default=5,
                   help="Stride in raw frames between exported windows.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--architecture", default="jepa_wm")
    p.add_argument("--seed", default="unknown")
    p.add_argument("--training-stage", type=int, default=None)
    p.add_argument("--action-distribution", default=None,
                   help="Declared action protocol used for matched-model strata.")
    p.add_argument("--save-enriched-records", action="store_true",
                   help="Save action/state/source-latent records for external metrics.")
    p.add_argument(
        "--visual-source",
        choices=("dataset", "dataset_matched_renderer"),
        default="dataset",
        help=(
            "Use stored RGB episodes or render states with the calibration-only "
            "dataset-matched RGB renderer used by online PointMaze planning."
        ),
    )
    p.add_argument("--renderer-calibration-start", type=int, default=1800)
    p.add_argument("--renderer-calibration-end", type=int, default=1816)
    p.add_argument(
        "--renderer-calibration-frames",
        type=int,
        default=768,
        help="Maximum calibration RGB frames, matching the frozen online-planning default.",
    )
    args = p.parse_args()
    if args.windows_per_episode < 1 or args.window_stride < 1:
        p.error("windows-per-episode and window-stride must be positive")
    if args.action_mode == "within_episode_shuffle" and args.action_permutation_seed is None:
        p.error("--action-permutation-seed is required for within_episode_shuffle")
    return args


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_sweep(x: torch.Tensor, y: torch.Tensor, ranks: list[int]) -> dict[str, dict[str, float]]:
    """Project source/target features into source PCA subspaces before fitting.

    This avoids interpreting a 384-dimensional least-squares fit from only a
    few hundred episodes as a fully identified operator.
    """
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    out = {}
    max_rank = min(x.shape[0] - 1, vh.shape[0])
    for rank in ranks:
        r = min(rank, max_rank)
        if r < 2:
            continue
        basis = vh[:r]
        xr, yr = x @ basis.T, y @ basis.T
        cov = (xr.T @ xr) / max(1, len(xr) - 1)
        op = torch.linalg.lstsq(xr, yr).solution
        observed = float(torch.linalg.matrix_norm(cov @ op - op @ cov) / (torch.linalg.matrix_norm(cov @ op) + 1e-8))
        vals = []
        for seed in range(32):
            q, _ = torch.linalg.qr(torch.randn(r, r, generator=torch.Generator().manual_seed(seed)))
            aq = q @ op @ q.T
            vals.append(float(torch.linalg.matrix_norm(cov @ aq - aq @ cov) / (torch.linalg.matrix_norm(cov @ aq) + 1e-8)))
        out[str(r)] = {"observed": observed, "null_mean": sum(vals) / len(vals)}
    return out


def canonical_pooled_latent(value: torch.Tensor, feature_dim: int = 384) -> torch.Tensor:
    """Pool token/spatial axes while preserving the declared feature axis.

    Older PointMaze exports accidentally reduced a token axis first and wrote
    a 16-dimensional ``source_latent`` next to 384-dimensional predictions.
    Selecting the last axis heuristically is unsafe because intermediate model
    layouts can be ``[tokens, features]`` or ``[channels, tokens]``.  This
    helper identifies the declared feature dimension, moves it last, pools all
    other axes, and fails closed when the representation is ambiguous.
    """
    tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        if tensor.shape[0] != feature_dim:
            raise ValueError(f"source latent has dimension {tuple(tensor.shape)}, expected {feature_dim}")
        return tensor
    feature_axes = [axis for axis, size in enumerate(tensor.shape) if int(size) == feature_dim]
    if len(feature_axes) != 1:
        raise ValueError(
            f"cannot identify unique feature axis in source latent shape {tuple(tensor.shape)} "
            f"for feature_dim={feature_dim}"
        )
    feature_axis = feature_axes[0]
    moved = tensor.movedim(feature_axis, -1).reshape(-1, feature_dim)
    return moved.mean(dim=0)


def build_pointmaze_datasets(dataset_cls, dataset_root: Path):
    """Build statistics plus raw observations for single-pass preprocessing."""

    stats_dataset = dataset_cls(data_path=str(dataset_root), normalize_action=True)
    observation_dataset = dataset_cls(
        data_path=str(dataset_root), transform=None, normalize_action=False
    )
    if getattr(observation_dataset, "transform", None) is not None:
        raise RuntimeError(
            "PointMaze temporal export requires raw observations because "
            "model.encode applies the image transform"
        )
    return stats_dataset, observation_dataset


def rendered_observation(renderer, states: torch.Tensor, frame_ids: list[int]) -> dict[str, torch.Tensor]:
    """Build raw 0--255 CHW observations from simulator states.

    No image transform is applied here.  As with stored dataset observations,
    ``model.encode`` remains the sole owner of visual preprocessing.
    """

    selected = torch.as_tensor(states)[frame_ids].detach().cpu()
    images = np.stack([renderer.render(state.numpy()) for state in selected], axis=0)
    visual = torch.from_numpy(images).permute(0, 3, 1, 2).to(torch.float32)
    return {"visual": visual, "proprio": selected.to(torch.float32)}


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
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

    from app.plan_common.datasets.point_maze_dset import PointMazeDataset
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module
    from tensordict import TensorDict

    config_path = root / "external/jepa-wms/configs/vjepa_wm/mz_sweep/mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml"
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
    stats_dataset, dataset = build_pointmaze_datasets(
        PointMazeDataset, args.dataset_root
    )
    renderer = None
    renderer_calibration_episodes: list[int] = []
    if args.visual_source == "dataset_matched_renderer":
        from experiments.run_live_pointmaze_planning import (
            DatasetMatchedRenderer,
            fit_visual_calibration,
            load_episode,
        )

        renderer_calibration_episodes = list(range(
            int(args.renderer_calibration_start), int(args.renderer_calibration_end)
        ))
        if not renderer_calibration_episodes:
            raise ValueError("renderer calibration range must be non-empty")
        calibration = fit_visual_calibration(
            args.dataset_root,
            renderer_calibration_episodes,
            int(args.renderer_calibration_frames),
        )
        calibration_images, _ = load_episode(
            args.dataset_root, renderer_calibration_episodes[0]
        )
        renderer = DatasetMatchedRenderer.from_episode(calibration_images, calibration)
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
        device,
        action_dim=2,
        proprio_dim=4,
        preprocessor=preprocessor,
        cfgs_data=cfg["data"],
        wrapper_kwargs={"ctxt_window": 1},
    )
    model.eval()
    horizons = sorted(set(args.horizons))
    max_h = max(horizons)
    context_len = args.context_length
    if args.split_mode == "tail":
        start = max(0, min(args.holdout_start, len(dataset)))
        end = max(start, min(args.holdout_end, len(dataset)))
        episodes = list(range(start, min(end, start + args.episodes)))
    else:
        perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
        episodes = perm[-min(args.episodes, len(perm)) :]
    window_specs = []
    for episode in episodes:
        sequence_length = int(dataset.get_seq_length(episode))
        for window_index in range(args.windows_per_episode):
            window_start = window_index * args.window_stride
            last_required = window_start + (context_len + max_h - 1) * 5
            if last_required < sequence_length:
                window_specs.append((episode, window_start))
    per_horizon = {str(h): [] for h in horizons}
    predicted_vectors = {str(h): [] for h in horizons}
    target_vectors = {str(h): [] for h in horizons}
    enriched_records: list[dict[str, object]] = []
    action_block_permutations: list[list[int]] = []
    action_permutation = None
    if args.action_mode == "episode_shuffle" and args.action_permutation_seed is not None:
        from action_interventions import derangement_indices
        action_permutation = derangement_indices(len(window_specs), args.action_permutation_seed)
    with torch.no_grad():
        for offset in range(0, len(window_specs), args.batch_size):
            window_batch = window_specs[offset : offset + args.batch_size]
            episode_batch = [item[0] for item in window_batch]
            window_starts = [item[1] for item in window_batch]
            obs_batch = []
            for episode, window_start in window_batch:
                frame_ids = [window_start + 5 * i for i in range(context_len + max_h)]
                if renderer is None:
                    obs, _, _, _, _ = dataset.get_frames(episode, frame_ids)
                    obs["visual"] = obs["visual"] * 255.0
                else:
                    obs = rendered_observation(renderer, dataset.states[episode], frame_ids)
                obs_batch.append(obs)
            obs = {k: torch.stack([item[k] for item in obs_batch]).to(device) for k in obs_batch[0]}
            encoded = model.encode(obs)
            context = TensorDict(
                {"visual": encoded["visual"][:, :context_len], "proprio": encoded["proprio"][:, :context_len]},
                batch_size=[len(episode_batch), context_len],
            )
            batch_blocks = []
            for local_index, episode in enumerate(episode_batch):
                if args.action_mode == "episode_shuffle":
                    if action_permutation is not None:
                        action_episode, action_window_start = window_specs[action_permutation[offset + local_index]]
                    else:
                        action_episode, action_window_start = window_specs[(offset + local_index + 1) % len(window_specs)]
                else:
                    action_episode = episode
                    action_window_start = window_starts[local_index]
                raw = dataset.actions[action_episode]
                blocks = []
                for h in range(max_h):
                    start = action_window_start + (h + context_len - 1) * 5
                    block = (raw[start : start + 5] - stats_dataset.action_mean) / stats_dataset.action_std
                    blocks.append(block.reshape(-1))
                if args.action_mode == "within_episode_shuffle":
                    from action_interventions import derange_action_blocks
                    action_tensor, permutation = derange_action_blocks(
                        torch.stack(blocks), args.action_permutation_seed or 0, episode, preserve_prefix=0
                    )
                    blocks = list(action_tensor.unbind(0))
                    action_block_permutations.append(permutation)
                batch_blocks.append(torch.stack(blocks))
            if args.action_mode == "zero":
                batch_blocks = [torch.zeros_like(actions) for actions in batch_blocks]
            elif args.action_mode == "reverse":
                batch_blocks = [actions.flip(0) for actions in batch_blocks]
            rollout = model.unroll(context, act_suffix=torch.stack(batch_blocks).transpose(0, 1).to(device))
            for h in horizons:
                pred = rollout["visual"][context_len - 1 + h].mean(dim=1)
                target = encoded["visual"][:, context_len - 1 + h, :, :384].mean(dim=1)
                while pred.ndim > 2:
                    pred = pred.mean(dim=1)
                while target.ndim > 2:
                    target = target.mean(dim=1)
                errors = torch.mean((pred - target) ** 2, dim=1)
                per_horizon[str(h)].extend(errors.tolist())
                predicted_vectors[str(h)].extend(pred.tolist())
                target_vectors[str(h)].extend(target.tolist())
                if args.save_enriched_records:
                    for local_index, episode in enumerate(episode_batch):
                        sample_id = f"{episode}:{window_starts[local_index]}"
                        # Compute this record's source frame on every horizon
                        # iteration.  Reusing a batch-local ``source_index``
                        # from the last newly created record silently wrote
                        # duplicated h=2/h=4 targets when several windows
                        # were processed together.
                        source_index = window_starts[local_index] + (context_len - 1) * 5
                        existing = next((item for item in enriched_records if item["sample_id"] == sample_id), None)
                        if existing is None:
                            source_frame_latent = canonical_pooled_latent(
                                encoded["visual"][local_index, context_len - 1],
                                feature_dim=384,
                            )
                            if int(pred.shape[-1]) != int(source_frame_latent.shape[-1]):
                                raise RuntimeError(
                                    "source/predicted latent feature mismatch after canonicalization: "
                                    f"source={tuple(source_frame_latent.shape)}, pred={tuple(pred.shape)}"
                                )
                            existing = {
                                "sample_id": sample_id,
                                "episode_id": int(episode),
                                "window_start": int(window_starts[local_index]),
                                "source_frame": source_index,
                                "source_state_raw": dataset.states[episode, source_index].detach().cpu().tolist(),
                                "source_proprio_raw": dataset.proprios[episode, source_index].detach().cpu().tolist(),
                                "source_latent": source_frame_latent.detach().cpu().tolist(),
                                "action_blocks_raw": [
                                    dataset.actions[episode, source_index + j * 5 : source_index + (j + 1) * 5]
                                    .reshape(-1).detach().cpu().tolist()
                                    for j in range(max_h)
                                ],
                                "action_blocks_normalized": [
                                    ((dataset.actions[episode, source_index + j * 5 : source_index + (j + 1) * 5]
                                      - stats_dataset.action_mean) / stats_dataset.action_std)
                                    .reshape(-1).detach().cpu().tolist()
                                    for j in range(max_h)
                                ],
                                "latent_by_horizon": {},
                            }
                            enriched_records.append(existing)
                        existing["latent_by_horizon"][str(h)] = {
                            "predicted": pred[local_index].detach().cpu().tolist(),
                            "target": target[local_index].detach().cpu().tolist(),
                            "target_state_raw": dataset.states[episode, source_index + h * 5].detach().cpu().tolist(),
                        }

    commutators, null_means, rank_diagnostics = {}, {}, {}
    for h in horizons:
        p, t = torch.tensor(predicted_vectors[str(h)]), torch.tensor(target_vectors[str(h)])
        x, y = t - t.mean(0, keepdim=True), p - p.mean(0, keepdim=True)
        cov = (x.T @ x) / max(1, len(x) - 1)
        op = torch.linalg.lstsq(x, y).solution
        commutators[str(h)] = float(torch.linalg.matrix_norm(cov @ op - op @ cov) / (torch.linalg.matrix_norm(cov @ op) + 1e-8))
        vals = []
        for seed in range(32):
            q, _ = torch.linalg.qr(torch.randn(cov.shape[0], cov.shape[0], generator=torch.Generator().manual_seed(seed)))
            aq = q @ op @ q.T
            vals.append(float(torch.linalg.matrix_norm(cov @ aq - aq @ cov) / (torch.linalg.matrix_norm(cov @ aq) + 1e-8)))
        null_means[str(h)] = sum(vals) / len(vals)
        rank_diagnostics[str(h)] = rank_sweep(x, y, [8, 16, 32, 64, 128])
    metrics = {
        "latent_prediction_error_by_horizon": {h: {"mean": sum(v) / len(v), "episodes": len(v)} for h, v in per_horizon.items() if v},
        "covariance_operator_commutator_by_horizon": commutators,
        "spectrum_preserving_null_by_horizon": null_means,
        "rank_truncated_commutator_by_horizon": rank_diagnostics,
        "protocol": {
            "context_length": context_len,
            "frame_stride": 5,
            "horizons": horizons,
            "split": (
                f"tail_holdout_{episodes[0]}_{episodes[-1] + 1}"
                if args.split_mode == "tail" and episodes else
                f"code_reconstructed_pointmaze_seed{args.split_seed}"
            ),
            "train_episode_count": len(dataset) - len(episodes),
            "holdout_episode_count": len(episodes),
            "episodes_used": episodes,
            "split_seed": args.split_seed if args.split_mode == "code_reconstructed" else None,
            "split_mode": args.split_mode,
            "visual_preprocessing": "raw_0_255_then_single_model_encode_transform",
            "preprocessing_owner": "model.encode",
            "visual_source": args.visual_source,
            "renderer_calibration_episodes": renderer_calibration_episodes,
            "renderer_calibration_frames": (
                int(args.renderer_calibration_frames) if renderer is not None else None
            ),
            "renderer_boundary": (
                "calibration-only dataset-matched rendering from recorded simulator states; "
                "not untouched raw sensor RGB"
                if renderer is not None else None
            ),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "dataset_archive_sha256": sha256_file(args.dataset_root.parent.parent / "point_maze.zip"),
            "normalization": "Official JEPA-WM PointMazeDataset full-archive statistics.",
            "action_mode": args.action_mode,
            "action_permutation_seed": args.action_permutation_seed,
            "action_block_count": max_h,
            "windows_per_episode": args.windows_per_episode,
            "window_stride": args.window_stride,
            "window_count": len(window_specs),
            "device": str(device),
        },
        "evidence_boundary": "JEPA-WM PointMaze checkpoint diagnostic on a code-reconstructed episode split; not a universal learned-world-model theorem.",
    }
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
            "metric_use": "train-only external task/action metric; episode-disjoint evaluation",
        }
        export = make_payload(
            architecture=args.architecture,
            checkpoint=str(args.checkpoint),
            environment="PointMaze",
            seed=args.seed,
            training_stage=(
                int(args.training_stage)
                if args.training_stage is not None
                else int(checkpoint.get("epoch", -1))
            ),
            latent_dimension=len(enriched_records[0]["source_latent"]),
            action_distribution=(args.action_distribution or f"pointmaze/{args.action_mode}/normalized_dataset_stats"),
            split={
                "name": metrics["protocol"]["split"],
                "episode_ids": [int(item) for item in episodes],
                "split_seed": args.split_seed if args.split_mode == "code_reconstructed" else None,
                "split_mode": args.split_mode,
                "provenance": "JEPA-WM evaluator-defined episode export",
            },
            horizons=horizons,
            enriched_records=enriched_records,
            checkpoint_sha256=sha256_file(args.checkpoint),
            dataset_sha256=sha256_file(args.dataset_root.parent.parent / "point_maze.zip"),
            protocol=metrics["protocol"],
        )
        (args.output.parent / f"{args.output.stem}_enriched_export.json").write_text(json.dumps(export, indent=2) + "\n")
    if args.action_mode == "within_episode_shuffle":
        metrics["protocol"]["action_block_permutations"] = action_block_permutations
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
