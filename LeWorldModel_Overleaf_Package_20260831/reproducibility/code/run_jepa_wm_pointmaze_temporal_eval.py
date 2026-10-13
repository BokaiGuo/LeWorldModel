"""JEPA-WM PointMaze temporal/operator diagnostic on the public split."""

from __future__ import annotations

import argparse
import hashlib
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
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--dinov2-source", type=Path, default=Path("/tmp/dinov2"))
    return p.parse_args()


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
    stats_dataset = PointMazeDataset(data_path=str(args.dataset_root), normalize_action=True)
    dataset = PointMazeDataset(
        data_path=str(args.dataset_root), transform=transform, normalize_action=False
    )
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
        proprio_dim=4,
        preprocessor=preprocessor,
        cfgs_data=cfg["data"],
        wrapper_kwargs={"ctxt_window": 1},
    )
    model.eval()
    horizons = sorted(set(args.horizons))
    max_h = max(horizons)
    context_len = 2
    perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    episodes = perm[-min(args.episodes, len(perm)) :]
    per_horizon = {str(h): [] for h in horizons}
    predicted_vectors = {str(h): [] for h in horizons}
    target_vectors = {str(h): [] for h in horizons}
    with torch.no_grad():
        for offset in range(0, len(episodes), args.batch_size):
            episode_batch = [e for e in episodes[offset : offset + args.batch_size]
                             if int(dataset.get_seq_length(e)) >= (context_len + max_h) * 5]
            if not episode_batch:
                continue
            frame_ids = [5 * i for i in range(context_len + max_h)]
            obs_batch = []
            for episode in episode_batch:
                obs, _, _, _, _ = dataset.get_frames(episode, frame_ids)
                obs["visual"] = obs["visual"] * 255.0
                obs_batch.append(obs)
            obs = {k: torch.stack([item[k] for item in obs_batch]) for k in obs_batch[0]}
            encoded = model.encode(obs)
            context = TensorDict(
                {"visual": encoded["visual"][:, :context_len], "proprio": encoded["proprio"][:, :context_len]},
                batch_size=[len(episode_batch), context_len],
            )
            batch_blocks = []
            for episode in episode_batch:
                raw = dataset.actions[episode]
                blocks = []
                for h in range(max_h):
                    start = (h + context_len - 1) * 5
                    block = (raw[start : start + 5] - stats_dataset.action_mean) / stats_dataset.action_std
                    blocks.append(block.reshape(-1))
                batch_blocks.append(torch.stack(blocks))
            rollout = model.unroll(context, act_suffix=torch.stack(batch_blocks).transpose(0, 1))
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
            "split": "code_reconstructed_pointmaze_seed42",
            "train_episode_count": len(dataset) - len(episodes),
            "holdout_episode_count": len(episodes),
            "episodes_used": episodes,
            "split_seed": 42,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "dataset_archive_sha256": sha256_file(args.dataset_root.parent.parent / "point_maze.zip"),
            "normalization": "Official JEPA-WM PointMazeDataset full-archive statistics.",
        },
        "evidence_boundary": "JEPA-WM PointMaze checkpoint diagnostic on a code-reconstructed episode split; not a universal learned-world-model theorem.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
