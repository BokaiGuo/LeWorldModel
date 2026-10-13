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
    return p.parse_args()


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
    device = torch.device("cpu")
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
    per_horizon: dict[str, list[float]] = {str(h): [] for h in horizons}
    predicted_vectors: dict[str, list[list[float]]] = {str(h): [] for h in horizons}
    target_vectors: dict[str, list[list[float]]] = {str(h): [] for h in horizons}

    with torch.no_grad():
        for episode in range(min(args.episodes, len(dataset))):
            length = int(dataset.get_seq_length(episode))
            if length < (max_h + 1) * 5:
                continue
            context_len = 2
            frame_ids = [5 * h for h in range(context_len + max_h)]
            obs, _, _, _, _ = dataset.get_frames(episode, frame_ids)
            obs["visual"] = obs["visual"] * 255.0  # wrapper performs /255 once
            encoded = model.encode(
                {"visual": obs["visual"].unsqueeze(0), "proprio": obs["proprio"].unsqueeze(0)}
            )
            context = TensorDict(
                {"visual": encoded["visual"][:, :context_len], "proprio": encoded["proprio"][:, :context_len]},
                batch_size=[1, context_len],
            )
            raw = dataset.actions[episode]
            blocks = []
            for h in range(max_h):
                # The first predicted frame follows the last context frame;
                # therefore consume action blocks starting at frame 1.
                start = (h + context_len - 1) * 5
                block = raw[start : start + 5] / ACTION_STD
                blocks.append(block.reshape(-1))
            act_suffix = torch.stack(blocks).unsqueeze(1)
            rollout = model.unroll(context, act_suffix=act_suffix)
            for h in horizons:
                # `unroll` returns the original context followed by predicted
                # frames.  Horizon one is therefore at index context_len,
                # not index one.
                pred = rollout["visual"][context_len - 1 + h, 0].mean(dim=(0, 1, 2))
                target = encoded["visual"][0, h].mean(dim=(0, 1, 2))
                per_horizon[str(h)].append(float(torch.mean((pred - target) ** 2)))
                predicted_vectors[str(h)].append(pred.tolist())
                target_vectors[str(h)].append(target.tolist())

    metrics: dict[str, object] = {
        "latent_prediction_error_by_horizon": {
            h: {"mean": sum(v) / len(v), "episodes": len(v)} for h, v in per_horizon.items() if v
        },
        "protocol": {
                "context_length": 2,
            "frame_stride": 5,
            "horizons": horizons,
            "split": "val",
            "device": "cpu",
            "checkpoint_epoch": 50,
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
    metrics["evidence_boundary"] = "Diagnostic only; no learned-world-model generality claim is admitted."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
