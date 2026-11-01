"""Source-only, CUDA-only repair of Cube five-action successor ranking.

The released visual representation stays frozen.  This trains a new dynamics
checkpoint from the official HDF5 only, selecting it strictly on episode-
disjoint source validation rather than on its optimisation batches.  It must
still clear a fresh live P0/P1 protocol before it can be used for planning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability import cube_d2p_training_source as source
from stable_pretraining import data as spt_data
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset


HISTORY_SIZE = 3
FRAME_SKIP = source.ACTION_BLOCK
NUM_STEPS = HISTORY_SIZE + 1
SCHEMA = "cube_d2p_causal_dynamics_repair.v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(**fields: Any) -> None:
    print(json.dumps({"repair_progress": fields}, sort_keys=True), flush=True)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _EpisodeDisjointClips(Dataset[dict[str, torch.Tensor]]):
    """An episode split, never an overlapping clip-row split."""

    def __init__(self, base: HDF5Dataset, *, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(f"unknown split {split}")
        want_train = split == "train"
        self.base = base
        self.indices = [
            index for index, (episode, _) in enumerate(base.clip_indices)
            if ((int(episode) % 10) < 8) == want_train
        ]
        self.episode_ids = sorted({int(base.clip_indices[index][0]) for index in self.indices})
        if not self.indices:
            raise ValueError(f"empty {split} Cube episode split")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.base[self.indices[index]]


class _EpisodeBatchSampler(Sampler[list[int]]):
    """Use multiple source episodes per batch and a fixed validation panel."""

    def __init__(self, dataset: _EpisodeDisjointClips, *, batch_size: int, batches: int, seed: int, advance_epoch: bool) -> None:
        if batch_size < 2 or batches < 1:
            raise ValueError("batch size and batch count must be positive")
        self.batch_size, self.batches, self.seed, self.advance_epoch = batch_size, batches, seed, advance_epoch
        self.epoch = 0
        self.by_episode: dict[int, list[int]] = {}
        for local_index, base_index in enumerate(dataset.indices):
            self.by_episode.setdefault(int(dataset.base.clip_indices[base_index][0]), []).append(local_index)
        if len(self.by_episode) < 2:
            raise ValueError("action-swap ranking requires at least two source episodes")

    def __len__(self) -> int:
        return self.batches

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self.epoch)
        if self.advance_epoch:
            self.epoch += 1
        episodes = np.asarray(sorted(self.by_episode), dtype=np.int64)
        for _ in range(self.batches):
            selected = generator.choice(episodes, size=self.batch_size, replace=True)
            yield [int(generator.choice(self.by_episode[int(episode)])) for episode in selected]


def _pixels(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 5 or value.shape[1] != NUM_STEPS or value.shape[2] != 3:
        raise ValueError(f"expected Cube RGB [batch,{NUM_STEPS},3,H,W], got {tuple(value.shape)}")
    stats = spt_data.dataset_stats.ImageNet
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=value.device).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(stats["std"], dtype=torch.float32, device=value.device).view(1, 1, 3, 1, 1)
    return value.to(dtype=torch.float32).div_(255.0).sub_(mean).div_(std)


def _loss(model: torch.nn.Module, rgb: torch.Tensor, action: torch.Tensor, temperature: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if action.ndim != 3 or tuple(action.shape[1:]) != (NUM_STEPS, source.ACTION_BLOCK * source.RAW_ACTION_DIM):
        raise ValueError(f"expected Cube action [batch,{NUM_STEPS},25], got {tuple(action.shape)}")
    with torch.no_grad():
        embedding = model.encode({"pixels": rgb})["emb"].detach()
    context, target = embedding[:, :HISTORY_SIZE], embedding[:, 1:].detach()
    prediction = model.predict(context, model.action_encoder(action[:, :HISTORY_SIZE]))
    prediction_mse = F.mse_loss(prediction, target)
    positive = (prediction[:, -1] - target[:, -1]).square().mean(dim=-1)
    swapped = model.predict(context, model.action_encoder(action.roll(1, 0)[:, :HISTORY_SIZE]))
    negative = (swapped[:, -1] - target[:, -1]).square().mean(dim=-1)
    ranking = F.softplus((positive - negative) / temperature).mean()
    return prediction_mse + ranking, {
        "prediction_mse": prediction_mse.detach(),
        "action_swap_ranking_loss": ranking.detach(),
        "action_swap_ranking_accuracy": (positive < negative).float().mean().detach(),
        "target_requires_grad": torch.tensor(float(target.requires_grad), device=rgb.device),
    }


def _epoch(*, model: torch.nn.Module, loader: DataLoader[dict[str, torch.Tensor]], mean: torch.Tensor, std: torch.Tensor, optimizer: torch.optim.Optimizer | None, max_batches: int, epoch: int, split: str) -> dict[str, float]:
    train = optimizer is not None
    for module in (model.predictor, model.action_encoder, model.pred_proj):
        module.train(train)
    totals = {key: 0.0 for key in ("loss", "prediction_mse", "action_swap_ranking_loss", "action_swap_ranking_accuracy")}
    examples = 0
    _progress(phase="epoch_started", epoch=epoch, split=split, max_batches=max_batches)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        rgb = _pixels(batch["pixels"].to("cuda", non_blocking=True))
        action = torch.nan_to_num((batch["action"].to("cuda", dtype=torch.float32, non_blocking=True) - mean) / std)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _loss(model, rgb, action, 0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                loss, metrics = _loss(model, rgb, action, 0.1)
        if float(metrics["target_requires_grad"]) != 0.0:
            raise RuntimeError("Cube successor target unexpectedly received a dynamics gradient")
        count = int(rgb.shape[0]); examples += count
        totals["loss"] += float(loss.detach()) * count
        for key in tuple(metrics):
            if key != "target_requires_grad":
                totals[key] += float(metrics[key]) * count
        if batch_index == 0 or (batch_index + 1) % 64 == 0:
            _progress(phase="batch_completed", epoch=epoch, split=split, completed_batches=batch_index + 1, loss=float(loss.detach()))
    if examples == 0:
        raise RuntimeError(f"empty {split} loader")
    return {key: value / examples for key, value in totals.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    args.output = args.output.resolve(); args.dataset = args.dataset.resolve(); args.receipt = args.receipt.resolve(); args.base_checkpoint = args.base_checkpoint.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable repaired checkpoint: {args.output}")
    if min(args.epochs, args.batches, args.validation_batches) < 1 or args.batch_size < 2 or args.learning_rate <= 0.0:
        raise ValueError("invalid Cube repair optimisation arguments")
    host = require_cuda_allocation("cuda")
    _seed(args.seed)
    receipt = source.validate_receipt(args.receipt)
    receipt_payload = json.loads(args.receipt.read_text(encoding="utf-8"))
    receipt_source = receipt_payload["source"]
    expected_dataset = Path(receipt_source["dataset"]).resolve()
    if args.dataset != expected_dataset or _sha256(args.dataset) != receipt_source["dataset_sha256"]:
        raise ValueError("Cube repair dataset differs from its SHA-bound receipt")
    mean = torch.as_tensor(receipt["calibration_action_mean"], dtype=torch.float32, device="cuda").view(1, 1, -1)
    std = torch.as_tensor(receipt["calibration_action_std"], dtype=torch.float32, device="cuda").view(1, 1, -1)
    base = HDF5Dataset(path=args.dataset, frameskip=FRAME_SKIP, num_steps=NUM_STEPS, transform=None, keys_to_load=["pixels", "action"])
    train_set, validation_set = _EpisodeDisjointClips(base, split="train"), _EpisodeDisjointClips(base, split="validation")
    if set(train_set.episode_ids) & set(validation_set.episode_ids):
        raise RuntimeError("Cube repair episode split overlaps")
    train_loader = DataLoader(train_set, batch_sampler=_EpisodeBatchSampler(train_set, batch_size=args.batch_size, batches=args.batches, seed=args.seed, advance_epoch=True), num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_sampler=_EpisodeBatchSampler(validation_set, batch_size=args.batch_size, batches=args.validation_batches, seed=args.seed + 1, advance_epoch=False), num_workers=0, pin_memory=True)
    _progress(phase="datasets_ready", train_clips=len(train_set), validation_clips=len(validation_set), train_episodes=len(train_set.episode_ids), validation_episodes=len(validation_set.episode_ids))
    model = load_compatible_pretrained(args.base_checkpoint, device="cuda").eval()
    for module in (model.encoder, model.projector):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    trainable = []
    for module in (model.predictor, model.action_encoder, model.pred_proj):
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True); trainable.append(parameter)
    if not trainable: raise RuntimeError("no trainable Cube dynamics parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1.0e-4)
    history: list[dict[str, Any]] = []; best_key = (-float("inf"), float("inf")); best_state = None; best_epoch = 0; started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = _epoch(model=model, loader=train_loader, mean=mean, std=std, optimizer=optimizer, max_batches=args.batches, epoch=epoch, split="train")
        validation_metrics = _epoch(model=model, loader=validation_loader, mean=mean, std=std, optimizer=None, max_batches=args.validation_batches, epoch=epoch, split="validation")
        key = (validation_metrics["action_swap_ranking_accuracy"], -validation_metrics["prediction_mse"])
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        _progress(phase="epoch_completed", epoch=epoch, validation=validation_metrics)
    if best_state is None: raise RuntimeError("Cube repair produced no checkpoint candidate")
    args.output.mkdir(parents=True)
    torch.save(best_state, args.output / "weights.pt")
    shutil.copy2(args.base_checkpoint / "config.json", args.output / "config.json")
    manifest = {"schema": SCHEMA, "status": "trained_train_source_only", "base_checkpoint": str(args.base_checkpoint), "base_weights_sha256": _sha256(args.base_checkpoint / "weights.pt"), "dataset": str(args.dataset), "dataset_sha256": _sha256(args.dataset), "receipt": str(args.receipt), "receipt_sha256": _sha256(args.receipt), "device": host, "training_contract": {"visual_encoder_and_projector": "frozen", "future_successor_target": "detached frozen-checkpoint RGB latent", "trainable_modules": ["predictor", "action_encoder", "pred_proj"], "loss": "all-position successor MSE plus last-successor true-action vs in-batch action-swap ranking", "p0_p1_roots_used": False, "cpu_fallback": False}, "episode_split": {"rule": "episode_index modulo 10; slots 0..7 train, 8..9 validation", "train_episode_count": len(train_set.episode_ids), "validation_episode_count": len(validation_set.episode_ids), "overlap": False}, "optimization": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "train_batches_per_epoch": args.batches, "validation_batches": args.validation_batches, "validation_panel": "fixed seeded episode-disjoint panel"}, "epochs": history, "selected_epoch": best_epoch, "selected_validation": history[best_epoch - 1]["validation"], "elapsed_seconds": time.time() - started, "claim_boundary": "Training-source-only repair checkpoint; requires fresh CUDA P0, P1 calibration, and root-disjoint outcome gates before any planning claim."}
    (args.output / "repair_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected_validation": manifest["selected_validation"], "status": manifest["status"]}, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--receipt", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/cube/lewm")); parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--batches", type=int, default=256); parser.add_argument("--validation-batches", type=int, default=128); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--learning-rate", type=float, default=2e-5); parser.add_argument("--seed", type=int, default=202610301); parser.add_argument("--device", choices=("cuda",), default="cuda")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
