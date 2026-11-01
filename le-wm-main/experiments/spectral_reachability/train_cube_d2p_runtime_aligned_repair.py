"""CUDA-only Cube repair whose training example is the deployed P1 decision.

This repair deliberately does *not* reuse the v2 action-swap objective.  The
deployed P1 model observes ``[x1,x2,x3]`` at an already observed root and
must rank a physical candidate ``a3`` by its prediction of ``x4``.  V2
swapped all three context tokens, allowing its source validation to be solved
from historical actions which P1 holds fixed.  Here both the positive and the
negative share ``a1,a2`` and differ only in ``a3``.
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
import h5py
from torch.utils.data import DataLoader, Dataset, Sampler

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_cube_action_adapter_planning import _image_transform
from experiments.spectral_reachability import cube_d2p_training_source as source
from experiments.spectral_reachability import preflight_tworoom_d2p_p0 as runtime
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset


HISTORY_SIZE = 3
FRAME_SKIP = source.ACTION_BLOCK
NUM_STEPS = HISTORY_SIZE + 2  # x0..x4 and a0..a4; deployed terminal is x4.
SCHEMA = "cube_d2p_runtime_aligned_repair.v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _progress(**fields: Any) -> None:
    print(json.dumps({"runtime_aligned_repair_progress": fields}, sort_keys=True), flush=True)


class _EpisodeDisjointClips(Dataset[dict[str, torch.Tensor]]):
    """Episode split rather than a leaky overlapping-clip split."""

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


class _RuntimePoolClips(Dataset[dict[str, torch.Tensor]]):
    """Direct reader for fresh live-runtime clips, not synthetic HDF5 strides."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        with h5py.File(self.path, "r", swmr=True) as handle:
            pixels, action, seeds = handle["pixels"], handle["action"], handle["root_seed"]
            if pixels.ndim != 5 or tuple(pixels.shape[1:2]) != (NUM_STEPS,) or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
                raise ValueError("runtime pool pixels must be [sample,5,H,W,3] uint8")
            if action.shape != (pixels.shape[0], NUM_STEPS, source.ACTION_BLOCK * source.RAW_ACTION_DIM) or action.dtype != np.float32:
                raise ValueError("runtime pool actions must be [sample,5,25] float32")
            if seeds.shape != (pixels.shape[0],) or len(seeds) < 32:
                raise ValueError("runtime pool root seeds are invalid")
            self.root_seeds = np.asarray(seeds, dtype=np.int64)
        self.clip_indices = [(int(index), 0) for index in range(len(self.root_seeds))]
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.root_seeds)

    def _open(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r", swmr=True)
        return self._handle

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        handle = self._open()
        rgb = torch.from_numpy(np.asarray(handle["pixels"][index], dtype=np.uint8)).permute(0, 3, 1, 2).contiguous()
        action = torch.from_numpy(np.asarray(handle["action"][index], dtype=np.float32))
        return {"pixels": rgb, "action": action}


class _DistinctEpisodeBatchSampler(Sampler[list[int]]):
    """One clip per source episode so a shifted candidate is never identical."""

    def __init__(self, dataset: _EpisodeDisjointClips, *, batch_size: int, batches: int, seed: int, advance_epoch: bool) -> None:
        self.batch_size, self.batches, self.seed, self.advance_epoch, self.epoch = batch_size, batches, seed, advance_epoch, 0
        self.by_episode: dict[int, list[int]] = {}
        for local_index, base_index in enumerate(dataset.indices):
            self.by_episode.setdefault(int(dataset.base.clip_indices[base_index][0]), []).append(local_index)
        if batch_size < 2 or batches < 1 or batch_size > len(self.by_episode):
            raise ValueError("candidate-only action swap needs 2..N distinct source episodes per batch")

    def __len__(self) -> int:
        return self.batches

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self.epoch)
        if self.advance_epoch:
            self.epoch += 1
        episodes = np.asarray(sorted(self.by_episode), dtype=np.int64)
        for _ in range(self.batches):
            chosen = generator.choice(episodes, size=self.batch_size, replace=False)
            yield [int(generator.choice(self.by_episode[int(episode)])) for episode in chosen]


def _runtime_rgb(value: torch.Tensor, *, transform: Any, device: torch.device) -> torch.Tensor:
    if value.ndim != 5 or tuple(value.shape[1:3]) != (NUM_STEPS, 3) or value.dtype != torch.uint8:
        raise ValueError(f"expected source Cube RGB [batch,{NUM_STEPS},3,H,W] uint8, got {tuple(value.shape)} {value.dtype}")
    # This is exactly the P1 RGB transform, including its resize behavior.
    return transform(value).to(device, non_blocking=True)


def _normalized_packets(value: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> torch.Tensor:
    width = source.ACTION_BLOCK * source.RAW_ACTION_DIM
    if value.ndim != 3 or tuple(value.shape[1:]) != (NUM_STEPS, width):
        raise ValueError(f"expected source Cube actions [batch,{NUM_STEPS},{width}], got {tuple(value.shape)}")
    raw = value.to(device, dtype=torch.float32, non_blocking=True)
    return torch.nan_to_num((raw - mean) / std)


def _deployed_inputs(rgb: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact deployed ``[x1,x2,x3], [a1,a2,a3]`` tensors."""

    # P1 assembles RGB with ``torch.stack`` and action tokens with ``cat``.
    # A temporal slice is value-equivalent but not stride-equivalent on CUDA;
    # ViT kernels then produced a measurable terminal drift.  Materialize the
    # identical contiguous deployment layout before every trainer rollout.
    return rgb[:, 1:4].contiguous(), action[:, 1:4].contiguous()


def _candidate_only_swap(action: torch.Tensor) -> torch.Tensor:
    """Keep P1's causal history fixed; swap only the physical candidate token."""

    if action.ndim != 3 or action.shape[0] < 2 or action.shape[1] != HISTORY_SIZE:
        raise ValueError("candidate-only swap requires a batched three-token deployed action context")
    swapped = action.clone()
    swapped[:, -1] = action.roll(1, dims=0)[:, -1]
    if not torch.equal(swapped[:, :-1], action[:, :-1]):
        raise RuntimeError("candidate-only swap mutated deployed history")
    return swapped


def _terminal_from_deployed(model: torch.nn.Module, deployed_rgb: torch.Tensor, deployed_action: torch.Tensor) -> torch.Tensor:
    if deployed_rgb.shape[1] != HISTORY_SIZE or deployed_action.shape[1] != HISTORY_SIZE:
        raise ValueError("runtime-aligned terminal requires the deployed three-token context")
    output = model.rollout(
        {"pixels": deployed_rgb[:, None]}, deployed_action[:, None], history_size=HISTORY_SIZE,
    )
    terminal = output["predicted_emb"][:, 0, -1]
    if terminal.ndim != 2 or not torch.isfinite(terminal).all():
        raise RuntimeError("runtime-aligned Cube terminal is invalid")
    return terminal


def _terminal(model: torch.nn.Module, rgb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    deployed_rgb, deployed_action = _deployed_inputs(rgb, action)
    return _terminal_from_deployed(model, deployed_rgb, deployed_action)


def _loss(model: torch.nn.Module, rgb: torch.Tensor, action: torch.Tensor, temperature: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with torch.no_grad():
        target = model.encode({"pixels": rgb[:, 4:5]})["emb"][:, 0].detach()
    deployed_rgb, deployed_action = _deployed_inputs(rgb, action)
    positive_prediction = _terminal_from_deployed(model, deployed_rgb, deployed_action)
    negative_prediction = _terminal_from_deployed(model, deployed_rgb, _candidate_only_swap(deployed_action))
    positive = (positive_prediction - target).square().mean(dim=-1)
    negative = (negative_prediction - target).square().mean(dim=-1)
    mse = positive.mean()
    ranking = F.softplus((positive - negative) / temperature).mean()
    return mse + ranking, {
        "terminal_prediction_mse": mse.detach(),
        "candidate_only_swap_ranking_loss": ranking.detach(),
        "candidate_only_swap_ranking_accuracy": (positive < negative).float().mean().detach(),
        "target_requires_grad": torch.tensor(float(target.requires_grad), device=rgb.device),
    }


def _epoch(*, model: torch.nn.Module, loader: DataLoader[dict[str, torch.Tensor]], mean: torch.Tensor, std: torch.Tensor, transform: Any, device: torch.device, optimizer: torch.optim.Optimizer | None, max_batches: int, epoch: int, split: str) -> dict[str, float]:
    train = optimizer is not None
    for module in (model.predictor, model.action_encoder, model.pred_proj):
        module.train(train)
    totals = {key: 0.0 for key in ("loss", "terminal_prediction_mse", "candidate_only_swap_ranking_loss", "candidate_only_swap_ranking_accuracy")}
    examples = 0
    _progress(phase="epoch_started", epoch=epoch, split=split, max_batches=max_batches)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        rgb = _runtime_rgb(batch["pixels"], transform=transform, device=device)
        action = _normalized_packets(batch["action"], mean=mean, std=std, device=device)
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
            raise RuntimeError("frozen terminal target unexpectedly received a gradient")
        count = int(rgb.shape[0]); examples += count
        totals["loss"] += float(loss.detach()) * count
        for key in totals:
            if key != "loss":
                totals[key] += float(metrics[key]) * count
        if batch_index == 0 or (batch_index + 1) % 64 == 0:
            _progress(phase="batch_completed", epoch=epoch, split=split, completed_batches=batch_index + 1, loss=float(loss.detach()))
    if examples == 0:
        raise RuntimeError(f"empty {split} loader")
    return {key: value / examples for key, value in totals.items()}


def _runtime_alignment_audit(*, model: torch.nn.Module, base: Dataset[dict[str, torch.Tensor]], mean: np.ndarray, std: np.ndarray, transform: Any, device: torch.device, samples: int) -> dict[str, float | int]:
    """Prove the trainer terminal matches P1's raw-physical runtime call."""

    if samples < 1:
        raise ValueError("runtime alignment audit needs at least one sample")
    low = torch.full((source.RAW_ACTION_DIM,), -1.0, device=device)
    high = -low
    errors: list[float] = []
    for index in range(samples):
        item = base[index]
        rgb_u8 = item["pixels"].unsqueeze(0)
        action_raw = item["action"].unsqueeze(0)
        rgb = _runtime_rgb(rgb_u8, transform=transform, device=device)
        action = _normalized_packets(action_raw, mean=torch.as_tensor(mean, device=device).view(1, 1, -1), std=torch.as_tensor(std, device=device).view(1, 1, -1), device=device)
        with torch.no_grad():
            direct = _terminal(model, rgb, action)[0]
            packed = action_raw[0].reshape(NUM_STEPS, source.ACTION_BLOCK, source.RAW_ACTION_DIM)
            runtime_value = runtime._terminal_prediction(
                model, [rgb_u8[0, step].permute(1, 2, 0).numpy() for step in range(3)], rgb_u8[0, 3].permute(1, 2, 0).numpy(), packed[:3].numpy(), packed[3:4].to(device),
                mean=mean, std=std, action_block=source.ACTION_BLOCK, action_dim=source.RAW_ACTION_DIM,
                transform=transform, device=device, low=low, high=high, future_packets_are_raw=True,
            )
        errors.append(float((direct - runtime_value).abs().max().detach().cpu()))
    maximum = max(errors)
    if maximum > 2.0e-5:
        raise RuntimeError(f"trainer/P1 terminal mismatch: max_abs_error={maximum}")
    return {"samples": samples, "max_abs_error": maximum, "tolerance": 2.0e-5}


def _validate_runtime_pool(path: Path, receipt_path: Path, source_receipt: Path) -> dict[str, Any]:
    """Bind a direct live pool to its source action receipt and bytes."""

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("schema") not in {
        "cube_d2p_live_runtime_repair_pool.v1",
        "cube_d2p_live_runtime_repair_pool.v2",
    } or payload.get("status") != "pass":
        raise ValueError("runtime repair pool receipt is not complete")
    pool = payload.get("pool", {})
    source_meta = payload.get("source", {})
    if Path(pool.get("path", "")).resolve() != path or pool.get("sha256") != _sha256(path):
        raise ValueError("runtime repair pool receipt does not bind its data bytes")
    if source_meta.get("receipt_sha256") != _sha256(source_receipt):
        raise ValueError("runtime repair pool used a different action-stat receipt")
    with h5py.File(path, "r", swmr=True) as handle:
        if handle.attrs.get("schema", "") not in {"cube_d2p_live_runtime_repair_pool.v1", "cube_d2p_live_runtime_repair_pool.v2"}:
            raise ValueError("runtime repair pool has an invalid schema")
        if handle.attrs.get("source_receipt_sha256", "") != _sha256(source_receipt):
            raise ValueError("runtime repair pool HDF5 receipt binding differs")
        if handle.attrs.get("candidate_distribution", "source") != pool.get("candidate_distribution", "source"):
            raise ValueError("runtime repair pool candidate distribution differs from its receipt")
    return payload


def train(args: argparse.Namespace) -> dict[str, Any]:
    args.output, args.dataset, args.receipt, args.base_checkpoint = (args.output.resolve(), args.dataset.resolve(), args.receipt.resolve(), args.base_checkpoint.resolve())
    if args.runtime_pool is not None:
        args.runtime_pool, args.runtime_pool_receipt = args.runtime_pool.resolve(), args.runtime_pool_receipt.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable repaired checkpoint: {args.output}")
    if min(args.epochs, args.batches, args.validation_batches, args.alignment_samples) < 1 or args.batch_size < 2 or args.learning_rate <= 0.0:
        raise ValueError("invalid runtime-aligned Cube repair arguments")
    host = require_cuda_allocation("cuda")
    device = torch.device("cuda")
    _seed(args.seed)
    receipt = source.validate_receipt(args.receipt)
    receipt_payload = json.loads(args.receipt.read_text(encoding="utf-8"))
    if args.dataset != Path(receipt_payload["source"]["dataset"]).resolve() or _sha256(args.dataset) != receipt_payload["source"]["dataset_sha256"]:
        raise ValueError("Cube repair data differs from the SHA-bound receipt")
    mean_np = np.asarray(receipt["calibration_action_mean"], dtype=np.float32)
    std_np = np.asarray(receipt["calibration_action_std"], dtype=np.float32)
    mean = torch.as_tensor(mean_np, device=device).view(1, 1, -1)
    std = torch.as_tensor(std_np, device=device).view(1, 1, -1)
    runtime_pool = None
    if args.runtime_pool is None:
        base: Dataset[dict[str, torch.Tensor]] = HDF5Dataset(path=args.dataset, frameskip=FRAME_SKIP, num_steps=NUM_STEPS, transform=None, keys_to_load=["pixels", "action"])
    else:
        runtime_pool = _validate_runtime_pool(args.runtime_pool, args.runtime_pool_receipt, args.receipt)
        base = _RuntimePoolClips(args.runtime_pool)
    train_set, validation_set = _EpisodeDisjointClips(base, split="train"), _EpisodeDisjointClips(base, split="validation")
    if set(train_set.episode_ids) & set(validation_set.episode_ids):
        raise RuntimeError("Cube runtime-aligned repair episode split overlaps")
    train_loader = DataLoader(train_set, batch_sampler=_DistinctEpisodeBatchSampler(train_set, batch_size=args.batch_size, batches=args.batches, seed=args.seed, advance_epoch=True), num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_sampler=_DistinctEpisodeBatchSampler(validation_set, batch_size=args.batch_size, batches=args.validation_batches, seed=args.seed + 1, advance_epoch=False), num_workers=0, pin_memory=True)
    model = load_compatible_pretrained(args.base_checkpoint, device=device).eval()
    for module in (model.encoder, model.projector):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for module in (model.predictor, model.action_encoder, model.pred_proj) for parameter in module.parameters()]
    if not trainable:
        raise RuntimeError("no trainable Cube dynamics parameters")
    for parameter in trainable:
        parameter.requires_grad_(True)
    transform = _image_transform()
    alignment = _runtime_alignment_audit(model=model, base=base, mean=mean_np, std=std_np, transform=transform, device=device, samples=args.alignment_samples)
    _progress(phase="runtime_alignment_passed", **alignment)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1.0e-4)
    history: list[dict[str, Any]] = []; best_key = (-float("inf"), float("inf")); best_state = None; best_epoch = 0; started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = _epoch(model=model, loader=train_loader, mean=mean, std=std, transform=transform, device=device, optimizer=optimizer, max_batches=args.batches, epoch=epoch, split="train")
        validation_metrics = _epoch(model=model, loader=validation_loader, mean=mean, std=std, transform=transform, device=device, optimizer=None, max_batches=args.validation_batches, epoch=epoch, split="validation")
        key = (validation_metrics["candidate_only_swap_ranking_accuracy"], -validation_metrics["terminal_prediction_mse"])
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        _progress(phase="epoch_completed", epoch=epoch, validation=validation_metrics)
    if best_state is None:
        raise RuntimeError("runtime-aligned Cube repair produced no checkpoint candidate")
    args.output.mkdir(parents=True)
    torch.save(best_state, args.output / "weights.pt")
    shutil.copy2(args.base_checkpoint / "config.json", args.output / "config.json")
    mode = "runtime_pool" if runtime_pool is not None else "source_only"
    manifest = {"schema": SCHEMA, "status": "trained_runtime_aligned_repair" if runtime_pool is not None else "trained_train_source_only", "base_checkpoint": str(args.base_checkpoint), "base_weights_sha256": _sha256(args.base_checkpoint / "weights.pt"), "dataset": str(args.dataset), "dataset_sha256": _sha256(args.dataset), "receipt": str(args.receipt), "receipt_sha256": _sha256(args.receipt), "runtime_pool": None if runtime_pool is None else {"path": str(args.runtime_pool), "sha256": _sha256(args.runtime_pool), "receipt": str(args.runtime_pool_receipt), "receipt_sha256": _sha256(args.runtime_pool_receipt), "root_seed_range": runtime_pool["pool"]["root_seed_range"], "candidate_distribution": runtime_pool["pool"].get("candidate_distribution", "source")}, "device": host, "runtime_alignment": alignment, "training_contract": {"mode": mode, "visual_encoder_and_projector": "frozen", "deployed_mapping": "[x1,x2,x3] + [a1,a2,a3] -> x4 via model.rollout(history_size=3)", "negative": "same [a1,a2], candidate-only a3 swap across distinct episodes", "candidate_distribution": "source" if runtime_pool is None else runtime_pool["pool"].get("candidate_distribution", "source"), "trainable_modules": ["predictor", "action_encoder", "pred_proj"], "p0_p1_roots_used": False, "cpu_fallback": False}, "episode_split": {"rule": "root index modulo 10; slots 0..7 train, 8..9 validation", "train_root_count": len(train_set.episode_ids), "validation_root_count": len(validation_set.episode_ids), "overlap": False}, "optimization": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "train_batches_per_epoch": args.batches, "validation_batches": args.validation_batches, "validation_panel": "fixed seeded root-disjoint panel"}, "epochs": history, "selected_epoch": best_epoch, "selected_validation": history[best_epoch - 1]["validation"], "elapsed_seconds": time.time() - started, "claim_boundary": "This is a runtime-distribution repair, not a frozen-model D2P result. A disjoint CUDA P0/P1 chain is mandatory before any prediction claim."}
    (args.output / "repair_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected_validation": manifest["selected_validation"], "status": manifest["status"]}, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--receipt", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/cube/lewm")); parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--batches", type=int, default=256); parser.add_argument("--validation-batches", type=int, default=128); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--learning-rate", type=float, default=2e-5); parser.add_argument("--seed", type=int, default=202610325); parser.add_argument("--alignment-samples", type=int, default=4); parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--runtime-pool", type=Path); parser.add_argument("--runtime-pool-receipt", type=Path)
    args = parser.parse_args()
    if (args.runtime_pool is None) != (args.runtime_pool_receipt is None):
        parser.error("--runtime-pool and --runtime-pool-receipt must be supplied together")
    train(args)


if __name__ == "__main__":
    main()
