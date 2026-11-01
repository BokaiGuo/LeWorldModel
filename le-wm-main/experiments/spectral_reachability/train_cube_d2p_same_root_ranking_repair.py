"""CUDA-only Cube repair with the exact P1 same-root top-one ranking contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_cube_action_adapter_planning import _image_transform
from experiments.spectral_reachability import cube_d2p_training_source as source
from experiments.spectral_reachability import preflight_tworoom_d2p_p0 as runtime
from experiments.spectral_reachability.build_cube_d2p_same_root_ranking_pool import SCHEMA as POOL_SCHEMA


HISTORY_SIZE = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(**fields: Any) -> None:
    print(json.dumps({"same_root_ranking_repair_progress": fields}, sort_keys=True), flush=True)


def _seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


class _Groups(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: Path, *, split: str) -> None:
        self.path = path.resolve(); self._handle: h5py.File | None = None
        with h5py.File(self.path, "r", swmr=True) as handle:
            context, prefix, candidates, successors, seeds = (handle[key] for key in ("context_rgb", "prefix_packets", "candidate_packets", "successor_rgb", "root_seed"))
            if context.ndim != 5 or context.shape[1] != HISTORY_SIZE + 1 or context.shape[-1] != 3 or context.dtype != np.uint8:
                raise ValueError("same-root ranking context layout is invalid")
            if prefix.shape[:2] != (context.shape[0], HISTORY_SIZE) or prefix.shape[-2:] != (source.ACTION_BLOCK, source.RAW_ACTION_DIM):
                raise ValueError("same-root ranking prefix layout is invalid")
            if candidates.shape[0] != context.shape[0] or candidates.shape[1] < 2 or candidates.shape[-2:] != (source.ACTION_BLOCK, source.RAW_ACTION_DIM):
                raise ValueError("same-root ranking candidate layout is invalid")
            if successors.shape[:2] != candidates.shape[:2] or successors.shape[2:] != context.shape[2:] or successors.dtype != np.uint8:
                raise ValueError("same-root ranking successor layout is invalid")
            self.indices = [i for i, seed in enumerate(np.asarray(seeds)) if (int(seed) % 10 < 8) == (split == "train")]
            self.root_seeds = np.asarray(seeds, dtype=np.int64)
            self.candidates = int(candidates.shape[1])
        if split not in {"train", "validation"} or not self.indices:
            raise ValueError("same-root ranking split is empty or invalid")

    def __len__(self) -> int: return len(self.indices)
    def _open(self) -> h5py.File:
        if self._handle is None: self._handle = h5py.File(self.path, "r", swmr=True)
        return self._handle
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.indices[index]; handle = self._open()
        return {
            "context": torch.from_numpy(np.asarray(handle["context_rgb"][row], dtype=np.uint8)).permute(0, 3, 1, 2).contiguous(),
            "prefix": torch.from_numpy(np.asarray(handle["prefix_packets"][row], dtype=np.float32)),
            "candidates": torch.from_numpy(np.asarray(handle["candidate_packets"][row], dtype=np.float32)),
            "successors": torch.from_numpy(np.asarray(handle["successor_rgb"][row], dtype=np.uint8)).permute(0, 3, 1, 2).contiguous(),
        }


def _rgb(value: torch.Tensor, *, transform: Any, device: torch.device) -> torch.Tensor:
    if value.dtype != torch.uint8 or value.ndim not in {4, 5} or value.shape[-3] != 3:
        raise ValueError("same-root ranking RGB tensor is malformed")
    return transform(value).to(device, non_blocking=True)


def _normalise(value: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> torch.Tensor:
    raw = value.to(device, dtype=torch.float32, non_blocking=True)
    return torch.nan_to_num((raw.reshape(*raw.shape[:-2], -1) - mean) / std)


def _terminal(model: torch.nn.Module, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if context.shape[1] != HISTORY_SIZE + 1 or actions.shape[1] != HISTORY_SIZE:
        raise ValueError("same-root ranking deployed window is malformed")
    # Training uses mini-batches, but checkpoint selection must use the exact
    # singleton P1 kernel/layout.  CUDA batch-size drift is small but is not
    # allowed to silently change which checkpoint is selected for deployment.
    if not model.predictor.training and context.shape[0] > 1:
        return torch.cat([
            _terminal(model, context[index:index + 1].contiguous(), actions[index:index + 1].contiguous())
            for index in range(context.shape[0])
        ], dim=0)
    output = model.rollout({"pixels": context[:, 1:4].contiguous()[:, None]}, actions.contiguous()[:, None], history_size=HISTORY_SIZE)
    terminal = output["predicted_emb"][:, 0, -1]
    if terminal.ndim != 2 or not torch.isfinite(terminal).all(): raise RuntimeError("same-root ranking terminal is invalid")
    return terminal


def _loss(model: torch.nn.Module, batch: dict[str, torch.Tensor], *, mean: torch.Tensor, std: torch.Tensor, transform: Any, device: torch.device, temperature: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    context = _rgb(batch["context"], transform=transform, device=device)
    successors = _rgb(batch["successors"].flatten(0, 1), transform=transform, device=device)
    prefix = _normalise(batch["prefix"], mean=mean, std=std, device=device)
    candidates = _normalise(batch["candidates"], mean=mean, std=std, device=device)
    b, c = candidates.shape[:2]
    action = torch.cat((prefix[:, 1:].unsqueeze(1).expand(-1, c, -1, -1), candidates.unsqueeze(2)), dim=2).flatten(0, 1)
    deployed_context = context.unsqueeze(1).expand(-1, c, -1, -1, -1, -1).flatten(0, 1)
    with torch.no_grad():
        target = model.encode({"pixels": successors[:, None]})["emb"][:, 0].detach()
        goal = target.reshape(b, c, -1)[:, 0]
    prediction = _terminal(model, deployed_context, action)
    mse = (prediction - target).square().mean(dim=-1).mean()
    cost = (prediction.reshape(b, c, -1) - goal[:, None]).square().mean(dim=-1)
    ranking = F.softplus((cost[:, :1] - cost[:, 1:]) / temperature).mean()
    return mse + ranking, {"successor_mse": mse.detach(), "same_root_top1": (cost.argmin(dim=1) == 0).float().mean().detach(), "same_root_pair_accuracy": (cost[:, :1] < cost[:, 1:]).float().mean().detach(), "ranking_loss": ranking.detach()}


def _epoch(*, model: torch.nn.Module, loader: DataLoader[dict[str, torch.Tensor]], mean: torch.Tensor, std: torch.Tensor, transform: Any, device: torch.device, optimizer: torch.optim.Optimizer | None, epoch: int, split: str) -> dict[str, float]:
    train = optimizer is not None
    for module in (model.predictor, model.action_encoder, model.pred_proj): module.train(train)
    totals = {key: 0.0 for key in ("loss", "successor_mse", "same_root_top1", "same_root_pair_accuracy", "ranking_loss")}; count = 0
    for batch_index, batch in enumerate(loader):
        if train:
            optimizer.zero_grad(set_to_none=True); loss, metrics = _loss(model, batch, mean=mean, std=std, transform=transform, device=device, temperature=0.1); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); optimizer.step()
        else:
            with torch.no_grad(): loss, metrics = _loss(model, batch, mean=mean, std=std, transform=transform, device=device, temperature=0.1)
        size = int(batch["context"].shape[0]); count += size; totals["loss"] += float(loss.detach()) * size
        for key in metrics: totals[key] += float(metrics[key]) * size
        if batch_index == 0 or (batch_index + 1) % 32 == 0: _progress(phase="batch_completed", epoch=epoch, split=split, completed_batches=batch_index + 1, loss=float(loss.detach()))
    return {key: value / count for key, value in totals.items()}


def _validate_pool(path: Path, receipt_path: Path, source_receipt: Path) -> dict[str, Any]:
    payload = json.loads(receipt_path.read_text(encoding="utf-8")); pool = payload.get("pool", {})
    if payload.get("schema") != POOL_SCHEMA or payload.get("status") != "pass" or Path(pool.get("path", "")).resolve() != path or pool.get("sha256") != _sha256(path): raise ValueError("same-root ranking pool receipt is invalid")
    if payload.get("source", {}).get("receipt_sha256") != _sha256(source_receipt) or pool.get("positive_candidate_index") != 0: raise ValueError("same-root ranking pool has invalid source/positive binding")
    with h5py.File(path, "r", swmr=True) as handle:
        if handle.attrs.get("schema", "") != POOL_SCHEMA or int(handle.attrs.get("positive_candidate_index", -1)) != 0 or handle.attrs.get("source_receipt_sha256", "") != _sha256(source_receipt): raise ValueError("same-root ranking HDF5 binding is invalid")
    return payload


def train(args: argparse.Namespace) -> dict[str, Any]:
    args.output, args.dataset, args.receipt, args.pool, args.pool_receipt, args.base_checkpoint = (value.resolve() for value in (args.output, args.dataset, args.receipt, args.pool, args.pool_receipt, args.base_checkpoint))
    if args.output.exists() or min(args.epochs, args.batch_size) < 1 or args.batch_size < 2: raise ValueError("same-root ranking arguments are invalid or output exists")
    host = require_cuda_allocation("cuda"); device = torch.device("cuda"); _seed(args.seed)
    receipt = source.validate_receipt(args.receipt); receipt_payload = json.loads(args.receipt.read_text(encoding="utf-8"))
    if args.dataset != Path(receipt_payload["source"]["dataset"]).resolve() or _sha256(args.dataset) != receipt_payload["source"]["dataset_sha256"]: raise ValueError("same-root ranking dataset differs from source receipt")
    pool_meta = _validate_pool(args.pool, args.pool_receipt, args.receipt)
    train_set, validation_set = _Groups(args.pool, split="train"), _Groups(args.pool, split="validation")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True); validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    mean = torch.as_tensor(np.asarray(receipt["calibration_action_mean"], np.float32), device=device).view(1, 1, -1); std = torch.as_tensor(np.asarray(receipt["calibration_action_std"], np.float32), device=device).view(1, 1, -1)
    model = load_compatible_pretrained(args.base_checkpoint, device=device).eval()
    for module in (model.encoder, model.projector):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    trainable = [p for module in (model.predictor, model.action_encoder, model.pred_proj) for p in module.parameters()]
    for parameter in trainable: parameter.requires_grad_(True)
    transform = _image_transform(); optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    history: list[dict[str, Any]] = []; best_key = (-float("inf"), -float("inf")); best_state = None; best_epoch = 0; started = time.time()
    patience = getattr(args, 'patience', 0)
    if patience < 0:
        raise ValueError('validation patience must be nonnegative')
    for epoch in range(1, args.epochs + 1):
        _progress(phase="epoch_started", epoch=epoch, split="train"); train_metrics = _epoch(model=model, loader=train_loader, mean=mean, std=std, transform=transform, device=device, optimizer=optimizer, epoch=epoch, split="train")
        _progress(phase="epoch_started", epoch=epoch, split="validation"); validation_metrics = _epoch(model=model, loader=validation_loader, mean=mean, std=std, transform=transform, device=device, optimizer=None, epoch=epoch, split="validation")
        key = (validation_metrics["same_root_top1"], validation_metrics["same_root_pair_accuracy"])
        if key > best_key: best_key, best_epoch, best_state = key, epoch, {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics}); _progress(phase="epoch_completed", epoch=epoch, validation=validation_metrics)
        if patience and epoch - best_epoch >= patience:
            _progress(phase='validation_patience_exhausted', epoch=epoch, selected_epoch=best_epoch, patience=patience)
            break
    if best_state is None: raise RuntimeError("same-root ranking repair produced no checkpoint")
    args.output.mkdir(parents=True); torch.save(best_state, args.output / "weights.pt"); shutil.copy2(args.base_checkpoint / "config.json", args.output / "config.json")
    manifest = {"schema": "cube_d2p_same_root_ranking_repair.v1", "status": "trained_same_root_ranking_repair", "base_checkpoint": str(args.base_checkpoint), "base_weights_sha256": _sha256(args.base_checkpoint / "weights.pt"), "dataset": str(args.dataset), "dataset_sha256": _sha256(args.dataset), "receipt": str(args.receipt), "receipt_sha256": _sha256(args.receipt), "pool": {"path": str(args.pool), "sha256": _sha256(args.pool), "receipt": str(args.pool_receipt), "receipt_sha256": _sha256(args.pool_receipt), **pool_meta["pool"]}, "device": host, "training_contract": {"mode": "same_root_ranking_pool", "deployed_mapping": "[x1,x2,x3] + [a1,a2,a3] -> x4", "positive_candidate": "source packet which generated goal RGB", "negative_candidates": "same-root raw Uniform[-1,1] physical packets", "visual_encoder_and_projector": "frozen", "trainable_modules": ["predictor", "action_encoder", "pred_proj"], "p0_p1_roots_used": False, "cpu_fallback": False}, "episode_split": {"rule": "root seed modulo 10; 0..7 train, 8..9 validation", "train_roots": len(train_set), "validation_roots": len(validation_set), "overlap": False}, "optimization": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate}, "epochs": history, "selected_epoch": best_epoch, "selected_validation": history[best_epoch - 1]["validation"], "elapsed_seconds": time.time() - started, "claim_boundary": "This is a same-root P1 ranking diagnostic repair, not a frozen-model D2P result. A disjoint CUDA P1 audit is mandatory."}
    manifest['optimization'].update({'maximum_epochs': args.epochs, 'completed_epochs': len(history), 'validation_patience': patience, 'validation_kernel': 'singleton P1-equivalent rollout'})
    manifest['implementation'] = {'trainer_sha256': _sha256(Path(__file__)), 'pool_builder_sha256': _sha256(Path(__file__).with_name('build_cube_d2p_same_root_ranking_pool.py'))}
    (args.output / "repair_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": manifest["status"], "selected_validation": manifest["selected_validation"]}, sort_keys=True)); return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--receipt", type=Path, required=True); parser.add_argument("--pool", type=Path, required=True); parser.add_argument("--pool-receipt", type=Path, required=True); parser.add_argument("--base-checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--learning-rate", type=float, default=2e-5); parser.add_argument("--seed", type=int, default=202615101); parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument('--patience', type=int, default=0, help='Stop only on training-pool validation plateau; zero disables early stopping')
    train(parser.parse_args())


if __name__ == "__main__": main()
