"""CUDA-only counterfactual calibration repair for the Reacher D2P predictor.

This is deliberately separate from the ordinary observed-transition repair.
It teaches the frozen visual representation's dynamics modules on matched
same-root random-packet versus raw-zero successors, including their relative
goal-cost order.  The pool must be independent of every P0/P1 root.
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

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_reacher_planning import _load_action_stats
from experiments.spectral_reachability.build_reacher_d2p_training_action_stats import validate_training_semantic_receipt
from experiments.spectral_reachability.train_reacher_d2p_dynamics_repair import _freeze_visual_target, _preprocess_pixels, _require_cuda


SCHEMA = "reacher_d2p_counterfactual_calibration_repair.v1"
HISTORY_LEN = 3
ACTION_WIDTH = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _Pairs(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: Path, *, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("unknown pair-pool split")
        self.handle = h5py.File(path, "r")
        required = {"context_frames", "root_frame", "goal_frame", "candidate_successor", "zero_successor", "history_raw_actions", "candidate_raw_action", "episode_index"}
        if not required <= set(self.handle):
            raise ValueError("counterfactual pool is missing required datasets")
        episodes = np.asarray(self.handle["episode_index"], dtype=np.int64)
        train = split == "train"
        self.indices = np.flatnonzero((episodes % 10 < 8) == train)
        if self.indices.size == 0:
            raise ValueError(f"empty {split} counterfactual split")

    def __len__(self) -> int:
        return int(self.indices.size)

    @staticmethod
    def _image(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        frames = np.asarray(self.handle["context_frames"][row], dtype=np.uint8)
        return {
            "context": torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2),
            "root": self._image(np.asarray(self.handle["root_frame"][row], dtype=np.uint8)),
            "goal": self._image(np.asarray(self.handle["goal_frame"][row], dtype=np.uint8)),
            "candidate_successor": self._image(np.asarray(self.handle["candidate_successor"][row], dtype=np.uint8)),
            "zero_successor": self._image(np.asarray(self.handle["zero_successor"][row], dtype=np.uint8)),
            "history": torch.from_numpy(np.asarray(self.handle["history_raw_actions"][row], dtype=np.float32)),
            "candidate": torch.from_numpy(np.asarray(self.handle["candidate_raw_action"][row], dtype=np.float32)),
        }


def _normalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != ACTION_WIDTH or mean.numel() != ACTION_WIDTH or std.numel() != ACTION_WIDTH:
        raise ValueError("counterfactual action packet does not match frozen ten-wide Reacher contract")
    return (value.float() - mean.to(value).view(*([1] * (value.ndim - 1)), -1)) / std.to(value).view(*([1] * (value.ndim - 1)), -1)


def _predict_successor(model: torch.nn.Module, context: torch.Tensor, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    if tuple(context.shape[1:]) != (HISTORY_LEN, 3, 224, 224) or tuple(history.shape[1:]) != (HISTORY_LEN, ACTION_WIDTH) or tuple(action.shape[1:]) != (ACTION_WIDTH,):
        raise ValueError("counterfactual batch violates the root-anchored D2P contract")
    with torch.no_grad():
        encoded = model.encode({"pixels": context})["emb"].detach()
    action_context = torch.cat([history[:, 1:], action[:, None]], dim=1)
    return model.predict(encoded, model.action_encoder(action_context))[:, -1]


def counterfactual_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor], *, mean: torch.Tensor, std: torch.Tensor, temperature: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if temperature <= 0.0:
        raise ValueError("counterfactual ranking temperature must be positive")
    root_anchored = torch.cat([batch["context"][:, 1:], batch["root"][:, None]], dim=1)
    history = _normalize(batch["history"], mean, std)
    candidate_action = _normalize(batch["candidate"], mean, std)
    zero_action = _normalize(torch.zeros_like(batch["candidate"]), mean, std)
    candidate_prediction = _predict_successor(model, root_anchored, history, candidate_action)
    zero_prediction = _predict_successor(model, root_anchored, history, zero_action)
    with torch.no_grad():
        targets = model.encode({"pixels": torch.stack([batch["candidate_successor"], batch["zero_successor"], batch["goal"]], dim=1)})["emb"].detach()
    candidate_target, zero_target, goal_target = targets[:, 0], targets[:, 1], targets[:, 2]
    successor_mse = F.mse_loss(candidate_prediction, candidate_target) + F.mse_loss(zero_prediction, zero_target)
    predicted_delta = (candidate_prediction - goal_target).square().mean(dim=1) - (zero_prediction - goal_target).square().mean(dim=1)
    observed_delta = (candidate_target - goal_target).square().mean(dim=1) - (zero_target - goal_target).square().mean(dim=1)
    non_tied = observed_delta.abs() > 1.0e-6
    if not bool(non_tied.any()):
        raise RuntimeError("counterfactual batch has no non-tied goal-order labels")
    order = F.softplus(-torch.sign(observed_delta[non_tied]) * predicted_delta[non_tied] / temperature).mean()
    total = successor_mse + order
    return total, {
        "successor_mse": successor_mse.detach(),
        "counterfactual_order_loss": order.detach(),
        "counterfactual_order_accuracy": (torch.sign(predicted_delta[non_tied]) == torch.sign(observed_delta[non_tied])).float().mean().detach(),
        "target_requires_grad": torch.tensor(float(targets.requires_grad), device=total.device),
    }


def _run_epoch(*, model: torch.nn.Module, loader: DataLoader[dict[str, torch.Tensor]], device: torch.device, mean: torch.Tensor, std: torch.Tensor, optimizer: torch.optim.Optimizer | None, temperature: float) -> dict[str, float]:
    train = optimizer is not None
    for module in (model.predictor, model.action_encoder, model.pred_proj):
        module.train(train)
    totals = {"loss": 0.0, "successor_mse": 0.0, "counterfactual_order_loss": 0.0, "counterfactual_order_accuracy": 0.0}
    examples = 0
    for batch in loader:
        cuda_batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        processed: dict[str, torch.Tensor] = {}
        for key, value in cuda_batch.items():
            if key == "context":
                processed[key] = _preprocess_pixels(value)
            elif key in {"root", "goal", "candidate_successor", "zero_successor"}:
                processed[key] = _preprocess_pixels(value[:, None])[:, 0]
            else:
                processed[key] = value
        cuda_batch = processed
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = counterfactual_loss(model, cuda_batch, mean=mean, std=std, temperature=temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                loss, metrics = counterfactual_loss(model, cuda_batch, mean=mean, std=std, temperature=temperature)
        count = int(cuda_batch["context"].shape[0]); examples += count
        totals["loss"] += float(loss.detach()) * count
        for key in tuple(totals):
            if key != "loss":
                totals[key] += float(metrics[key]) * count
        if float(metrics["target_requires_grad"]) != 0.0:
            raise RuntimeError("counterfactual RGB targets received predictor gradient")
    if examples == 0:
        raise RuntimeError("empty counterfactual repair loader")
    return {key: value / examples for key, value in totals.items()}


def train(*, pool: Path, pool_manifest: Path, adapter_artifact: Path, base_checkpoint: Path, output: Path, seed: int, epochs: int, batch_size: int, learning_rate: float, temperature: float, device_name: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable counterfactual checkpoint: {output}")
    if min(epochs, batch_size) < 1 or learning_rate <= 0.0:
        raise ValueError("invalid counterfactual repair optimization arguments")
    source = json.loads(pool_manifest.read_text(encoding="utf-8"))
    if source.get("schema") != "reacher_d2p_counterfactual_pair_pool.v1" or source.get("status") != "completed" or source.get("p0_p1_calibration_or_outcome_roots_used") is not False:
        raise ValueError("counterfactual pool provenance is invalid or contaminates P0/P1")
    if Path(source.get("pairs_hdf5", "")).resolve() != pool.resolve() or source.get("pairs_hdf5_sha256") != _sha256(pool):
        raise ValueError("counterfactual pool differs from its immutable manifest")
    device = _require_cuda(device_name)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    validate_training_semantic_receipt(adapter_artifact)
    action_mean, action_std = _load_action_stats(adapter_artifact, "swm/ReacherDMControl-v0")
    mean = torch.as_tensor(action_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(action_std, dtype=torch.float32, device=device)
    train_set, validation_set = _Pairs(pool, split="train"), _Pairs(pool, split="validation")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0, pin_memory=True, drop_last=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = load_compatible_pretrained(base_checkpoint, device=device).eval()
    trainable = _freeze_visual_target(model)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1.0e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch, best_accuracy = 0, -float("inf")
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model=model, loader=train_loader, device=device, mean=mean, std=std, optimizer=optimizer, temperature=temperature)
        validation_metrics = _run_epoch(model=model, loader=validation_loader, device=device, mean=mean, std=std, optimizer=None, temperature=temperature)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["counterfactual_order_accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["counterfactual_order_accuracy"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("counterfactual repair did not select a checkpoint")
    output.mkdir(parents=True)
    torch.save(best_state, output / "weights.pt")
    shutil.copy2(base_checkpoint / "config.json", output / "config.json")
    result = {
        "schema": SCHEMA, "status": "trained_counterfactual_source_only",
        "base_checkpoint": str(base_checkpoint.resolve()), "base_weights_sha256": _sha256(base_checkpoint / "weights.pt"),
        "counterfactual_pool": str(pool.resolve()), "counterfactual_pool_sha256": _sha256(pool),
        "counterfactual_pool_manifest": str(pool_manifest.resolve()), "counterfactual_pool_manifest_sha256": _sha256(pool_manifest),
        "adapter_artifact": str(adapter_artifact.resolve()), "adapter_sha256": _sha256(adapter_artifact),
        "device": {"name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "training_contract": {"visual_encoder_and_projector": "frozen", "future_successor_targets": "detached candidate and same-root raw-zero RGB latents", "trainable_modules": ["predictor", "action_encoder", "pred_proj"], "counterfactual_goal_order": "same-root candidate versus zero packet", "p0_p1_roots_used": False, "cpu_fallback": False},
        "episode_split": "episode_index modulo 10; slots 0..7 train, 8..9 validation", "optimization": {"seed": seed, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "ranking_temperature": temperature, "validation_panel": "fixed full episode-disjoint validation split"},
        "epochs": history, "selected_epoch": best_epoch, "selected_validation": history[best_epoch - 1]["validation"], "elapsed_seconds": time.time() - started,
        "claim_boundary": "Counterfactual-source-only repair checkpoint. Fresh CUDA P0/P1 admissions remain required.",
    }
    (output / "repair_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True); parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True); parser.add_argument("--base-checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20267001); parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--learning-rate", type=float, default=1.0e-5); parser.add_argument("--ranking-temperature", type=float, default=0.05); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train(pool=args.pool, pool_manifest=args.pool_manifest, adapter_artifact=args.adapter_artifact, base_checkpoint=args.base_checkpoint, output=args.output, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, temperature=args.ranking_temperature, device_name=args.device)
    print(json.dumps({"output": str(args.output), "selected_validation": result["selected_validation"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
