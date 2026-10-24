"""Joint CUDA-only Reacher dynamics repair with matched counterfactuals.

The ordinary repair's cross-episode action swaps establish transition
conditioning but are not a sufficient test of the P1 decision: P1 chooses a
candidate packet against a zero packet at the *same* RGB root.  Conversely,
the small matched-pair repair by itself can damage general transition
prediction.  This trainer keeps the visual representation frozen and updates
only dynamics modules using both independent sources in one objective.

Neither source may contain P0/P1 calibration or outcome roots.  This produces
a new diagnostic checkpoint, not a planning result; fresh CUDA gates remain
mandatory afterwards.
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
from torch.utils.data import DataLoader

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_reacher_planning import _load_action_stats
from experiments.spectral_reachability.build_reacher_d2p_training_action_stats import validate_training_semantic_receipt
from experiments.spectral_reachability.train_reacher_d2p_counterfactual_repair import _Pairs, counterfactual_loss
from experiments.spectral_reachability.train_reacher_d2p_dynamics_repair import (
    _EpisodeChunkBatchSampler,
    _EpisodeDisjointClips,
    _build_dataset,
    _freeze_visual_target,
    _normalize_actions,
    _preprocess_pixels,
    _require_cuda,
    _require_training_semantic_packet_stats,
    _source_action_stats,
    causal_dynamics_losses,
)


SCHEMA = "reacher_d2p_joint_transition_counterfactual_repair.v1"
PAIR_POOL_SCHEMAS = {
    "reacher_d2p_counterfactual_pair_pool.v1",
    "reacher_d2p_p1_proposal_pair_pool.v1",
    "reacher_d2p_p1_proposal_pair_mixture.v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_pair(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    raw = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    result: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if key == "context":
            result[key] = _preprocess_pixels(value)
        elif key in {"root", "goal", "candidate_successor", "zero_successor"}:
            result[key] = _preprocess_pixels(value[:, None])[:, 0]
        else:
            result[key] = value
    return result


def _next(iterator: Any, loader: DataLoader[Any]) -> tuple[dict[str, torch.Tensor], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _transition_metrics(
    model: torch.nn.Module, batch: dict[str, torch.Tensor], *, device: torch.device,
    mean: torch.Tensor, std: torch.Tensor, temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pixels = _preprocess_pixels(batch["pixels"].to(device, non_blocking=True))
    actions = _normalize_actions(batch["action"].to(device, non_blocking=True), mean=mean, std=std)
    return causal_dynamics_losses(model, pixels, actions, ranking_temperature=temperature)


def _metric_values(transition: dict[str, torch.Tensor], counterfactual: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "transition_loss": float(transition["loss"].detach()),
        "transition_prediction_mse": float(transition["prediction_mse"].detach()),
        "transition_action_swap_accuracy": float(transition["action_swap_ranking_accuracy"].detach()),
        "counterfactual_loss": float(counterfactual["loss"].detach()),
        "counterfactual_successor_mse": float(counterfactual["successor_mse"].detach()),
        "counterfactual_order_accuracy": float(counterfactual["counterfactual_order_accuracy"].detach()),
    }


def _run_epoch(
    *, model: torch.nn.Module, transition_loader: DataLoader[Any], pair_loader: DataLoader[Any],
    retention_pair_loader: DataLoader[Any] | None,
    device: torch.device, mean: torch.Tensor, std: torch.Tensor, transition_temperature: float,
    counterfactual_temperature: float, counterfactual_weight: float, optimizer: torch.optim.Optimizer | None,
    steps: int,
) -> dict[str, float]:
    if steps < 1 or counterfactual_weight <= 0.0:
        raise ValueError("joint repair needs positive steps and counterfactual weight")
    train = optimizer is not None
    for module in (model.predictor, model.action_encoder, model.pred_proj):
        module.train(train)
    transition_iter, pair_iter = iter(transition_loader), iter(pair_loader)
    retention_iter = iter(retention_pair_loader) if retention_pair_loader is not None else None
    totals: dict[str, float] = {}
    for _ in range(steps):
        transition_batch, transition_iter = _next(transition_iter, transition_loader)
        pair_batch, pair_iter = _next(pair_iter, pair_loader)
        if train:
            optimizer.zero_grad(set_to_none=True)
        transition_loss, transition = _transition_metrics(
            model, transition_batch, device=device, mean=mean, std=std, temperature=transition_temperature,
        )
        pair_loss, counterfactual = counterfactual_loss(
            model, _prepare_pair(pair_batch, device), mean=mean, std=std, temperature=counterfactual_temperature,
        )
        retention_loss = torch.zeros_like(pair_loss)
        retention = None
        if retention_pair_loader is not None and retention_iter is not None:
            retention_batch, retention_iter = _next(retention_iter, retention_pair_loader)
            retention_loss, retention = counterfactual_loss(model, _prepare_pair(retention_batch, device), mean=mean, std=std, temperature=counterfactual_temperature)
        loss = transition_loss + float(counterfactual_weight) * (pair_loss + retention_loss)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
            optimizer.step()
        values = _metric_values({**transition, "loss": transition_loss}, {**counterfactual, "loss": pair_loss})
        if retention is not None:
            retained = _metric_values({**transition, "loss": transition_loss}, {**retention, "loss": retention_loss})
            values["retention_counterfactual_loss"] = retained["counterfactual_loss"]
            values["retention_counterfactual_order_accuracy"] = retained["counterfactual_order_accuracy"]
        values["joint_loss"] = float(loss.detach())
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        if float(transition["target_requires_grad"]) != 0.0 or float(counterfactual["target_requires_grad"]) != 0.0 or (retention is not None and float(retention["target_requires_grad"]) != 0.0):
            raise RuntimeError("a frozen RGB successor target received dynamics gradient")
    return {key: value / steps for key, value in totals.items()}


def train(
    *, dataset: Path, pair_pool: Path, pair_manifest: Path, retention_pair_pool: Path | None, retention_pair_manifest: Path | None, adapter_artifact: Path,
    base_checkpoint: Path, output: Path, seed: int, epochs: int, transition_batch_size: int,
    pair_batch_size: int, steps_per_epoch: int, learning_rate: float, counterfactual_weight: float,
    transition_temperature: float, counterfactual_temperature: float, device_name: str,
) -> dict[str, Any]:
    if output.exists() or min(epochs, transition_batch_size, pair_batch_size, steps_per_epoch) < 1:
        raise ValueError("joint repair output exists or dimensions are invalid")
    if learning_rate <= 0.0 or counterfactual_weight <= 0.0:
        raise ValueError("joint repair optimization arguments must be positive")
    provenance = json.loads(pair_manifest.read_text(encoding="utf-8"))
    if provenance.get("schema") not in PAIR_POOL_SCHEMAS or provenance.get("status") != "completed" or provenance.get("p0_p1_calibration_or_outcome_roots_used") is not False:
        raise ValueError("joint repair pair source is invalid or contaminates P0/P1 roots")
    if Path(provenance.get("pairs_hdf5", "")).resolve() != pair_pool.resolve() or provenance.get("pairs_hdf5_sha256") != _sha256(pair_pool):
        raise ValueError("joint repair pair source differs from immutable manifest")
    if (retention_pair_pool is None) != (retention_pair_manifest is None):
        raise ValueError("retention pool and manifest must be supplied together")
    retention_provenance = None
    if retention_pair_pool is not None and retention_pair_manifest is not None:
        retention_provenance = json.loads(retention_pair_manifest.read_text(encoding="utf-8"))
        if retention_provenance.get("schema") not in PAIR_POOL_SCHEMAS or retention_provenance.get("status") != "completed" or retention_provenance.get("p0_p1_calibration_or_outcome_roots_used") is not False or Path(retention_provenance.get("pairs_hdf5", "")).resolve() != retention_pair_pool.resolve() or retention_provenance.get("pairs_hdf5_sha256") != _sha256(retention_pair_pool):
            raise ValueError("retention pair source is invalid or contaminates P0/P1 roots")
    device = _require_cuda(device_name)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    validate_training_semantic_receipt(adapter_artifact)
    base = _build_dataset(dataset)
    source_mean, source_std = _source_action_stats(base)
    runtime_mean, runtime_std = _load_action_stats(adapter_artifact, "swm/ReacherDMControl-v0")
    mean, std = torch.as_tensor(runtime_mean, dtype=torch.float32, device=device), torch.as_tensor(runtime_std, dtype=torch.float32, device=device)
    _require_training_semantic_packet_stats(mean.cpu(), std.cpu(), source_mean=source_mean, source_std=source_std)
    transition_train, transition_validation = _EpisodeDisjointClips(base, split="train"), _EpisodeDisjointClips(base, split="validation")
    pair_train, pair_validation = _Pairs(pair_pool, split="train"), _Pairs(pair_pool, split="validation")
    retention_train = _Pairs(retention_pair_pool, split="train") if retention_pair_pool is not None else None
    retention_validation = _Pairs(retention_pair_pool, split="validation") if retention_pair_pool is not None else None
    if set(transition_train.episode_ids) & set(transition_validation.episode_ids):
        raise RuntimeError("official transition train/validation episode sets overlap")
    transition_train_loader = DataLoader(transition_train, batch_sampler=_EpisodeChunkBatchSampler(transition_train, batch_size=transition_batch_size, batches=steps_per_epoch, seed=seed), num_workers=0, pin_memory=True)
    transition_validation_loader = DataLoader(transition_validation, batch_sampler=_EpisodeChunkBatchSampler(transition_validation, batch_size=transition_batch_size, batches=steps_per_epoch, seed=seed + 1, advance_epoch=False), num_workers=0, pin_memory=True)
    pair_train_loader = DataLoader(pair_train, batch_size=pair_batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 2), num_workers=0, pin_memory=True, drop_last=True)
    pair_validation_loader = DataLoader(pair_validation, batch_size=pair_batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    retention_train_loader = DataLoader(retention_train, batch_size=pair_batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 3), num_workers=0, pin_memory=True, drop_last=True) if retention_train is not None else None
    retention_validation_loader = DataLoader(retention_validation, batch_size=pair_batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True) if retention_validation is not None else None
    model = load_compatible_pretrained(base_checkpoint, device=device).eval()
    trainable = _freeze_visual_target(model)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1.0e-4)
    history: list[dict[str, Any]] = []
    best_key, best_epoch, best_state = float("inf"), 0, None
    started = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model=model, transition_loader=transition_train_loader, pair_loader=pair_train_loader, retention_pair_loader=retention_train_loader, device=device, mean=mean, std=std, transition_temperature=transition_temperature, counterfactual_temperature=counterfactual_temperature, counterfactual_weight=counterfactual_weight, optimizer=optimizer, steps=steps_per_epoch)
        with torch.no_grad():
            validation_metrics = _run_epoch(model=model, transition_loader=transition_validation_loader, pair_loader=pair_validation_loader, retention_pair_loader=retention_validation_loader, device=device, mean=mean, std=std, transition_temperature=transition_temperature, counterfactual_temperature=counterfactual_temperature, counterfactual_weight=counterfactual_weight, optimizer=None, steps=steps_per_epoch)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["joint_loss"] < best_key:
            best_key, best_epoch = validation_metrics["joint_loss"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("joint repair did not select a checkpoint")
    output.mkdir(parents=True)
    torch.save(best_state, output / "weights.pt")
    shutil.copy2(base_checkpoint / "config.json", output / "config.json")
    result = {
        "schema": SCHEMA, "status": "trained_joint_source_only",
        "base_checkpoint": str(base_checkpoint.resolve()), "base_weights_sha256": _sha256(base_checkpoint / "weights.pt"),
        "transition_dataset": str(dataset.resolve()), "transition_dataset_sha256": _sha256(dataset),
        "counterfactual_pool": str(pair_pool.resolve()), "counterfactual_pool_sha256": _sha256(pair_pool),
        "counterfactual_manifest": str(pair_manifest.resolve()), "counterfactual_manifest_sha256": _sha256(pair_manifest),
        "retention_counterfactual_pool": None if retention_pair_pool is None else str(retention_pair_pool.resolve()), "retention_counterfactual_manifest": None if retention_pair_manifest is None else str(retention_pair_manifest.resolve()),
        "adapter_artifact": str(adapter_artifact.resolve()), "adapter_sha256": _sha256(adapter_artifact),
        "device": {"name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "training_contract": {"visual_encoder_and_projector": "frozen", "trainable_modules": ["predictor", "action_encoder", "pred_proj"], "loss": "official detached successor transition loss plus deployment-P1 and optional retained random same-root candidate-versus-zero losses", "p0_p1_roots_used": False, "cpu_fallback": False},
        "optimization": {"seed": seed, "epochs": epochs, "transition_batch_size": transition_batch_size, "pair_batch_size": pair_batch_size, "steps_per_epoch": steps_per_epoch, "learning_rate": learning_rate, "counterfactual_weight": counterfactual_weight, "checkpoint_selection": "minimum fixed root-disjoint-source joint validation loss"},
        "epochs": history, "selected_epoch": best_epoch, "selected_validation": history[best_epoch - 1]["validation"], "elapsed_seconds": time.time() - started,
        "claim_boundary": "Joint source-only diagnostic checkpoint. Fresh CUDA P0/P1 fidelity, calibration, and root-disjoint outcome gates are still required.",
    }
    (output / "repair_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected_epoch": best_epoch, "selected_validation": result["selected_validation"]}, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--pair-pool", type=Path, required=True); parser.add_argument("--pair-manifest", type=Path, required=True); parser.add_argument("--retention-pair-pool", type=Path); parser.add_argument("--retention-pair-manifest", type=Path)
    parser.add_argument("--adapter-artifact", type=Path, required=True); parser.add_argument("--base-checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20268001); parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--transition-batch-size", type=int, default=32); parser.add_argument("--pair-batch-size", type=int, default=8); parser.add_argument("--steps-per-epoch", type=int, default=128); parser.add_argument("--learning-rate", type=float, default=1.0e-5); parser.add_argument("--counterfactual-weight", type=float, default=1.0); parser.add_argument("--transition-temperature", type=float, default=0.1); parser.add_argument("--counterfactual-temperature", type=float, default=0.05); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(dataset=args.dataset, pair_pool=args.pair_pool, pair_manifest=args.pair_manifest, retention_pair_pool=args.retention_pair_pool, retention_pair_manifest=args.retention_pair_manifest, adapter_artifact=args.adapter_artifact, base_checkpoint=args.base_checkpoint, output=args.output, seed=args.seed, epochs=args.epochs, transition_batch_size=args.transition_batch_size, pair_batch_size=args.pair_batch_size, steps_per_epoch=args.steps_per_epoch, learning_rate=args.learning_rate, counterfactual_weight=args.counterfactual_weight, transition_temperature=args.transition_temperature, counterfactual_temperature=args.counterfactual_temperature, device_name=args.device)


if __name__ == "__main__":
    main()
