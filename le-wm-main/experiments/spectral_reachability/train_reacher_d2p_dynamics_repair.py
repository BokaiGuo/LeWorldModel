"""Repair the Reacher action-conditioned dynamics contract on host CUDA.

The historical released LeWM training entry point optimized a prediction
against an *attached* future embedding.  That made the visual target move with
the predictor loss and left no direct pressure for an action token to rank its
own observed successor above a counterfactual token.  The current main
training entry point detaches that target; this separate immutable repair
branch additionally keeps the released visual encoder and projector fixed,
trains only the dynamics/action modules against detached observed-successor
latents, and adds an in-batch action-swap ranking loss.

It never reads P0/P1 roots.  The input is the public Reacher training HDF5;
validation is an episode-disjoint split of that training source only.  The
result is a new checkpoint which must subsequently clear fresh CUDA P0/P1
gates before it can support a planning claim.
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
from experiments.run_live_reacher_planning import _load_action_stats
from stable_pretraining import data as spt_data
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset


SCHEMA = "reacher_d2p_causal_dynamics_repair.v1"
HISTORY_SIZE = 3
FRAME_SKIP = 5
NUM_STEPS = HISTORY_SIZE + 1


def _progress(**fields: Any) -> None:
    """Emit bounded, machine-readable host-job progress without checkpoints."""

    print(json.dumps({"repair_progress": fields}, sort_keys=True), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_cuda(device: str) -> torch.device:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("causal Reacher dynamics repair requires host CUDA exactly; CPU fallback is forbidden")
    requested = torch.device("cuda")
    torch.empty(0, device=requested)
    return requested


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _EpisodeDisjointClips(Dataset[dict[str, torch.Tensor]]):
    """Subset a clip dataset by source episode, never by overlapping rows."""

    def __init__(self, base: HDF5Dataset, *, split: str, modulus: int = 10, train_slots: int = 8):
        if split not in {"train", "validation"}:
            raise ValueError(f"unknown split: {split}")
        if not 1 <= train_slots < modulus:
            raise ValueError("invalid episode split specification")
        self.base = base
        train = split == "train"
        self.indices = [
            index
            for index, (episode, _) in enumerate(base.clip_indices)
            if ((int(episode) % modulus) < train_slots) == train
        ]
        self.episode_ids = sorted({int(base.clip_indices[index][0]) for index in self.indices})
        if not self.indices:
            raise ValueError(f"{split} episode-disjoint split is empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.base[self.indices[index]]


class _EpisodeChunkBatchSampler(Sampler[list[int]]):
    """Draw diverse batches while reusing the public HDF5's RGB chunks.

    The archive stores 100 RGB frames per compressed chunk.  Uniform clip-row
    sampling makes a 32-example batch decompress roughly 32 random chunks.
    Instead, sample short consecutive runs from four episodes and interleave
    them: action-swap negatives are still normally cross-episode, while HDF5
    reuses only a few compressed chunks per batch.
    """

    def __init__(
        self,
        dataset: _EpisodeDisjointClips,
        *,
        batch_size: int,
        batches: int,
        seed: int,
        episode_groups: int = 4,
        advance_epoch: bool = True,
    ):
        if batch_size < 2 or batches < 1 or episode_groups < 1 or batch_size % episode_groups:
            raise ValueError("invalid Reacher chunk-aware batch sampler dimensions")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.batches = int(batches)
        self.seed = int(seed)
        self.episode_groups = int(episode_groups)
        # Training should explore fresh source clips each epoch, but checkpoint
        # selection requires comparable validation metrics.  Keeping the
        # validation sampler fixed prevents a lucky validation draw from
        # selecting a different checkpoint than a worse model on the same
        # held-out episodes.
        self.advance_epoch = bool(advance_epoch)
        self.per_episode = self.batch_size // self.episode_groups
        grouped: dict[int, list[int]] = {}
        for local_index, base_index in enumerate(dataset.indices):
            episode = int(dataset.base.clip_indices[base_index][0])
            grouped.setdefault(episode, []).append(local_index)
        self.by_episode = {episode: rows for episode, rows in grouped.items() if len(rows) >= self.per_episode}
        if len(self.by_episode) < self.episode_groups:
            raise ValueError("insufficient episode-disjoint clips for chunk-aware sampling")
        self._epoch = 0

    def __len__(self) -> int:
        return self.batches

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self._epoch)
        if self.advance_epoch:
            self._epoch += 1
        episodes = np.asarray(sorted(self.by_episode), dtype=np.int64)
        for _ in range(self.batches):
            selected = generator.choice(episodes, size=self.episode_groups, replace=False)
            runs: list[list[int]] = []
            for episode in selected:
                rows = self.by_episode[int(episode)]
                start = int(generator.integers(0, len(rows) - self.per_episode + 1))
                runs.append(rows[start : start + self.per_episode])
            yield np.asarray(runs, dtype=np.int64).T.reshape(-1).tolist()


def _image_transform() -> spt_data.transforms.Compose:
    stats = spt_data.dataset_stats.ImageNet
    return spt_data.transforms.Compose(
        spt_data.transforms.ToImage(**stats, source="pixels", target="pixels"),
        spt_data.transforms.Resize(224, source="pixels", target="pixels"),
    )


def _build_dataset(path: Path) -> HDF5Dataset:
    if not path.is_file():
        raise FileNotFoundError(f"Reacher training HDF5 not found: {path}")
    dataset = HDF5Dataset(
        path=path,
        frameskip=FRAME_SKIP,
        num_steps=NUM_STEPS,
        # The official source is already RGB 224x224.  ImageNet conversion is
        # performed after transfer in ``_preprocess_pixels`` so CPU workers do
        # not normalize every frame serially before the CUDA update.
        transform=None,
        keys_to_load=["pixels", "action"],
    )
    if dataset.get_dim("action") * FRAME_SKIP != 10:
        raise ValueError("Reacher repair expects two actions over a five-step action packet")
    return dataset


def _preprocess_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """Exactly reproduce the released 224px ImageNet transform on-device."""

    if pixels.ndim != 5 or pixels.shape[-3] != 3 or tuple(pixels.shape[-2:]) != (224, 224):
        raise ValueError(f"expected uint8 RGB [batch, steps, 3, 224, 224], got {tuple(pixels.shape)}")
    if pixels.dtype != torch.uint8:
        raise ValueError(f"expected source uint8 RGB before preprocessing, got {pixels.dtype}")
    stats = spt_data.dataset_stats.ImageNet
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=pixels.device).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(stats["std"], dtype=torch.float32, device=pixels.device).view(1, 1, 3, 1, 1)
    return pixels.to(dtype=torch.float32).div_(255.0).sub_(mean).div_(std)


def _source_action_stats(dataset: HDF5Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    actions = np.asarray(dataset.get_col_data("action")).reshape(-1, 2)
    finite = actions[np.isfinite(actions).all(axis=1)]
    if finite.size == 0:
        raise ValueError("training actions contain no finite rows")
    # Match ``utils.get_column_normalizer`` exactly: it fits one physical
    # two-dimensional statistic before HDF5Dataset turns five adjacent actions
    # into a ten-wide time-major action token.
    values = torch.from_numpy(np.ascontiguousarray(finite))
    return (
        values.mean(dim=0).to(torch.float32),
        values.std(dim=0).clamp_min(1.0e-6).to(torch.float32),
    )


def _normalize_actions(action: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if action.ndim != 3 or action.shape[-1] != 10:
        raise ValueError(f"expected [batch, {NUM_STEPS}, 10] action packets, got {tuple(action.shape)}")
    # Training-source HDF5 statistics are per physical two-dimensional action,
    # while the frozen runtime receipt stores the flattened ten-dimensional
    # packet expected by the checkpoint.  Supporting both forms is safe only
    # when each resolves to this exact time-major packet width; otherwise fail
    # rather than silently repeat a ten-wide runtime vector five times.
    if mean.numel() == 2 and std.numel() == 2:
        packet_mean = mean.repeat(FRAME_SKIP)
        packet_std = std.repeat(FRAME_SKIP)
    elif mean.numel() == 10 and std.numel() == 10:
        packet_mean = mean.reshape(-1)
        packet_std = std.reshape(-1)
    else:
        raise ValueError(f"action statistics must be two physical or ten packet values, got {mean.numel()} and {std.numel()}")
    packet_mean = packet_mean.view(1, 1, -1).to(action)
    packet_std = packet_std.view(1, 1, -1).to(action)
    # HDF5 clip windows can pad an unavailable action at an episode boundary
    # with NaN.  The released training entry point applies ``nan_to_num``
    # *after* its action z-score transform, so the missing normalized token is
    # the neutral value zero, not a raw zero action shifted by the dataset
    # mean.  Keep the repair branch semantically identical; otherwise a
    # boundary-only artifact can poison the action-conditioned ranking loss.
    normalized = (action.float() - packet_mean) / packet_std
    return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def _require_training_semantic_packet_stats(
    mean: torch.Tensor, std: torch.Tensor, *, source_mean: torch.Tensor, source_std: torch.Tensor,
) -> None:
    """Reject a position-specific live calibration in place of train-time stats."""

    expected_mean = source_mean.repeat(FRAME_SKIP)
    expected_std = source_std.repeat(FRAME_SKIP)
    if mean.numel() != expected_mean.numel() or std.numel() != expected_std.numel():
        raise ValueError("Reacher runtime receipt does not have one five-step action packet")
    if not torch.allclose(mean.reshape(-1).cpu(), expected_mean.cpu(), rtol=1.0e-5, atol=1.0e-6):
        raise ValueError("Reacher runtime action means differ across packet slots or from official training semantics")
    if not torch.allclose(std.reshape(-1).cpu(), expected_std.cpu(), rtol=1.0e-5, atol=1.0e-6):
        raise ValueError("Reacher runtime action stds differ across packet slots or from official training semantics")


def _derangement(batch: int, *, device: torch.device) -> torch.Tensor:
    if batch < 2:
        raise ValueError("counterfactual action ranking needs at least two samples")
    return torch.roll(torch.arange(batch, device=device), shifts=1)


def causal_dynamics_losses(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    normalized_actions: torch.Tensor,
    *,
    ranking_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return detached-target prediction and true-vs-swapped-action losses.

    The caller has frozen the encoder/projector.  Detaching here remains an
    explicit contract test: future RGB never receives gradient through the
    dynamics objective even if this helper is reused with an unfrozen model.
    """

    if pixels.ndim != 5 or normalized_actions.ndim != 3:
        raise ValueError("invalid Reacher repair batch ranks")
    if pixels.shape[1] != NUM_STEPS or normalized_actions.shape[1] != NUM_STEPS:
        raise ValueError("Reacher repair requires four RGB/action tokens per example")
    if ranking_temperature <= 0.0:
        raise ValueError("ranking temperature must be positive")
    with torch.no_grad():
        frozen_embeddings = model.encode({"pixels": pixels})["emb"].detach()
    context = frozen_embeddings[:, :HISTORY_SIZE]
    target = frozen_embeddings[:, 1:].detach()
    true_actions = model.action_encoder(normalized_actions[:, :HISTORY_SIZE])
    predicted = model.predict(context, true_actions)
    prediction_mse = F.mse_loss(predicted, target)
    positive_distance = (predicted[:, -1] - target[:, -1]).square().mean(dim=-1)
    permuted = _derangement(pixels.shape[0], device=pixels.device)
    negative_actions = model.action_encoder(normalized_actions[permuted, :HISTORY_SIZE])
    negative_prediction = model.predict(context, negative_actions)
    negative_distance = (negative_prediction[:, -1] - target[:, -1]).square().mean(dim=-1)
    ranking_loss = F.softplus((positive_distance - negative_distance) / ranking_temperature).mean()
    accuracy = (positive_distance < negative_distance).float().mean()
    total = prediction_mse + ranking_loss
    return total, {
        "prediction_mse": prediction_mse.detach(),
        "action_swap_ranking_loss": ranking_loss.detach(),
        "action_swap_ranking_accuracy": accuracy.detach(),
        "target_requires_grad": torch.tensor(float(target.requires_grad), device=pixels.device),
    }


def _freeze_visual_target(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    for module in (model.encoder, model.projector):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    trainable: list[torch.nn.Parameter] = []
    for module in (model.predictor, model.action_encoder, model.pred_proj):
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("no trainable dynamics parameters after freezing the visual target")
    return trainable


def _epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    ranking_temperature: float,
    max_batches: int,
    epoch: int,
    split: str,
) -> dict[str, float]:
    train = optimizer is not None
    if train:
        model.predictor.train(); model.action_encoder.train(); model.pred_proj.train()
    else:
        model.predictor.eval(); model.action_encoder.eval(); model.pred_proj.eval()
    _progress(phase="epoch_started", epoch=epoch, split=split, max_batches=max_batches)
    totals: dict[str, float] = {"loss": 0.0, "prediction_mse": 0.0, "action_swap_ranking_loss": 0.0, "action_swap_ranking_accuracy": 0.0}
    examples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        pixels = _preprocess_pixels(batch["pixels"].to(device, non_blocking=True))
        action = _normalize_actions(batch["action"].to(device, non_blocking=True), mean=action_mean, std=action_std)
        if batch_index == 0:
            _progress(phase="first_batch_loaded", epoch=epoch, split=split, pixel_shape=list(pixels.shape), pixel_device=str(pixels.device))
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss, pieces = causal_dynamics_losses(model, pixels, action, ranking_temperature=ranking_temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0,
            )
            optimizer.step()
        else:
            with torch.no_grad():
                loss, pieces = causal_dynamics_losses(model, pixels, action, ranking_temperature=ranking_temperature)
        count = int(pixels.shape[0])
        examples += count
        totals["loss"] += float(loss.detach()) * count
        for key in tuple(pieces):
            if key != "target_requires_grad":
                totals[key] += float(pieces[key]) * count
        if float(pieces["target_requires_grad"]) != 0.0:
            raise RuntimeError("future successor target received dynamics gradient")
        if batch_index == 0 or (batch_index + 1) % 64 == 0:
            _progress(phase="batch_completed", epoch=epoch, split=split, completed_batches=batch_index + 1, loss=float(loss.detach()))
    if examples == 0:
        raise RuntimeError("empty repair loader")
    return {key: value / examples for key, value in totals.items()}


def train(
    *,
    dataset: Path,
    base_checkpoint: Path,
    adapter_artifact: Path,
    output: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    ranking_temperature: float,
    train_batches_per_epoch: int,
    validation_batches: int,
    device_name: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable repaired checkpoint directory: {output}")
    if epochs < 1 or batch_size < 2 or learning_rate <= 0.0 or train_batches_per_epoch < 1 or validation_batches < 1:
        raise ValueError("invalid repair optimization parameters")
    device = _require_cuda(device_name)
    _seed_everything(seed)
    base = _build_dataset(dataset)
    train_set = _EpisodeDisjointClips(base, split="train")
    validation_set = _EpisodeDisjointClips(base, split="validation")
    train_episodes = set(train_set.episode_ids)
    validation_episodes = set(validation_set.episode_ids)
    if train_episodes & validation_episodes:
        raise RuntimeError("episode-disjoint repair split overlaps")
    _progress(phase="datasets_ready", train_clips=len(train_set), validation_clips=len(validation_set), train_episodes=len(train_episodes), validation_episodes=len(validation_episodes))
    train_loader = DataLoader(
        train_set,
        batch_sampler=_EpisodeChunkBatchSampler(train_set, batch_size=batch_size, batches=train_batches_per_epoch, seed=seed),
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_sampler=_EpisodeChunkBatchSampler(
            validation_set,
            batch_size=batch_size,
            batches=validation_batches,
            seed=seed + 1,
            advance_epoch=False,
        ),
        num_workers=0,
        pin_memory=True,
    )
    # The repaired predictor must receive the exact token normalization used by
    # P0/P1.  Recomputing it from a locally unpacked HDF5 can introduce a tiny
    # but avoidable training/runtime mismatch (for example after a source
    # revision or filtering change), so the frozen runtime receipt is the
    # canonical source.  We still record its relation to the raw source.
    runtime_mean, runtime_std = _load_action_stats(adapter_artifact, "swm/ReacherDMControl-v0")
    action_mean = torch.as_tensor(runtime_mean, dtype=torch.float32)
    action_std = torch.as_tensor(runtime_std, dtype=torch.float32)
    source_mean, source_std = _source_action_stats(base)
    _require_training_semantic_packet_stats(
        action_mean, action_std, source_mean=source_mean, source_std=source_std,
    )
    _progress(phase="action_contract_verified")
    model = load_compatible_pretrained(base_checkpoint, device=device).eval()
    trainable = _freeze_visual_target(model)
    _progress(phase="model_ready", trainable_parameter_count=sum(parameter.numel() for parameter in trainable))
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1.0e-4)
    history: list[dict[str, Any]] = []
    best_key = (-float("inf"), float("inf"))
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model=model, loader=train_loader, device=device, action_mean=action_mean, action_std=action_std, optimizer=optimizer, ranking_temperature=ranking_temperature, max_batches=train_batches_per_epoch, epoch=epoch, split="train")
        validation_metrics = _epoch(model=model, loader=validation_loader, device=device, action_mean=action_mean, action_std=action_std, optimizer=None, ranking_temperature=ranking_temperature, max_batches=validation_batches, epoch=epoch, split="validation")
        key = (validation_metrics["action_swap_ranking_accuracy"], -validation_metrics["prediction_mse"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        _progress(phase="epoch_completed", epoch=epoch, validation=validation_metrics)
    if best_state is None:
        raise RuntimeError("repair training did not produce a checkpoint")
    output.mkdir(parents=True)
    torch.save(best_state, output / "weights.pt")
    shutil.copy2(base_checkpoint / "config.json", output / "config.json")
    manifest = {
        "schema": SCHEMA,
        "status": "trained_train_source_only",
        "base_checkpoint": str(base_checkpoint.resolve()),
        "base_weights_sha256": _sha256(base_checkpoint / "weights.pt"),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": _sha256(dataset),
        "runtime_action_stats_artifact": str(adapter_artifact.resolve()),
        "runtime_action_stats_artifact_sha256": _sha256(adapter_artifact),
        "device": {"name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "training_contract": {
            "visual_encoder_and_projector": "frozen",
            "future_successor_target": "detached released-checkpoint RGB latent",
            "trainable_modules": ["predictor", "action_encoder", "pred_proj"],
            "loss": "all-position successor MSE plus last-successor true-action vs in-batch action-swap ranking",
            "ranking_temperature": ranking_temperature,
            "p0_p1_roots_used": False,
            "cpu_fallback": False,
        },
        "episode_split": {"rule": "episode_index modulo 10; slots 0..7 train, 8..9 validation", "train_episode_count": len(train_episodes), "validation_episode_count": len(validation_episodes), "overlap": False},
        "action_normalization": {
            "mean": action_mean.tolist(), "std": action_std.tolist(),
            "packet_layout": "five consecutive two-dimensional raw actions with source statistics tiled across packet slots",
            "source_dataset_mean": source_mean.tolist(), "source_dataset_std": source_std.tolist(),
            "runtime_minus_source_packet_mean_max_abs": float(torch.max(torch.abs(action_mean - source_mean.repeat(FRAME_SKIP)))),
            "runtime_minus_source_packet_std_max_abs": float(torch.max(torch.abs(action_std - source_std.repeat(FRAME_SKIP)))),
        },
        "optimization": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_batches_per_epoch": train_batches_per_epoch,
            "validation_batches": validation_batches,
            "validation_panel": "fixed seeded episode-disjoint chunk-aware panel across every epoch",
        },
        "epochs": history,
        "selected_epoch": best_epoch,
        "selected_validation": history[best_epoch - 1]["validation"],
        "elapsed_seconds": time.time() - started,
        "claim_boundary": "Training-source-only dynamics repair checkpoint. It is not a P1 result and must pass fresh CUDA fidelity, calibration, and root-disjoint outcome gates before supporting any planning claim.",
    }
    (output / "repair_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "selected_validation": manifest["selected_validation"], "status": manifest["status"]}, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm"))
    parser.add_argument("--adapter-artifact", type=Path, default=Path("results/live_cross_environment_prediction_20261015/reacher_cuda/cross_environment_prediction.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--ranking-temperature", type=float, default=0.1)
    parser.add_argument("--train-batches-per-epoch", type=int, default=512, help="fixed randomized source batches per epoch; avoids silently treating an unfinished full sweep as evidence")
    parser.add_argument("--validation-batches", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(dataset=args.dataset, base_checkpoint=args.base_checkpoint, adapter_artifact=args.adapter_artifact, output=args.output, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, ranking_temperature=args.ranking_temperature, train_batches_per_epoch=args.train_batches_per_epoch, validation_batches=args.validation_batches, device_name=args.device)


if __name__ == "__main__":
    main()
