"""Build the decisive renderer-backed PointMaze temporal action-OOD v3 panel.

This panel exists because only episodes 1800--1899 have stored RGB.  It uses
fresh episodes below 1800 for a 64/48/32 root-disjoint train/calibration/outcome
design, renders their *causal* observations through one fixed RGB chart fitted
from stored-RGB calibration episodes, and encodes them with the frozen JEPA
backbone.  It is deliberately labelled a renderer-backed frozen-observation
control: it is not a live-RGB deployment result.

No v1/v2 outcome root, candidate hash, or timing template may enter this
panel.  Candidate outcomes are sealed in a separate file until post-fit
evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark.audit_unseen_action_bank_contract import sequence_hash
from experiments.decision_benchmark.build_pointmaze_temporal_ood_bank import (
    FRAME_STRIDE,
    HISTORY_WINDOWS,
    causal_history,
    temporal_template_fingerprint,
)
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import _load_records, _simulate_sequence
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_pointmaze_planning import (
    DatasetMatchedRenderer,
    _load_jepa_model,
    _make_context,
    _make_gdd_source_contract_context,
    _pool_visual,
    fit_visual_calibration,
    green_center,
    load_episode,
)


SCHEMA = "pointmaze_temporal_ood_v3_renderer_control.v1"
V3_ACTION_OOD_FAMILY = "world_fixed_random_five_pulse_temporal_ood_v3"
V3_TRAIN_FAMILY = "world_fixed_random_train_five_pulse_temporal_ood_v3"
DEFAULT_ROOT_SPLIT_SEED = 202610221
DEFAULT_ACTION_BANK_SEED = 202610222
DEFAULT_TRAIN_ACTION_SEED = 202610223
SPLIT_COUNTS = {"train": 64, "calibration": 48, "outcome": 32}
SOURCE_FRAME = 20
GOAL = np.asarray([1.2, 1.2], dtype=np.float64)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def v3_action_sequences(*, seed: int, count: int = 16, steps: int = 20, train: bool = False) -> np.ndarray:
    """Draw a seed-fixed five-pulse bank without reading any outcome label."""

    if count < 8 or steps != 20:
        raise ValueError("v3 fixes a 16-or-more candidate, 20-step action bank")
    rng = np.random.default_rng(int(seed))
    values: list[np.ndarray] = []
    for _ in range(int(count)):
        # Five four-step pulses prevent accidental reuse of the structured
        # v1/v2 four-pulse schedules.  A distinct training seed/family keeps
        # fitting candidates disjoint from the held action-OOD bank.
        angles = rng.uniform(-np.pi, np.pi, size=5)
        low, high = ((0.12, 0.60) if train else (0.28, 0.78))
        magnitudes = rng.uniform(low, high, size=5)
        pulses = np.stack((np.cos(angles), np.sin(angles)), axis=1) * magnitudes[:, None]
        value = np.repeat(pulses, 4, axis=0)
        if float(np.abs(value).max()) > 1.0:
            raise RuntimeError("v3 action exceeds PointMaze control support")
        values.append(value)
    bank = np.stack(values).astype(np.float64)
    if len({sequence_hash(item) for item in bank}) != len(bank):
        raise RuntimeError("v3 action bank has duplicate hashes")
    if len({temporal_template_fingerprint(item) for item in bank}) != len(bank):
        raise RuntimeError("v3 action bank has duplicate legacy timing fingerprints")
    return bank


def _context_identities(*contexts: Path) -> tuple[set[int], set[str], set[str]]:
    roots, hashes, templates = set(), set(), set()
    for path in contexts:
        metadata, records = _load_records(path)
        if not metadata.get("candidate_family"):
            raise ValueError(f"context lacks candidate family: {path}")
        roots.update(int(row["start_episode_id"]) for row in records if str(row["split"]) == "outcome")
        hashes.update(str(row["candidate_action_hash"]) for row in records)
        templates.update(str(row["candidate_template_fingerprint"]) for row in records)
    if not roots or not hashes or not templates:
        raise ValueError("v1/v2 identity ledgers must be non-empty")
    return roots, hashes, templates


def _split_roots(*, seed: int, forbidden: set[int]) -> dict[int, str]:
    total = sum(SPLIT_COUNTS.values())
    eligible = np.asarray([value for value in range(1800) if value not in forbidden], dtype=np.int64)
    if len(eligible) < total:
        raise ValueError("insufficient fresh pre-1800 PointMaze roots")
    ordered = eligible[np.random.default_rng(int(seed)).permutation(len(eligible))[:total]]
    split: dict[int, str] = {}
    offset = 0
    for name in ("train", "calibration", "outcome"):
        width = SPLIT_COUNTS[name]
        split.update({int(root): name for root in ordered[offset : offset + width]})
        offset += width
    return split


def _renderer_calibration_roots(dataset_root: Path, forbidden: set[int]) -> list[int]:
    available = [episode for episode in range(1800, 1900) if (dataset_root / "obses" / f"episode_{episode:03d}.pth").exists()]
    usable = [episode for episode in available if episode not in forbidden]
    if len(usable) < 40:
        raise ValueError("need 40 stored-RGB roots not used as v1/v2 outcomes for renderer calibration")
    return usable[:40]


def _proprio(renderer: DatasetMatchedRenderer, calibration: Any, states: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray]:
    image = renderer.render(states[frame])
    previous = renderer.render(states[max(0, frame - FRAME_STRIDE)])
    center, old_center = green_center(image), green_center(previous)
    xy = calibration.position(center)
    velocity = calibration.velocity((center[0] - old_center[0], center[1] - old_center[1]))
    return image, np.concatenate((xy, velocity)).astype(np.float64)


def _encode_root_history(
    model: Any, transform: Any, pca_mean: np.ndarray, pca_basis: np.ndarray,
    renderer: DatasetMatchedRenderer, calibration: Any, states: np.ndarray,
) -> np.ndarray:
    frames = [SOURCE_FRAME - (HISTORY_WINDOWS - 1 - index) * FRAME_STRIDE for index in range(HISTORY_WINDOWS)]
    images, proprios = zip(*[_proprio(renderer, calibration, states, frame) for frame in frames])
    # Each frame is independently encoded with RGB-derived velocity from its
    # preceding five-step interval.  No frame or action after t=20 enters it.
    with torch.inference_mode():
        values = []
        for image, proprio in zip(images, proprios):
            context = _make_context([image], [proprio])
            encoded = model.encode(_make_gdd_source_contract_context(context, transform))
            pooled = _pool_visual(encoded["visual"][:, -1]).detach().cpu().numpy()[0]
            values.append((np.asarray(pooled, dtype=np.float64) - pca_mean) @ pca_basis.T)
    return np.stack(values).reshape(-1)


def build(
    dataset_root: Path, v1_context: Path, v2_context: Path, output: Path, *,
    encoder_checkpoint: Path, pca_checkpoint: Path, dinov2_source: Path, root_split_seed: int = DEFAULT_ROOT_SPLIT_SEED,
    action_bank_seed: int = DEFAULT_ACTION_BANK_SEED, train_action_seed: int = DEFAULT_TRAIN_ACTION_SEED,
    device: str = "cuda",
) -> dict[str, Any]:
    if device != "cuda":
        raise ValueError("formal v3 panel requires CUDA")
    cuda = require_cuda_allocation(device)
    v12_outcomes, v12_hashes, v12_templates = _context_identities(v1_context, v2_context)
    split = _split_roots(seed=root_split_seed, forbidden=v12_outcomes)
    if set(split) & v12_outcomes:
        raise RuntimeError("v3 root split overlaps a v1/v2 outcome root")
    calibration_roots = _renderer_calibration_roots(dataset_root, v12_outcomes)
    calibration = fit_visual_calibration(dataset_root, calibration_roots, max_frames=1600)
    renderer_images, _ = load_episode(dataset_root, calibration_roots[0])
    renderer = DatasetMatchedRenderer.from_episode(renderer_images, calibration)
    root = Path(__file__).resolve().parents[3]
    # The 202MB world-model checkpoint initializes the JEPA encoder.  The
    # small history-adapter checkpoint supplies only the pre-existing PCA
    # projection.  Treating the latter as encoder weights is an invalid (and
    # newer PyTorch-incompatible) serialization contract.
    model, transform, _stats, _mean, _std = _load_jepa_model(root, dataset_root, encoder_checkpoint, dinov2_source, torch.device(device))
    payload = torch.load(pca_checkpoint, map_location="cpu", weights_only=False)
    pca_mean = np.asarray(payload["pca_mean"], dtype=np.float64)
    pca_basis = np.asarray(payload["pca_basis"], dtype=np.float64)
    states = np.asarray(torch.load(dataset_root / "states.pth", map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    actions = np.asarray(torch.load(dataset_root / "actions.pth", map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    outcome_bank = v3_action_sequences(seed=action_bank_seed, train=False)
    train_bank = v3_action_sequences(seed=train_action_seed, train=True)
    outcome_hashes = [sequence_hash(value) for value in outcome_bank]
    outcome_templates = [temporal_template_fingerprint(value) for value in outcome_bank]
    train_hashes = [sequence_hash(value) for value in train_bank]
    train_templates = [temporal_template_fingerprint(value) for value in train_bank]
    if set(outcome_hashes) & v12_hashes or set(outcome_templates) & v12_templates:
        raise RuntimeError("v3 held bank overlaps a v1/v2 action hash/template")
    if set(outcome_hashes) & set(train_hashes) or set(outcome_templates) & set(train_templates):
        raise RuntimeError("v3 fit and held action banks overlap")

    histories: dict[int, np.ndarray] = {}
    for episode in sorted(split):
        histories[episode] = _encode_root_history(model, transform, pca_mean, pca_basis, renderer, calibration, states[episode])
    source_records: list[dict[str, Any]] = []
    source_arrays: dict[str, np.ndarray] = {}
    contexts: list[dict[str, Any]] = []
    outcomes: dict[str, Any] = {}
    frames, _ = causal_history(SOURCE_FRAME, actions[0])
    for episode in sorted(split):
        state = np.asarray(states[episode, SOURCE_FRAME], dtype=np.float64)
        history_actions = causal_history(SOURCE_FRAME, actions[episode])[1]
        history = histories[episode]
        current = history[-len(pca_basis) :]
        common = {
            "instance_id": f"pointmaze_v3_episode_{episode}", "split": split[episode], "environment": "PointMaze",
            "start_episode_id": str(episode), "source_frame": SOURCE_FRAME,
            "history_frame_indices": frames, "history_action_interval": [frames[0], SOURCE_FRAME],
            "history_actions_raw": history_actions.tolist(), "source_latent_model": current.tolist(),
            "source_history_model": history.tolist(), "source_state_raw": state.tolist(), "goal_raw": GOAL.tolist(),
        }
        if split[episode] == "train":
            for index, candidate in enumerate(train_bank):
                key = f"terminal/v3/{episode}/{index}"
                source_arrays[key] = _simulate_sequence(state, candidate)
                source_records.append({**common, "candidate_id": index, "candidate_generator": V3_TRAIN_FAMILY,
                    "candidate_bank_hash": _json_hash(train_hashes), "candidate_action_hash": train_hashes[index],
                    "candidate_template_fingerprint": train_templates[index], "action_sequence": candidate.tolist(),
                    "realized_terminal_state_ref": f"arrays.npz::{key}"})
        for index, candidate in enumerate(outcome_bank):
            key = f"{common['instance_id']}/{index}"
            contexts.append({**common, "candidate_id": index, "candidate_generator": V3_ACTION_OOD_FAMILY,
                "candidate_bank_hash": _json_hash(outcome_hashes), "candidate_action_hash": outcome_hashes[index],
                "candidate_template_fingerprint": outcome_templates[index], "action_sequence": candidate.tolist(),
                "total_action_steps": 20, "outcome_label_status": "sealed_until_postfit_join"})
            terminal = _simulate_sequence(state, candidate)
            outcomes[key] = {"real_task_cost": float(np.linalg.norm(terminal[:2] - GOAL)), "realized_terminal_state": terminal.tolist()}
    output.mkdir(parents=True, exist_ok=True)
    source_path, arrays_path = output / "temporal_ood_v3_source_dataset.json", output / "temporal_ood_v3_source_arrays.npz"
    context_path, outcome_path = output / "temporal_ood_v3_context.json", output / "temporal_ood_v3_outcomes.json"
    source_path.write_text(json.dumps({"metadata": {"schema": "pointmaze_temporal_ood_v3_source.v1", "history_windows": 4,
        "renderer_backed_frozen_observation_control": True, "source_root_split": "v3_train_only"}, "records": source_records}, indent=2, sort_keys=True) + "\n")
    np.savez(arrays_path, **source_arrays)
    context_path.write_text(json.dumps({"metadata": {"schema": SCHEMA, "candidate_family": V3_ACTION_OOD_FAMILY,
        "root_split_seed": root_split_seed, "action_bank_seed": action_bank_seed, "train_action_seed": train_action_seed,
        "history_windows": 4, "history_frame_stride": FRAME_STRIDE, "history_is_causal": True,
        "history_actions_are_strictly_pre_candidate": True, "outcomes_sealed_from_feature_construction": True,
        "renderer_backed_frozen_observation_control": True, "renderer_calibration_roots": calibration_roots,
        "renderer_calibration_excludes_v1_v2_outcomes": True, "v1_v2_outcome_roots_excluded": sorted(v12_outcomes),
        "v1_v2_action_hash_overlap": [], "v1_v2_template_overlap": [], "encoder_checkpoint_sha256": _sha256(encoder_checkpoint),
        "pca_checkpoint_sha256": _sha256(pca_checkpoint),
        "source_chart_provenance": "fixed renderer calibrated only from stored RGB; v3 roots use causally rendered frozen observations"}, "records": contexts}, indent=2, sort_keys=True) + "\n")
    outcome_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n")
    report = {"schema": SCHEMA, "cuda_preflight": cuda, "context": str(context_path.resolve()), "sealed_outcomes": str(outcome_path.resolve()),
        "source_dataset": str(source_path.resolve()), "source_arrays": str(arrays_path.resolve()), "root_counts": {name: sum(value == name for value in split.values()) for name in SPLIT_COUNTS},
        "v1_v2_outcome_root_overlap": sorted(set(split) & v12_outcomes), "v1_v2_action_hash_overlap": sorted(set(outcome_hashes) & v12_hashes),
        "v1_v2_template_overlap": sorted(set(outcome_templates) & v12_templates), "renderer_backed_frozen_observation_control": True,
        "claim_boundary": "Offline renderer-backed frozen-observation mechanism panel; not stored-RGB or visual live deployment evidence."}
    (output / "build_v3_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--v1-context", type=Path, required=True)
    parser.add_argument("--v2-context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--pca-checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-source", type=Path, required=True)
    parser.add_argument("--root-split-seed", type=int, default=DEFAULT_ROOT_SPLIT_SEED)
    parser.add_argument("--action-bank-seed", type=int, default=DEFAULT_ACTION_BANK_SEED)
    parser.add_argument("--train-action-seed", type=int, default=DEFAULT_TRAIN_ACTION_SEED)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
