"""Historical PointMaze V5 recurrent action-effect debug replay.

V5 uses an eight-frame causal visual history and seven intervening executed
action blocks.  Its current GRU FiLM readout can modulate candidate effects by
history, but this panel is permanently debug-only: it materialized held labels
before selection, reused calibration roots for selection and authorization,
and confounded temporal value with action-support shift.  Use sealed V6 for
any new confirmation evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.decision_benchmark.audit_unseen_action_bank_contract import sequence_hash
from experiments.decision_benchmark.build_pointmaze_temporal_ood_bank import temporal_template_fingerprint
from experiments.decision_benchmark.build_pointmaze_temporal_ood_v3_panel import (
    _context_identities, _renderer_calibration_roots, _sha256, _split_roots, v3_action_sequences,
)
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import _bootstrap, _load_records, _selection, _simulate_sequence
from experiments.gpu_preflight import require_cuda_allocation, require_cuda_recurrent
from experiments.run_live_pointmaze_planning import (
    DatasetMatchedRenderer, _load_jepa_model, _make_context, _make_gdd_source_contract_context,
    _pool_visual, fit_visual_calibration, green_center, load_episode,
)


PANEL_SCHEMA = "pointmaze_temporal_ood_v5_recurrent_panel.v1"
PROTOCOL_MANIFEST_SCHEMA = "pointmaze_temporal_ood_v5_protocol_manifest.v1"
# The frozen panel is v5.  The FiLM readout below is a post-hoc structural
# repair after the original additive readout had already been evaluated, so a
# replay on that panel is explicitly a debugging result, not confirmation.
RESULT_SCHEMA = "pointmaze_temporal_ood_v5_recurrent_film_debug_result.v1"
V5_ACTION_FAMILY, V5_TRAIN_FAMILY = "world_fixed_random_five_pulse_temporal_ood_v5", "world_fixed_random_train_five_pulse_temporal_ood_v5"
ROOT_SEED, ACTION_SEED, TRAIN_ACTION_SEED = 202610240, 202610241, 202610242
SPLIT_COUNTS = {"train": 64, "calibration": 48, "outcome": 32}
HISTORY_FRAMES, FRAME_STRIDE, SOURCE_FRAME = 8, 5, 40
GOAL = np.asarray([1.2, 1.2], dtype=np.float64)
HIDDEN_WIDTHS, TEMPERATURES, SEEDS = (32, 64), (0.75, 1.0, 1.25), (8411, 8412, 8413, 8414, 8415)
V5_DEBUG_ONLY_MESSAGE = (
    "V5 is a historical debug-only action-OOD artifact: its builder materializes "
    "held outcomes before frozen model selection and reuses calibration roots for "
    "selection and authorization. Use the sealed V6 protocol for new evidence. "
    "Pass debug_replay=True only to reproduce a nonconfirmatory diagnostic."
)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _model_summary(model: nn.Module) -> dict[str, Any]:
    """A stable architecture witness for a runtime-trained module."""
    parameters = list(model.parameters())
    return {
        "class_name": model.__class__.__name__,
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad)),
        "state_dict_shapes": {name: list(value.shape) for name, value in model.state_dict().items()},
    }


def _file_manifest_entry(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _roots_by_split(rows: list[dict[str, Any]], *, description: str) -> dict[str, list[str]]:
    roots: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        try:
            split, root = str(row["split"]), str(row["start_episode_id"])
        except KeyError as error:
            raise ValueError(f"{description} row lacks {error.args[0]}") from error
        if split not in SPLIT_COUNTS:
            raise ValueError(f"{description} has unsupported split {split}")
        roots[split].add(root)
    return {split: sorted(roots.get(split, set())) for split in SPLIT_COUNTS}


def _candidate_bank_contract(rows: list[dict[str, Any]], *, family: str, description: str) -> dict[str, Any]:
    """Recompute candidate identity from sequence bytes; never trust row claims."""
    if not rows:
        raise ValueError(f"{description} candidate bank is empty")
    canonical: dict[int, dict[str, str]] = {}
    by_instance: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        required = {"candidate_id", "candidate_generator", "candidate_action_hash", "candidate_template_fingerprint", "candidate_bank_hash", "action_sequence"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"{description} row misses candidate fields {sorted(missing)}")
        if str(row["candidate_generator"]) != family:
            raise ValueError(f"{description} candidate generator mismatch")
        identifier = int(row["candidate_id"])
        instance_id = str(row.get("instance_id", ""))
        if not instance_id:
            raise ValueError(f"{description} row lacks instance_id")
        if identifier in by_instance[instance_id]:
            raise ValueError(f"{description} contains duplicate candidate {identifier} for {instance_id}")
        by_instance[instance_id].add(identifier)
        action = np.asarray(row["action_sequence"], dtype=np.float64)
        actual_hash, actual_template = sequence_hash(action), temporal_template_fingerprint(action)
        if str(row["candidate_action_hash"]) != actual_hash:
            raise ValueError(f"{description} candidate action hash mismatch")
        if str(row["candidate_template_fingerprint"]) != actual_template:
            raise ValueError(f"{description} candidate template mismatch")
        identity = {"action_hash": actual_hash, "template": actual_template}
        if identifier in canonical and canonical[identifier] != identity:
            raise ValueError(f"{description} candidate identity changes across roots")
        canonical[identifier] = identity
    candidate_ids = sorted(canonical)
    if candidate_ids != list(range(len(candidate_ids))):
        raise ValueError(f"{description} candidate ids are not contiguous from zero")
    for instance_id, identifiers in by_instance.items():
        if identifiers != set(candidate_ids):
            raise ValueError(
                f"{description} candidate bank is incomplete for {instance_id}: "
                f"expected {candidate_ids}, got {sorted(identifiers)}"
            )
    hashes = [canonical[identifier]["action_hash"] for identifier in candidate_ids]
    templates = [canonical[identifier]["template"] for identifier in candidate_ids]
    bank_hash = _json_hash(hashes)
    if any(str(row["candidate_bank_hash"]) != bank_hash for row in rows):
        raise ValueError(f"{description} candidate bank hash mismatch")
    return {
        "candidate_generator": family,
        "candidate_ids": candidate_ids,
        "action_hashes": hashes,
        "template_fingerprints": templates,
        "candidate_bank_hash": bank_hash,
        "template_bank_hash": _json_hash(templates),
    }


def _action_support_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize candidate amplitude support so OOD is not mistaken for memory value."""
    values = []
    for row in rows:
        action = np.asarray(row["action_sequence"], dtype=np.float64)
        if action.ndim != 2 or action.shape[1] != 2 or not np.isfinite(action).all():
            raise ValueError("candidate action sequence must have finite shape [T, 2]")
        values.append(action)
    flat = np.concatenate(values, axis=0)
    norms = np.linalg.norm(flat, axis=1)
    return {
        "component_min": np.min(flat, axis=0).tolist(),
        "component_max": np.max(flat, axis=0).tolist(),
        "absolute_component_max": float(np.max(np.abs(flat))),
        "pulse_norm_min": float(np.min(norms)),
        "pulse_norm_max": float(np.max(norms)),
        "pulse_norm_quantiles": np.quantile(norms, [0.05, 0.50, 0.95]).tolist(),
        "row_count": len(rows),
    }


def _validate_causal_history_rows(rows: list[dict[str, Any]], *, description: str) -> None:
    """Reject silent history leakage/misalignment in a sealed panel."""
    expected_frames = list(range(SOURCE_FRAME - (HISTORY_FRAMES - 1) * FRAME_STRIDE, SOURCE_FRAME + 1, FRAME_STRIDE))
    expected_interval = [expected_frames[0], SOURCE_FRAME]
    for row in rows:
        required = {"source_frame", "history_frame_indices", "history_action_interval", "history_actions_raw"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"{description} row misses causal history fields {sorted(missing)}")
        if int(row["source_frame"]) != SOURCE_FRAME:
            raise ValueError(f"{description} source_frame is not {SOURCE_FRAME}")
        if [int(value) for value in row["history_frame_indices"]] != expected_frames:
            raise ValueError(f"{description} history_frame_indices are not causal {expected_frames}")
        if [int(value) for value in row["history_action_interval"]] != expected_interval:
            raise ValueError(f"{description} history_action_interval is not {expected_interval}")
        history = np.asarray(row["history_actions_raw"], dtype=np.float64)
        if history.shape != ((HISTORY_FRAMES - 1) * FRAME_STRIDE, 2):
            raise ValueError(f"{description} history_actions_raw has wrong shape {history.shape}")


def _source_array_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        reference = str(row.get("realized_terminal_state_ref", ""))
        if not reference.startswith("arrays.npz::") or "::" not in reference:
            raise ValueError("source row has invalid realized terminal array reference")
        key = _array_key(reference)
        if not key or key in keys:
            raise ValueError("source rows must bind one unique realized terminal array key each")
        keys.append(key)
    return sorted(keys)


def _expected_label_bindings(records: list[dict[str, Any]], split: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in records:
        if str(row.get("split")) != split:
            continue
        key = f"{row.get('instance_id')}/{int(row.get('candidate_id'))}"
        if key in result:
            raise ValueError(f"duplicate {split} context label key {key}")
        result[key] = {
            "split": split,
            "start_episode_id": str(row.get("start_episode_id")),
            "candidate_action_hash": str(row.get("candidate_action_hash")),
            "candidate_template_fingerprint": str(row.get("candidate_template_fingerprint")),
            "candidate_bank_hash": str(row.get("candidate_bank_hash")),
        }
    return result


def _validate_labels(records: list[dict[str, Any]], split: str, path: Path) -> dict[str, Any]:
    values = _read_json_object(path, description=f"{split} outcomes")
    expected = _expected_label_bindings(records, split)
    if set(values) != set(expected):
        raise ValueError(f"{split} label keys do not exactly bind the context rows")
    required = {"real_task_cost", "realized_terminal_state", "split", "start_episode_id", "candidate_action_hash", "candidate_template_fingerprint", "candidate_bank_hash"}
    for key, binding in expected.items():
        label = values[key]
        if not isinstance(label, dict) or set(label) != required:
            raise ValueError(f"{split} label {key} has an invalid field contract")
        for field, expected_value in binding.items():
            if str(label[field]) != expected_value:
                raise ValueError(f"{split} label {key} {field} does not bind its context row")
    return {"keys": sorted(expected), "key_hash": _json_hash(sorted(expected)), "count": len(expected)}


def _protocol_contract(source_dataset: Path, source_arrays: Path, context: Path, calibration_outcomes: Path, outcome_outcomes: Path) -> dict[str, Any]:
    source = _read_json_object(source_dataset, description="source dataset")
    source_metadata, source_rows = source.get("metadata"), source.get("records")
    panel = _read_json_object(context, description="context")
    metadata, context_rows = panel.get("metadata"), panel.get("records")
    if not isinstance(source_metadata, dict) or not isinstance(source_rows, list):
        raise ValueError("source dataset lacks metadata/records")
    if not isinstance(metadata, dict) or not isinstance(context_rows, list):
        raise ValueError("context lacks metadata/records")
    if source_metadata.get("schema") != "pointmaze_temporal_ood_v5_source.v1" or source_metadata.get("history_frames") != HISTORY_FRAMES:
        raise ValueError("source dataset schema/history contract mismatch")
    if metadata.get("schema") != PANEL_SCHEMA or metadata.get("history_frames") != HISTORY_FRAMES:
        raise ValueError("context schema/history contract mismatch")
    if any(str(row.get("split")) != "train" for row in source_rows):
        raise ValueError("source dataset contains a non-training root")
    source_roots, context_roots = _roots_by_split(source_rows, description="source"), _roots_by_split(context_rows, description="context")
    all_context_roots = [root for roots in context_roots.values() for root in roots]
    if len(all_context_roots) != len(set(all_context_roots)):
        raise ValueError("context roots overlap across splits")
    if source_roots["train"] != context_roots["train"]:
        raise ValueError("source training roots do not exactly bind context training roots")
    source_candidates = _candidate_bank_contract(source_rows, family=V5_TRAIN_FAMILY, description="source")
    held_candidates = _candidate_bank_contract(context_rows, family=V5_ACTION_FAMILY, description="context")
    _validate_causal_history_rows(source_rows, description="source")
    _validate_causal_history_rows(context_rows, description="context")
    if set(source_candidates["action_hashes"]) & set(held_candidates["action_hashes"]):
        raise ValueError("source and held candidate action hashes overlap")
    if set(source_candidates["template_fingerprints"]) & set(held_candidates["template_fingerprints"]):
        raise ValueError("source and held candidate templates overlap")
    expected_array_keys = _source_array_keys(source_rows)
    try:
        with np.load(source_arrays, allow_pickle=False) as arrays:
            actual_array_keys = sorted(arrays.files)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid source arrays: {source_arrays}") from error
    if actual_array_keys != expected_array_keys:
        raise ValueError("source array key set does not exactly bind source references")
    checkpoint_info = {
        "encoder_checkpoint_sha256": metadata.get("encoder_checkpoint_sha256"),
        "pca_checkpoint_sha256": metadata.get("pca_checkpoint_sha256"),
    }
    if not all(isinstance(value, str) and value for value in checkpoint_info.values()):
        raise ValueError("context lacks checkpoint provenance")
    label_contract = {
        "calibration": _validate_labels(context_rows, "calibration", calibration_outcomes),
        "outcome": _validate_labels(context_rows, "outcome", outcome_outcomes),
    }
    return {
        "root_splits": context_roots,
        "source_candidates": source_candidates,
        "held_candidates": held_candidates,
        "source_action_support": _action_support_summary(source_rows),
        "held_action_support": _action_support_summary(context_rows),
        "action_support_shift": bool(
            source_candidates["action_hashes"]
            and (
                _action_support_summary(source_rows)["absolute_component_max"]
                != _action_support_summary(context_rows)["absolute_component_max"]
                or _action_support_summary(source_rows)["pulse_norm_min"]
                != _action_support_summary(context_rows)["pulse_norm_min"]
                or _action_support_summary(source_rows)["pulse_norm_max"]
                != _action_support_summary(context_rows)["pulse_norm_max"]
            )
        ),
        "source_array_keys_hash": _json_hash(actual_array_keys),
        "record_contract_hash": _json_hash({"roots": context_roots, "source_candidates": source_candidates, "held_candidates": held_candidates, "array_keys": actual_array_keys}),
        "checkpoint_info": checkpoint_info,
        "label_contract": label_contract,
    }


def write_protocol_manifest(source_dataset: Path, source_arrays: Path, context: Path, calibration_outcomes: Path, outcome_outcomes: Path, manifest: Path) -> dict[str, Any]:
    """Seal the v5 input contract immediately after panel construction."""
    contract = _protocol_contract(source_dataset, source_arrays, context, calibration_outcomes, outcome_outcomes)
    payload = {
        "schema": PROTOCOL_MANIFEST_SCHEMA,
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "files": {
            "source_dataset": _file_manifest_entry(source_dataset),
            "source_arrays": _file_manifest_entry(source_arrays),
            "context": _file_manifest_entry(context),
            "calibration_outcomes": _file_manifest_entry(calibration_outcomes),
            "outcome_outcomes": _file_manifest_entry(outcome_outcomes),
        },
        **contract,
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def validate_protocol_integrity(source_dataset: Path, source_arrays: Path, context: Path, calibration_outcomes: Path, outcome_outcomes: Path, manifest: Path | None = None) -> dict[str, Any]:
    """Fail closed unless all current inputs still match their sealed V5 contract."""
    manifest = manifest or context.parent / "temporal_ood_v5_protocol_manifest.json"
    payload = _read_json_object(manifest, description="protocol manifest")
    if payload.get("schema") != PROTOCOL_MANIFEST_SCHEMA:
        raise ValueError("protocol manifest schema mismatch")
    if payload.get("runner_source_sha256") != _sha256(Path(__file__).resolve()):
        raise ValueError("protocol manifest runner source hash mismatch")
    provided = {
        "source_dataset": source_dataset,
        "source_arrays": source_arrays,
        "context": context,
        "calibration_outcomes": calibration_outcomes,
        "outcome_outcomes": outcome_outcomes,
    }
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("protocol manifest lacks file bindings")
    for role, path in provided.items():
        entry = files.get(role)
        if not isinstance(entry, dict) or entry.get("path") != str(path.resolve()):
            raise ValueError(f"protocol manifest {role} path mismatch")
        if entry.get("sha256") != _sha256(path.resolve()):
            raise ValueError(f"protocol manifest {role} file hash mismatch")
    contract = _protocol_contract(source_dataset, source_arrays, context, calibration_outcomes, outcome_outcomes)
    for key in ("root_splits", "source_candidates", "held_candidates", "source_action_support", "held_action_support", "action_support_shift", "source_array_keys_hash", "record_contract_hash", "checkpoint_info", "label_contract"):
        if payload.get(key) != contract[key]:
            raise ValueError(f"protocol manifest {key} mismatch")
    return {"validated": True, "manifest": str(manifest.resolve()), "runner_source_sha256": payload["runner_source_sha256"], "root_counts": {split: len(roots) for split, roots in contract["root_splits"].items()}, "checkpoint_info": contract["checkpoint_info"]}


def causal_history8(frame: int, actions: np.ndarray) -> tuple[list[int], np.ndarray]:
    start = int(frame) - (HISTORY_FRAMES - 1) * FRAME_STRIDE
    indices = list(range(start, int(frame) + 1, FRAME_STRIDE))
    history = np.asarray(actions, dtype=np.float64)[start:int(frame)]
    if start < 0 or len(indices) != HISTORY_FRAMES or history.shape != ((HISTORY_FRAMES - 1) * FRAME_STRIDE, 2):
        raise ValueError("v5 requires eight causal frames and seven complete pre-candidate action blocks")
    return indices, history.copy()


def _encode_history8(model: Any, transform: Any, mean: np.ndarray, basis: np.ndarray, renderer: DatasetMatchedRenderer, calibration: Any, states: np.ndarray) -> np.ndarray:
    frames = causal_history8(SOURCE_FRAME, np.zeros((SOURCE_FRAME, 2)))[0]
    values = []
    with torch.inference_mode():
        for frame in frames:
            current, previous = renderer.render(states[frame]), renderer.render(states[frame - FRAME_STRIDE])
            center, old = green_center(current), green_center(previous)
            proprio = np.r_[calibration.position(center), calibration.velocity((center[0] - old[0], center[1] - old[1]))]
            context = _make_context([current], [proprio])
            encoded = model.encode(_make_gdd_source_contract_context(context, transform))
            pooled = _pool_visual(encoded["visual"][:, -1]).detach().cpu().numpy()[0]
            values.append((np.asarray(pooled, dtype=np.float64) - mean) @ basis.T)
    return np.stack(values)


def build_panel(dataset_root: Path, contexts: list[Path], output: Path, *, encoder_checkpoint: Path, pca_checkpoint: Path, dinov2_source: Path, device: str = "cuda", debug_replay: bool = False) -> dict[str, Any]:
    if not debug_replay:
        raise RuntimeError(V5_DEBUG_ONLY_MESSAGE)
    if device != "cuda":
        raise ValueError("formal v5 requires CUDA")
    cuda = require_cuda_allocation(device)
    old_outcomes, old_hashes, old_templates = _context_identities(*contexts)
    split = _split_roots(seed=ROOT_SEED, forbidden=old_outcomes)
    if set(split) & old_outcomes:
        raise RuntimeError("v5 root reuse from v1-v4 outcome panel")
    visual_roots = _renderer_calibration_roots(dataset_root, old_outcomes)
    visual_cal = fit_visual_calibration(dataset_root, visual_roots, max_frames=1600)
    reference_images, _ = load_episode(dataset_root, visual_roots[0])
    renderer = DatasetMatchedRenderer.from_episode(reference_images, visual_cal)
    root = Path(__file__).resolve().parents[3]
    model, transform, _stats, _m, _s = _load_jepa_model(root, dataset_root, encoder_checkpoint, dinov2_source, torch.device(device))
    pca = torch.load(pca_checkpoint, map_location="cpu", weights_only=False)
    mean, basis = np.asarray(pca["pca_mean"], dtype=np.float64), np.asarray(pca["pca_basis"], dtype=np.float64)
    states = np.asarray(torch.load(dataset_root / "states.pth", map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    actions = np.asarray(torch.load(dataset_root / "actions.pth", map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    held, train = v3_action_sequences(seed=ACTION_SEED, train=False), v3_action_sequences(seed=TRAIN_ACTION_SEED, train=True)
    hashes, templates = [sequence_hash(v) for v in held], [temporal_template_fingerprint(v) for v in held]
    train_hashes, train_templates = [sequence_hash(v) for v in train], [temporal_template_fingerprint(v) for v in train]
    if set(hashes) & old_hashes or set(templates) & old_templates:
        raise RuntimeError("v5 action identity overlaps v1-v4")
    if set(hashes) & set(train_hashes) or set(templates) & set(train_templates):
        raise RuntimeError("v5 training and held banks overlap")
    histories = {episode: _encode_history8(model, transform, mean, basis, renderer, visual_cal, states[episode]) for episode in sorted(split)}
    source, arrays, records = [], {}, []
    sealed: dict[str, dict[str, Any]] = {"calibration": {}, "outcome": {}}
    for episode in sorted(split):
        frames, prior_actions = causal_history8(SOURCE_FRAME, actions[episode])
        state, history = states[episode, SOURCE_FRAME].astype(np.float64), histories[episode]
        common = {"instance_id": f"pointmaze_v5_episode_{episode}", "split": split[episode], "environment": "PointMaze", "start_episode_id": str(episode), "source_frame": SOURCE_FRAME, "history_frame_indices": frames, "history_action_interval": [frames[0], SOURCE_FRAME], "history_actions_raw": prior_actions.tolist(), "source_latent_model": history[-1].tolist(), "source_history_model": history.reshape(-1).tolist(), "source_state_raw": state.tolist(), "goal_raw": GOAL.tolist()}
        if split[episode] == "train":
            for index, candidate in enumerate(train):
                key = f"terminal/v5/{episode}/{index}"; arrays[key] = _simulate_sequence(state, candidate)
                source.append({**common, "candidate_id": index, "candidate_generator": V5_TRAIN_FAMILY, "candidate_action_hash": train_hashes[index], "candidate_template_fingerprint": train_templates[index], "candidate_bank_hash": _json_hash(train_hashes), "action_sequence": candidate.tolist(), "realized_terminal_state_ref": f"arrays.npz::{key}"})
        for index, candidate in enumerate(held):
            key = f"{common['instance_id']}/{index}"
            records.append({**common, "candidate_id": index, "candidate_generator": V5_ACTION_FAMILY, "candidate_action_hash": hashes[index], "candidate_template_fingerprint": templates[index], "candidate_bank_hash": _json_hash(hashes), "action_sequence": candidate.tolist(), "outcome_label_status": "split_sealed_until_postfit"})
            if split[episode] in sealed:
                terminal = _simulate_sequence(state, candidate)
                sealed[split[episode]][key] = {
                    "real_task_cost": float(np.linalg.norm(terminal[:2] - GOAL)),
                    "realized_terminal_state": terminal.tolist(),
                    "split": split[episode],
                    "start_episode_id": str(episode),
                    "candidate_action_hash": hashes[index],
                    "candidate_template_fingerprint": templates[index],
                    "candidate_bank_hash": _json_hash(hashes),
                }
    output.mkdir(parents=True, exist_ok=True)
    source_path, arrays_path, context_path = output / "temporal_ood_v5_source_dataset.json", output / "temporal_ood_v5_source_arrays.npz", output / "temporal_ood_v5_context.json"
    cal_path, outcome_path = output / "temporal_ood_v5_calibration_outcomes.json", output / "temporal_ood_v5_outcomes.json"
    source_path.write_text(json.dumps({"metadata": {"schema": "pointmaze_temporal_ood_v5_source.v1", "history_frames": HISTORY_FRAMES}, "records": source}, indent=2, sort_keys=True) + "\n")
    np.savez(arrays_path, **arrays)
    context_path.write_text(json.dumps({"metadata": {"schema": PANEL_SCHEMA, "candidate_family": V5_ACTION_FAMILY, "root_split_seed": ROOT_SEED, "action_bank_seed": ACTION_SEED, "train_action_seed": TRAIN_ACTION_SEED, "history_frames": HISTORY_FRAMES, "history_is_causal": True, "history_actions_are_strictly_pre_candidate": True, "renderer_backed_frozen_observation_control": True, "v1_v4_outcome_roots_excluded": sorted(old_outcomes), "v1_v4_action_hash_overlap": [], "v1_v4_template_overlap": [], "encoder_checkpoint_sha256": _sha256(encoder_checkpoint), "pca_checkpoint_sha256": _sha256(pca_checkpoint)}, "records": records}, indent=2, sort_keys=True) + "\n")
    cal_path.write_text(json.dumps(sealed["calibration"], indent=2, sort_keys=True) + "\n"); outcome_path.write_text(json.dumps(sealed["outcome"], indent=2, sort_keys=True) + "\n")
    manifest_path = output / "temporal_ood_v5_protocol_manifest.json"
    write_protocol_manifest(source_path, arrays_path, context_path, cal_path, outcome_path, manifest_path)
    integrity = validate_protocol_integrity(source_path, arrays_path, context_path, cal_path, outcome_path, manifest_path)
    report = {"schema": PANEL_SCHEMA, "cuda_preflight": cuda, "context": str(context_path.resolve()), "source_dataset": str(source_path.resolve()), "source_arrays": str(arrays_path.resolve()), "calibration_outcomes": str(cal_path.resolve()), "outcome_outcomes": str(outcome_path.resolve()), "protocol_manifest": str(manifest_path.resolve()), "protocol_integrity": integrity, "root_counts": {name: sum(value == name for value in split.values()) for name in SPLIT_COUNTS}, "v1_v4_outcome_root_overlap": sorted(set(split) & old_outcomes), "v1_v4_action_hash_overlap": sorted(set(hashes) & old_hashes), "v1_v4_template_overlap": sorted(set(templates) & old_templates), "claim_boundary": "Renderer-backed offline temporal-memory control, not live visual deployment."}
    (output / "build_v5_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _array_key(ref: str) -> str: return str(ref).split("::", 1)[1]


def _flat_feature(row: dict[str, Any], family: str) -> np.ndarray:
    action = np.asarray(row["action_sequence"], dtype=np.float32).reshape(-1)
    if family == "action_only": context = np.empty(0, dtype=np.float32)
    elif family == "current_concat": context = np.asarray(row["source_latent_model"], dtype=np.float32)
    elif family == "fixed4_concat":
        history = np.asarray(row["source_history_model"], dtype=np.float32).reshape(HISTORY_FRAMES, -1)[-4:].reshape(-1)
        previous = np.asarray(row["history_actions_raw"], dtype=np.float32).reshape(HISTORY_FRAMES - 1, FRAME_STRIDE, 2)[-3:].reshape(-1)
        context = np.r_[history, previous]
    elif family == "privileged_state": context = np.asarray(row["source_state_raw"], dtype=np.float32)
    else: raise ValueError(f"unsupported flat family {family}")
    return np.r_[context, action]


def _flat_train(rows: list[dict[str, Any]], arrays: Any, family: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for row in rows:
        x.append(_flat_feature(row, family)); state = np.asarray(row["source_state_raw"], dtype=np.float64); terminal = np.asarray(arrays[_array_key(row["realized_terminal_state_ref"])], dtype=np.float64); y.append(terminal[:2] - state[:2])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


class FlatEffect(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)


class RecurrentModulatedEffect(nn.Module):
    """GRU history with a zero-initialized FiLM action-effect interaction.

    A purely additive readout ``b(history) + g(candidate)`` cannot change the
    relative predicted effects of two candidates as history changes.  Here the
    GRU state produces a feature-wise action modulation:

    ``b(h) + W[(1 + gamma(h)) * phi(a)]``.

    ``gamma`` starts at zero, preserving the former additive parameterization
    at initialization while allowing gradients to learn a history-dependent
    action effect under the same optimizer and normalization protocol.
    """
    def __init__(self, latent_dim: int, candidate_dim: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(latent_dim + 2, hidden), nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.memory_effect = nn.Linear(hidden, 2)
        self.action_encoder = nn.Sequential(nn.Linear(candidate_dim, hidden), nn.SiLU())
        self.action_effect = nn.Linear(hidden, 2)
        self.modulation = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
    def forward(self, history: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(history)
        _, final = self.gru(encoded)
        memory = final[-1]
        candidate_features = self.action_encoder(candidate)
        modulated_candidate = (1.0 + self.modulation(memory)) * candidate_features
        return self.memory_effect(memory) + self.action_effect(modulated_candidate)


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _fit_flat(x: np.ndarray, y: np.ndarray, *, seed: int, epochs: int, device: str) -> tuple[FlatEffect, dict[str, Any]]:
    _seed(seed); mean, scale, ym, ys = x.mean(0), np.maximum(x.std(0), 1e-3), y.mean(0), np.maximum(y.std(0), 1e-3); model = FlatEffect(x.shape[1]).to(device); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); tx, ty = torch.tensor((x-mean)/scale, device=device), torch.tensor((y-ym)/ys, device=device)
    for _ in range(epochs): opt.zero_grad(set_to_none=True); loss=((model(tx)-ty)**2).mean(); loss.backward(); opt.step()
    return model, {"x_mean": mean, "x_scale": scale, "y_mean": ym, "y_scale": ys, "model_summary": _model_summary(model)}


def _predict_flat(model: FlatEffect, fit: dict[str, Any], x: np.ndarray) -> np.ndarray:
    with torch.inference_mode(): return (model(torch.tensor((x-fit["x_mean"])/fit["x_scale"], dtype=torch.float32, device=next(model.parameters()).device)).cpu().numpy()*fit["y_scale"]+fit["y_mean"])


def _recurrent_tensors(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    history, candidate = [], []
    for row in rows:
        latent = np.asarray(row["source_history_model"], dtype=np.float32).reshape(HISTORY_FRAMES, -1)
        action = np.asarray(row["history_actions_raw"], dtype=np.float32).reshape(HISTORY_FRAMES-1, FRAME_STRIDE, 2).mean(axis=1)
        history.append(np.c_[latent, np.vstack([np.zeros((1,2), dtype=np.float32), action])]); candidate.append(np.asarray(row["action_sequence"], dtype=np.float32).reshape(-1))
    return np.asarray(history), np.asarray(candidate)


def _fit_recurrent(history: np.ndarray, candidate: np.ndarray, y: np.ndarray, *, hidden: int, seed: int, epochs: int, device: str) -> tuple[RecurrentModulatedEffect, dict[str, Any]]:
    _seed(seed); hm, hs = history.mean((0,1)), np.maximum(history.std((0,1)),1e-3); cm, cs = candidate.mean(0), np.maximum(candidate.std(0),1e-3); ym, ys = y.mean(0), np.maximum(y.std(0),1e-3); model = RecurrentModulatedEffect(history.shape[-1]-2,candidate.shape[-1],hidden).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); th=torch.tensor((history-hm)/hs,device=device); tc=torch.tensor((candidate-cm)/cs,device=device); ty=torch.tensor((y-ym)/ys,device=device)
    for _ in range(epochs): opt.zero_grad(set_to_none=True); loss=((model(th,tc)-ty)**2).mean(); loss.backward(); opt.step()
    return model,{"history_mean":hm,"history_scale":hs,"candidate_mean":cm,"candidate_scale":cs,"y_mean":ym,"y_scale":ys,"hidden":hidden,"model_summary":_model_summary(model)}


def _predict_recurrent(model: RecurrentModulatedEffect, fit: dict[str, Any], history: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    with torch.inference_mode(): value=model(torch.tensor((history-fit["history_mean"])/fit["history_scale"],dtype=torch.float32,device=next(model.parameters()).device),torch.tensor((candidate-fit["candidate_mean"])/fit["candidate_scale"],dtype=torch.float32,device=next(model.parameters()).device)).cpu().numpy()
    return value*fit["y_scale"]+fit["y_mean"]


def _labels(records: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    values = json.loads(path.read_text())
    result = []
    for row in records:
        key = f"{row['instance_id']}/{int(row['candidate_id'])}"
        if key not in values: raise ValueError(f"missing sealed outcome {key}")
        result.append({**row, "real_task_cost": float(values[key]["real_task_cost"])})
    return result


def _cost(records: list[dict[str, Any]], effects: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    state = np.asarray([row["source_state_raw"][:2] for row in records], dtype=np.float64)
    goal = np.asarray([row["goal_raw"][:2] for row in records], dtype=np.float64)
    return np.linalg.norm(state + float(temperature) * np.asarray(effects, dtype=np.float64) - goal, axis=1)


def _split_selection(records: list[dict[str, Any]], effects: np.ndarray, split: str, labels: Path, temperature: float = 1.0) -> dict[str, Any]:
    mask = np.asarray([str(row["split"]) == split for row in records])
    subset = [row for row, keep in zip(records, mask) if keep]
    return _selection(_labels(subset, labels), _cost(subset, effects[mask], temperature))


def _selection_for_rows(records: list[dict[str, Any]], effects: np.ndarray, labels: Path, temperature: float = 1.0) -> dict[str, Any]:
    """Evaluate one predeclared root subset without changing its labels."""
    return _selection(_labels(records, labels), _cost(records, effects, temperature))


def _calibration_roles(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split calibration roots before model selection to avoid double dipping."""
    roots = sorted({str(row["start_episode_id"]) for row in records if str(row["split"]) == "calibration"})
    if len(roots) < 4 or len(roots) % 2:
        raise ValueError("V5 requires an even calibration-root count of at least four")
    selection_roots, authorization_roots = set(roots[::2]), set(roots[1::2])
    selection = [row for row in records if str(row["start_episode_id"]) in selection_roots]
    authorization = [row for row in records if str(row["start_episode_id"]) in authorization_roots]
    if {str(row["start_episode_id"]) for row in selection} & {str(row["start_episode_id"]) for row in authorization}:
        raise RuntimeError("V5 calibration selection/authorization roots overlap")
    return selection, authorization


def _paired(method: dict[str, Any], control: dict[str, Any], *, seed: int, samples: int) -> dict[str, float]:
    roots = sorted(method["roots"])
    if roots != sorted(control["roots"]): raise ValueError("paired comparison root mismatch")
    values = np.asarray([control["roots"][root]["regret"]-method["roots"][root]["regret"] for root in roots], dtype=np.float64)
    return _bootstrap(values, roots, seed=seed, samples=samples)


def _ensemble_flat(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, *, epochs: int, device: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions, fits = [], []
    for seed in SEEDS:
        model, fit = _fit_flat(train_x, train_y, seed=seed, epochs=epochs, device=device); predictions.append(_predict_flat(model, fit, eval_x)); fits.append({"seed": seed, "model_summary": fit["model_summary"]})
    return np.mean(np.stack(predictions), axis=0), fits


def _ensemble_recurrent(train_h: np.ndarray, train_c: np.ndarray, train_y: np.ndarray, eval_h: np.ndarray, eval_c: np.ndarray, *, hidden: int, epochs: int, device: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions, fits = [], []
    for seed in SEEDS:
        model, fit = _fit_recurrent(train_h, train_c, train_y, hidden=hidden, seed=seed, epochs=epochs, device=device); predictions.append(_predict_recurrent(model, fit, eval_h, eval_c)); fits.append({"seed":seed,"hidden":hidden,"model_summary":fit["model_summary"]})
    return np.mean(np.stack(predictions),axis=0), fits


def run(source_dataset: Path, source_arrays: Path, context: Path, calibration_outcomes: Path, outcome_outcomes: Path, output: Path, *, epochs: int = 300, bootstrap_samples: int = 5000, device: str = "cuda", debug_replay: bool = False) -> dict[str, Any]:
    if not debug_replay:
        raise RuntimeError(V5_DEBUG_ONLY_MESSAGE)
    if device != "cuda": raise ValueError("formal v5 requires CUDA")
    integrity = validate_protocol_integrity(source_dataset, source_arrays, context, calibration_outcomes, outcome_outcomes)
    cuda = require_cuda_recurrent(device)
    metadata, records = _load_records(context)
    if metadata.get("schema") != PANEL_SCHEMA or metadata.get("history_frames") != HISTORY_FRAMES: raise ValueError("v5 context contract mismatch")
    counts = {name: len({str(row["instance_id"]) for row in records if str(row["split"])==name}) for name in SPLIT_COUNTS}
    if counts != SPLIT_COUNTS: raise ValueError(f"unexpected v5 split counts {counts}")
    source_meta, source_records = _load_records(source_dataset)
    if source_meta.get("history_frames") != HISTORY_FRAMES: raise ValueError("v5 source lacks 8-frame causal memory")
    arrays=np.load(source_arrays,allow_pickle=False); train=[r for r in source_records if str(r["split"])=="train"]
    flat_predictions, training = {}, {}
    for family in ("action_only","current_concat","fixed4_concat","privileged_state"):
        tx,ty=_flat_train(train,arrays,family); ex=np.asarray([_flat_feature(row,family) for row in records],dtype=np.float32); flat_predictions[family],training[family]=_ensemble_flat(tx,ty,ex,epochs=epochs,device=device)
    th,tc=_recurrent_tensors(train); ex_h,ex_c=_recurrent_tensors(records); states=np.asarray([r["source_state_raw"][:2] for r in train],dtype=np.float64); train_terminal=np.asarray([arrays[_array_key(r["realized_terminal_state_ref"])][:2] for r in train],dtype=np.float64); ty=(train_terminal-states).astype(np.float32)
    recurrent, rec_training={},{}
    for hidden in HIDDEN_WIDTHS: recurrent[hidden],rec_training[hidden]=_ensemble_recurrent(th,tc,ty,ex_h,ex_c,hidden=hidden,epochs=epochs,device=device)
    # All predictors are frozen.  Split calibration roots before opening their
    # labels: one half selects architecture/temperature, the other half is an
    # independent authorization gate.  This fixes V5's former double dipping.
    calibration_selection_records, calibration_authorization_records = _calibration_roles(records)
    selection_indices = [records.index(row) for row in calibration_selection_records]
    authorization_indices = [records.index(row) for row in calibration_authorization_records]
    calibration={family:_selection_for_rows(calibration_selection_records,pred[selection_indices],calibration_outcomes) for family,pred in flat_predictions.items()}
    choices=[]
    for hidden,pred in recurrent.items():
        for temp in TEMPERATURES:
            selected=_selection_for_rows(calibration_selection_records,pred[selection_indices],calibration_outcomes,temp); choices.append({"hidden":hidden,"temperature":temp,"selection":selected,"mean_regret":selected["mean_regret"]})
    selected_choice=min(choices,key=lambda x:(x["mean_regret"],x["hidden"],abs(x["temperature"]-1.0)))
    calibration["recurrent_gru"] = selected_choice["selection"]
    authorization={family:_selection_for_rows(calibration_authorization_records,pred[authorization_indices],calibration_outcomes) for family,pred in flat_predictions.items()}
    authorization["recurrent_gru"]=_selection_for_rows(calibration_authorization_records,recurrent[selected_choice["hidden"]][authorization_indices],calibration_outcomes,selected_choice["temperature"])
    gate=_paired(authorization["recurrent_gru"],authorization["action_only"],seed=2026102450,samples=bootstrap_samples)
    permitted=float(gate["lower"])>0.0
    output.mkdir(parents=True,exist_ok=True)
    selected_model_summary = rec_training[selected_choice["hidden"]][0]["model_summary"]
    runtime_provenance = {"runner_source_sha256": _sha256(Path(__file__).resolve()), "model_class": selected_model_summary["class_name"], "model_summary": selected_model_summary, "protocol_integrity": integrity}
    frozen={"schema":"pointmaze_temporal_ood_v5_calibration_selection.v2","selected_hidden":selected_choice["hidden"],"selected_temperature":selected_choice["temperature"],"architecture_temperature_grid":[{"hidden":c["hidden"],"temperature":c["temperature"],"mean_regret":c["mean_regret"]} for c in choices],"selection_root_ids":sorted({str(row["start_episode_id"]) for row in calibration_selection_records}),"authorization_root_ids":sorted({str(row["start_episode_id"]) for row in calibration_authorization_records}),"authorization":"recurrent paired authorization-root regret-gain lower CI > 0","authorization_gate":gate,"permitted":permitted,"outcome_file_not_opened_when_written":True,"runtime_provenance":runtime_provenance}
    (output/"temporal_ood_v5_frozen_calibration_selection.json").write_text(json.dumps(frozen,indent=2,sort_keys=True)+"\n")
    outcome={family:_split_selection(records,pred,"outcome",outcome_outcomes) for family,pred in flat_predictions.items()}
    outcome["recurrent_gru"]=_split_selection(records,recurrent[selected_choice["hidden"]],"outcome",outcome_outcomes,selected_choice["temperature"])
    primary=_paired(outcome["recurrent_gru"],outcome["action_only"],seed=2026102451,samples=bootstrap_samples)
    accepted=permitted and float(primary["lower"])>0.0
    report={"schema":RESULT_SCHEMA,"cuda_preflight":cuda,"runtime_provenance":runtime_provenance,"counts":counts,"arms":["action_only","current_concat","fixed4_concat","recurrent_gru","privileged_state"],"recurrent_structure":"GRU(state from 8 causal latent/action steps) + zero-initialized FiLM-modulated candidate-action effect readout","history_modulates_action_effect":True,"training":{**training,"recurrent":rec_training},"calibration_selection":calibration,"calibration_authorization":authorization,"calibration":calibration,"frozen_calibration_selection":frozen,"outcome":outcome,"primary":{"recurrent_over_action_only":primary,"calibration_gate_passed":permitted,"criterion":"positive paired held lower CI and independent authorization-root gate","satisfied":accepted},"policy_status":"recurrent_permitted_offline_only" if accepted else "action_only_fallback","protocol_status":"debug_replay_nonconfirmatory","claim_boundary":"Post-hoc FiLM repair replay on an already-opened v5 outcome panel. It is a GPU debugging diagnostic, not independent evidence; any promising direction requires a fresh-root v6 replication. Offline renderer-backed action-OOD only, not live visual deployment."}
    (output/"temporal_ood_v5_recurrent_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    build=sub.add_parser("build-panel"); build.add_argument("--dataset-root",type=Path,required=True); build.add_argument("--contexts",type=Path,nargs=4,required=True); build.add_argument("--output",type=Path,required=True); build.add_argument("--encoder-checkpoint",type=Path,required=True); build.add_argument("--pca-checkpoint",type=Path,required=True); build.add_argument("--dinov2-source",type=Path,required=True); build.add_argument("--device",choices=("cuda",),default="cuda"); build.add_argument("--debug-replay",action="store_true",help="acknowledge that V5 is nonconfirmatory; use V6 for fresh evidence")
    runner=sub.add_parser("run"); runner.add_argument("--source-dataset",type=Path,required=True); runner.add_argument("--source-arrays",type=Path,required=True); runner.add_argument("--context",type=Path,required=True); runner.add_argument("--calibration-outcomes",type=Path,required=True); runner.add_argument("--outcome-outcomes",type=Path,required=True); runner.add_argument("--output",type=Path,required=True); runner.add_argument("--epochs",type=int,default=300); runner.add_argument("--bootstrap-samples",type=int,default=5000); runner.add_argument("--device",choices=("cuda",),default="cuda"); runner.add_argument("--debug-replay",action="store_true",help="acknowledge that V5 is nonconfirmatory; use V6 for fresh evidence")
    args=vars(parser.parse_args()); command=args.pop("command"); result=build_panel(**args) if command=="build-panel" else run(**args); print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
