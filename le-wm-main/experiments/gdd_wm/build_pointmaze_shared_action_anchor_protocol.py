"""Build valid repeated PointMaze candidate contexts from one frozen bank.

The existing frozen predictive GDD teacher builder needs three candidate-set
contexts per root.  Each context must share at least two *exact* action
sequences while still containing declared context-specific alternatives.  It
also needs train/calibration/outcome roots to remain disjoint and held action
sequences to be absent from teacher training.  Historic PointMaze exports did
not satisfy both requirements simultaneously because every root reused one
global world-coordinate bank.

This standalone protocol builder creates three context datasets from one
already predicted frozen-checkpoint bank.  It never re-fits a predictor, reads
execution costs only as carried audit labels, or edits the source bank.  For
each root it reserves five disjoint source actions for its assigned root split:
two shared anchors plus one declared context-specific action for each of three
contexts.  The three split slots are globally disjoint, which makes held action
novelty auditable without fabricating model outputs.

The result is an input protocol for
``build_pointmaze_frozen_predictive_teacher_package.py``.  It is not itself a
teacher, a D2 authorization, or planning evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "gdd_wm_pointmaze_shared_action_anchor_protocol.v0.1"
SPLITS = ("train", "calibration", "outcome")
CONTEXT_COUNT = 3
ANCHOR_COUNT = 2
CONTEXT_ACTION_COUNT = 1
SLOT_COUNT = ANCHOR_COUNT + CONTEXT_COUNT * CONTEXT_ACTION_COUNT
MIN_ACTIONS_PER_ROOT = len(SPLITS) * SLOT_COUNT


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _action_hash(row: dict[str, Any]) -> str:
    sequence = row.get("action_sequence")
    if not isinstance(sequence, list) or not sequence:
        raise ValueError(f"{row.get('instance_id')}: action_sequence must be a non-empty list")
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError(f"{row.get('instance_id')}: action_sequence must be finite [steps, action_dim]")
    return _sha256_bytes(_canonical_bytes(sequence))


def _terminal_key(row: dict[str, Any], source: Path) -> str:
    reference = str(row.get("predicted_terminal_latent_ref", row.get("predicted_latent_ref", "")))
    if "arrays.npz::" not in reference:
        raise ValueError(f"{source}: {row.get('instance_id')} is missing predicted terminal latent reference")
    return reference.split("arrays.npz::", 1)[1]


def _world_model_sha256(metadata: dict[str, Any], source: Path) -> str:
    declared = metadata.get("world_model_sha256")
    if isinstance(declared, str) and len(declared) == 64:
        return declared
    checkpoint = metadata.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"{source}: metadata must declare checkpoint or world_model_sha256")
    path = Path(checkpoint)
    if not path.is_absolute():
        path = source.parent / path
    if not path.exists():
        raise ValueError(f"{source}: cannot pin missing frozen checkpoint {path}")
    return _sha256_file(path)


def _load_source(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], str]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    metadata, rows = payload.get("metadata"), payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(rows, list) or not rows:
        raise ValueError(f"{source}: expected non-empty metadata and records")
    if str(metadata.get("environment", rows[0].get("environment", ""))) != "PointMaze":
        raise ValueError(f"{source}: protocol accepts PointMaze only")
    arrays_path = source.parent / "arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"{source}: missing arrays.npz")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        values = {key: np.asarray(arrays[key]) for key in arrays.files}
    return dict(metadata), [dict(row) for row in rows], values, _world_model_sha256(metadata, source)


def _root_groups(rows: list[dict[str, Any]], source: Path) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    split_by_root: dict[str, str] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    horizons: set[int] = set()
    for row in rows:
        root, split = str(row.get("instance_id", "")), str(row.get("split", ""))
        if not root or split not in SPLITS:
            raise ValueError(f"{source}: each record requires root and train/calibration/outcome split")
        if split_by_root.setdefault(root, split) != split:
            raise ValueError(f"{source}: root {root} occurs in multiple splits")
        try:
            horizons.add(int(row.get("predicted_horizon", 0)))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source}: invalid predicted_horizon") from error
        groups[root].append(row)
    if set(split_by_root.values()) != set(SPLITS):
        raise ValueError(f"{source}: requires non-empty root-disjoint train/calibration/outcome assignments")
    if len(horizons) != 1 or next(iter(horizons)) < 1:
        raise ValueError(f"{source}: records must share one positive predicted_horizon")
    for root, root_rows in groups.items():
        hashes = [_action_hash(row) for row in root_rows]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"{source}: root {root} has duplicate action sequences")
        if len(hashes) < MIN_ACTIONS_PER_ROOT:
            raise ValueError(
                f"{source}: root {root} needs at least {MIN_ACTIONS_PER_ROOT} distinct action sequences "
                "for split-isolated anchors and context-specific actions"
            )
    return split_by_root, groups


def _selected_rows(root_rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Deterministically reserve disjoint slots for a root's declared split."""
    ordered = sorted(root_rows, key=lambda row: (_action_hash(row), str(row.get("candidate_id", ""))))
    offset = SPLITS.index(split) * SLOT_COUNT
    slot_rows = ordered[offset : offset + SLOT_COUNT]
    if len(slot_rows) != SLOT_COUNT:
        raise AssertionError("root action capacity was checked before selection")
    return slot_rows, [_action_hash(row) for row in slot_rows[:ANCHOR_COUNT]]


def _write_context(
    *,
    context_index: int,
    output: Path,
    metadata: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
    split_by_root: dict[str, str],
    source_arrays: dict[str, np.ndarray],
    source: Path,
    source_sha256: str,
    source_arrays_sha256: str,
    world_model_sha256: str,
) -> dict[str, Any]:
    context_id = f"shared_anchor_context_{context_index:02d}"
    directory = output / "contexts" / context_id
    directory.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    root_banks: dict[str, Any] = {}
    for root in sorted(groups):
        split = split_by_root[root]
        slot_rows, anchor_hashes = _selected_rows(groups[root], split)
        chosen = slot_rows[:ANCHOR_COUNT] + [slot_rows[ANCHOR_COUNT + context_index]]
        context_hashes: list[str] = []
        for position, source_row in enumerate(chosen):
            role = "shared_anchor" if position < ANCHOR_COUNT else "context_specific"
            action_hash = _action_hash(source_row)
            terminal_key = _terminal_key(source_row, source)
            if terminal_key not in source_arrays:
                raise ValueError(f"{source}: arrays.npz missing {terminal_key}")
            array_key = f"predicted::{root}/{context_id}/{action_hash[:16]}"
            arrays[array_key] = source_arrays[terminal_key]
            row = dict(source_row)
            row.update(
                {
                    "candidate_id": f"{context_id}:{role}:{source_row.get('candidate_id')}",
                    "source_candidate_id": source_row.get("candidate_id"),
                    "candidate_bank_context_id": context_id,
                    "teacher_set_id": f"{root}__{context_id}",
                    "selection_role": role,
                    "action_sequence_sha256": action_hash,
                    "predicted_terminal_latent_ref": f"arrays.npz::{array_key}",
                    "predicted_latent_ref": f"arrays.npz::{array_key}",
                }
            )
            records.append(row)
            context_hashes.append(action_hash)
        root_banks[root] = {
            "split": split,
            "shared_anchor_action_sequence_sha256": anchor_hashes,
            "context_specific_action_sequence_sha256": context_hashes[-1],
            "root_context_bank_sha256": _sha256_bytes(_canonical_bytes(context_hashes)),
        }
    context_bank_sha256 = _sha256_bytes(_canonical_bytes(root_banks))
    context_metadata = {
        "schema": SCHEMA,
        "environment": "PointMaze",
        "checkpoint": str(metadata.get("checkpoint", "")),
        "world_model_sha256": world_model_sha256,
        "candidate_family": "gdd_shared_action_anchor_v1",
        "candidate_bank_coordinate_frame": str(metadata.get("candidate_bank_coordinate_frame", "")),
        "candidate_context_id": context_id,
        "candidate_context_count": CONTEXT_COUNT,
        "shared_anchor_count": ANCHOR_COUNT,
        "declared_context_specific_action_count": CONTEXT_ACTION_COUNT,
        "candidate_bank_hash": context_bank_sha256,
        "predicted_horizon": int(records[0]["predicted_horizon"]),
        "root_split_assignment": str(metadata.get("split_assignment", "source_frozen")),
        "source_candidate_dataset": str(source.resolve()),
        "source_candidate_dataset_sha256": source_sha256,
        "source_arrays_sha256": source_arrays_sha256,
        "claim_boundary": (
            "A fixed-checkpoint candidate-context protocol only. It does not fit a teacher, "
            "read execution outcomes for selection, authorize D2, or establish planning value."
        ),
    }
    dataset = directory / "candidate_dataset.json"
    arrays_path = directory / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    dataset.write_text(json.dumps({"metadata": context_metadata, "records": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "context_id": context_id,
        "relative_candidate_dataset": str(dataset.relative_to(output)),
        "relative_arrays": str(arrays_path.relative_to(output)),
        "candidate_dataset_sha256": _sha256_file(dataset),
        "arrays_sha256": _sha256_file(arrays_path),
        "candidate_bank_sha256": context_bank_sha256,
        "record_count": len(records),
        "per_root": root_banks,
    }


def build_protocol(source_candidate_dataset: Path, output: Path) -> dict[str, Any]:
    """Write three auditable repeated contexts without mutating source data."""
    source = source_candidate_dataset.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite protocol output {output}")
    metadata, rows, arrays, world_model_sha256 = _load_source(source)
    split_by_root, groups = _root_groups(rows, source)
    source_sha256 = _sha256_file(source)
    source_arrays_sha256 = _sha256_file(source.parent / "arrays.npz")
    output.mkdir(parents=True, exist_ok=False)
    contexts = [
        _write_context(
            context_index=index,
            output=output,
            metadata=metadata,
            groups=groups,
            split_by_root=split_by_root,
            source_arrays=arrays,
            source=source,
            source_sha256=source_sha256,
            source_arrays_sha256=source_arrays_sha256,
            world_model_sha256=world_model_sha256,
        )
        for index in range(CONTEXT_COUNT)
    ]
    split_roots = {split: sorted(root for root, value in split_by_root.items() if value == split) for split in SPLITS}
    root_split_sha256 = _sha256_bytes(_canonical_bytes(split_roots))
    report = {
        "schema": SCHEMA,
        "status": "built_valid_shared_action_anchor_protocol",
        "source_candidate_dataset": str(source),
        "source_candidate_dataset_sha256": source_sha256,
        "source_arrays_sha256": source_arrays_sha256,
        "world_model_sha256": world_model_sha256,
        "predicted_horizon": int(rows[0]["predicted_horizon"]),
        "root_split_assignment_sha256": root_split_sha256,
        "root_split_counts": {split: len(split_roots[split]) for split in SPLITS},
        "root_split_assignments": split_roots,
        "shared_anchor_count": ANCHOR_COUNT,
        "context_specific_action_count": CONTEXT_ACTION_COUNT,
        "contexts": contexts,
        "context_datasets": [str(output / item["relative_candidate_dataset"]) for item in contexts],
        "claim_boundary": (
            "This output is only a provenance-bound input to the existing frozen predictive teacher builder. "
            "It neither validates the teacher nor bypasses D1/D2 repairability gates."
        ),
    }
    report["protocol_sha256"] = _sha256_bytes(_canonical_bytes({key: value for key, value in report.items() if key != "protocol_sha256"}))
    (output / "protocol_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-candidate-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_protocol(args.source_candidate_dataset, args.output)
    print(json.dumps({key: report[key] for key in ("status", "protocol_sha256", "context_datasets", "root_split_counts")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
