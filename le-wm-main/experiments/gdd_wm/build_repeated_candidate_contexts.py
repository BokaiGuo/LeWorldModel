"""Build a provenance-preserving repeated candidate-set context manifest.

This utility joins independently generated candidate banks by the exact
action sequence, not by candidate index.  It creates the data structure
needed to estimate preference stability for the same root/action pair across
candidate-set contexts, but it never creates teacher scores or execution
labels.  An output can therefore still be rejected by the D1 audit for
teacher leakage or missing realized references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path}: expected an object with a records list")
    return dict(payload.get("metadata", {})), list(payload["records"])


def _action_hash(row: dict[str, Any]) -> str:
    encoded = json.dumps(row["action_sequence"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _context_id(index: int, path: Path, metadata: dict[str, Any]) -> str:
    bank = str(metadata.get("candidate_bank_hash", ""))
    if not bank:
        bank = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    return f"context_{index:02d}_{bank}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="independent candidate_dataset.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) < 2:
        raise ValueError("at least two candidate-set contexts are required")

    loaded = [_load(path) for path in args.input]
    context_maps: list[dict[tuple[str, str], dict[str, Any]]] = []
    for path, (_, rows) in zip(args.input, loaded):
        current: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            if "instance_id" not in row or "action_sequence" not in row:
                raise ValueError(f"{path}: every row needs instance_id and action_sequence")
            key = (str(row["instance_id"]), _action_hash(row))
            if key in current:
                raise ValueError(f"{path}: duplicate root/action sequence {key}")
            current[key] = row
        context_maps.append(current)

    common_keys = set(context_maps[0])
    for current in context_maps[1:]:
        common_keys &= set(current)
    if not common_keys:
        raise ValueError("no identical root/action sequences occur in every context")

    records: list[dict[str, Any]] = []
    roots: set[str] = set()
    contexts: list[dict[str, Any]] = []
    for index, (path, (metadata, source_rows)) in enumerate(zip(args.input, loaded)):
        context_id = _context_id(index, path, metadata)
        source_bank_hash = metadata.get("candidate_bank_hash") or (
            source_rows[0].get("candidate_bank_hash") if source_rows else None
        )
        source_teacher_model = metadata.get("teacher_model_id") or (
            source_rows[0].get("teacher_model_id") if source_rows else None
        )
        contexts.append({
            "context_id": context_id,
            "source": str(path),
            "candidate_bank_hash": source_bank_hash,
            "candidate_family": metadata.get("candidate_family"),
            "teacher_model_id": source_teacher_model,
        })
        for root, action_hash in sorted(common_keys):
            source = context_maps[index][(root, action_hash)]
            roots.add(root)
            row = dict(source)
            row["instance_id"] = root
            row["candidate_id"] = f"action_{action_hash}"
            row["teacher_set_id"] = f"{root}__{context_id}"
            row["candidate_bank_context_id"] = context_id
            row["candidate_bank_hash"] = source_bank_hash or source.get("candidate_bank_hash")
            row["source_candidate_id"] = source.get("candidate_id")
            row["source_candidate_dataset"] = str(path)
            # Preserve the separate source used to resolve predicted latent
            # arrays.  The context manifest itself is a copied/joined
            # artifact and must not be mistaken for the model export.
            row["predicted_latent_source_dataset"] = source.get(
                "predicted_latent_source_dataset",
                source.get("source_candidate_dataset"),
            )
            row["matched_action_hash"] = action_hash
            records.append(row)

    teacher_models = sorted({str(row.get("teacher_model_id", "missing")) for row in records})
    exact_leakage = bool(records) and all(
        abs(float(row.get("teacher_score", float("nan"))) + float(row.get("real_task_cost", float("nan")))) <= 1e-10
        for row in records
    )
    report = {
        "schema": "gdd_wm_repeated_candidate_contexts.v0.1",
        "metadata": {
            "context_count": len(args.input),
            "root_count": len(roots),
            "matched_root_action_pairs": len(common_keys),
            "record_count": len(records),
            "contexts": contexts,
            "teacher_models": teacher_models,
            "teacher_score_exact_negative_real_task_cost": exact_leakage,
            "preference_stability_estimable_structure": True,
            "teacher_labels_created_by_builder": False,
            "execution_labels_created_by_builder": False,
        },
        "records": records,
        "claim_boundary": "context-manifest construction only; not D-JEPA teacher validation or D2 readiness",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "metadata": report["metadata"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
