"""Merge split RGB-successor reports into one strict candidate-package artifact.

This utility only joins already validated artifacts. It refuses mismatched
candidate datasets, duplicate keys, incomplete coverage, or inconsistent
array values. It does not alter teacher labels, costs, actions, or splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise ValueError(f"{path}: expected report object with rows")
    return value


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["instance_id"]), int(row["candidate_id"]), str(row["split"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def merge(candidate_dataset: Path, reports: list[Path], arrays_paths: list[Path], output: Path) -> dict[str, Any]:
    if len(reports) != len(arrays_paths) or not reports:
        raise ValueError("reports and arrays_paths must be non-empty and have equal length")
    dataset = json.loads(candidate_dataset.read_text(encoding="utf-8"))
    rows = dataset.get("records")
    if not isinstance(rows, list):
        raise ValueError("candidate dataset must contain records")
    candidate_keys = {_key(row) for row in rows}
    merged_rows: list[dict[str, Any]] = []
    merged_arrays: dict[str, np.ndarray] = {}
    report_splits: list[str] = []
    for report_path, arrays_path in zip(reports, arrays_paths):
        report = _load(report_path)
        declared = report.get("protocol", {}).get("candidate_dataset")
        if not isinstance(declared, str) or Path(declared).resolve() != candidate_dataset.resolve():
            raise ValueError(f"{report_path}: candidate_dataset provenance mismatch")
        gates = report.get("gates", {})
        if not all(gates.get(name) is True for name in ("realized_rgb_latents_available", "pca_projected_successor_comparison", "renderer_consistency")):
            raise ValueError(f"{report_path}: RGB successor gates are not all true")
        arrays = np.load(arrays_path, allow_pickle=False)
        local_keys: set[tuple[str, int, str]] = set()
        for row in report["rows"]:
            key = _key(row)
            if key in local_keys:
                raise ValueError(f"{report_path}: duplicate row key {key}")
            if key not in candidate_keys:
                raise ValueError(f"{report_path}: row key not in candidate dataset {key}")
            local_keys.add(key)
            split = str(row["split"])
            if split in report_splits:
                raise ValueError(f"split {split!r} appears in multiple reports")
            npz_key = f"projection::{key[0]}/{key[1]:03d}"
            if npz_key not in arrays.files:
                raise ValueError(f"{arrays_path}: missing {npz_key}")
            value = np.asarray(arrays[npz_key], dtype=np.float64).reshape(-1)
            declared_projection = row.get("realized_terminal_projection")
            if not isinstance(declared_projection, list) or not np.allclose(value, np.asarray(declared_projection), rtol=0.0, atol=1e-6):
                raise ValueError(f"{report_path}: projection mismatch for {npz_key}")
            if npz_key in merged_arrays and not np.allclose(merged_arrays[npz_key], value, rtol=0.0, atol=1e-8):
                raise ValueError(f"conflicting realized array {npz_key}")
            merged_arrays[npz_key] = value
            merged_rows.append(dict(row))
        report_splits.append(str(report["rows"][0]["split"]) if report["rows"] else "empty")
    merged_keys = {_key(row) for row in merged_rows}
    if merged_keys != candidate_keys:
        raise ValueError(f"coverage mismatch: missing={len(candidate_keys - merged_keys)} extra={len(merged_keys - candidate_keys)}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "realized_latents.npz", **merged_arrays)
    merged_report = {
        "schema": "gdd_wm_rgb_successor_merged_report.v0.1",
        "status": "passed",
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "source_reports": [str(path.resolve()) for path in reports],
            "source_arrays": [str(path.resolve()) for path in arrays_paths],
            "split_merge": sorted(report_splits),
            "key_contract": "(instance_id, candidate_id, split) / projection::<instance_id>/<candidate_id:03d>",
            "simulator_state_role": "action execution and posthoc renderer construction only",
        },
        "gates": {
            "realized_rgb_latents_available": True,
            "pca_projected_successor_comparison": True,
            "renderer_consistency": True,
        },
        "rows": sorted(merged_rows, key=_key),
        "summary": {"rows": len(merged_rows), "split_count": len(report_splits), "array_count": len(merged_arrays)},
        "claim_boundary": "provenance-preserving RGB successor join; no teacher or planning claim",
    }
    report_path = output / "rgb_successor_report.json"
    report_path.write_text(json.dumps(merged_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "schema": "gdd_wm_rgb_successor_split_merge_audit.v0.1",
        "status": "passed",
        "candidate_dataset_sha256": _sha256(candidate_dataset),
        "summary": merged_report["summary"],
        "claim_boundary": merged_report["claim_boundary"],
    }
    (output / "merge_report.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", required=True, type=Path)
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--arrays", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(merge(args.candidate_dataset, args.report, args.arrays, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
