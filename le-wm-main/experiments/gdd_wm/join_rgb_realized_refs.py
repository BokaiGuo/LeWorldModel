"""Join an existing RGB successor artifact into a copied GDD-WM candidate bank.

This is a provenance-only operation.  It does not recompute teacher scores,
real costs, splits, actions, or predictions.  The join is accepted only when
the RGB report explicitly names the exact input candidate dataset and its
realized projection artifact covers every candidate key exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


PROTECTED_FIELDS = (
    "instance_id",
    "candidate_id",
    "split",
    "teacher_set_id",
    "teacher_model_id",
    "teacher_score",
    "teacher_rank",
    "real_task_cost",
    "real_success",
    "action_sequence",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _candidate_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row["instance_id"]), int(row["candidate_id"]), str(row["split"]))


def _report_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row["instance_id"]), int(row["candidate_id"]), str(row["split"]))


def join_artifact(
    candidate_dataset: Path,
    rgb_report: Path,
    realized_npz: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Perform a strict, non-destructive reference join and write an artifact."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "gdd_wm_rgb_realized_reference_join.v0.1",
        "status": "rejected_fail_closed",
        "claim_boundary": (
            "provenance-safe reference join only; this does not validate a D-JEPA teacher, "
            "physical causality, or closed-loop planning"
        ),
        "inputs": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "rgb_report": str(rgb_report.resolve()),
            "realized_latents": str(realized_npz.resolve()),
        },
        "errors": [],
        "warnings": [],
    }

    try:
        dataset = _load_json(candidate_dataset)
        rgb = _load_json(rgb_report)
        records = dataset.get("records")
        rgb_rows = rgb.get("rows")
        if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
            raise ValueError("candidate dataset must contain a list of object records")
        if not isinstance(rgb_rows, list) or not all(isinstance(row, dict) for row in rgb_rows):
            raise ValueError("RGB report must contain a list of object rows")

        declared_candidate = rgb.get("protocol", {}).get("candidate_dataset")
        if not isinstance(declared_candidate, str):
            raise ValueError("RGB report protocol.candidate_dataset is missing")
        if Path(declared_candidate).resolve() != candidate_dataset.resolve():
            raise ValueError(
                "RGB report candidate_dataset does not exactly match the join input: "
                f"{declared_candidate!r}"
            )

        gates = rgb.get("gates", {})
        for gate in ("realized_rgb_latents_available", "pca_projected_successor_comparison", "renderer_consistency"):
            if gates.get(gate) is not True:
                raise ValueError(f"RGB report gate {gate!r} is not true")

        candidate_keys = [_candidate_key(row) for row in records]
        report_keys = [_report_key(row) for row in rgb_rows]
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError("candidate dataset contains duplicate (instance, candidate, split) keys")
        if len(report_keys) != len(set(report_keys)):
            raise ValueError("RGB report contains duplicate (instance, candidate, split) keys")
        if set(candidate_keys) != set(report_keys):
            missing = sorted(set(candidate_keys) - set(report_keys))[:10]
            extra = sorted(set(report_keys) - set(candidate_keys))[:10]
            raise ValueError(f"RGB key coverage mismatch: missing={missing}, extra={extra}")

        arrays = np.load(realized_npz, allow_pickle=False)
        expected_npz_keys: set[str] = set()
        refs: dict[tuple[str, int, str], str] = {}
        for row in rgb_rows:
            key = _report_key(row)
            instance, candidate, _split = key
            npz_key = f"projection::{instance}/{candidate:03d}"
            expected_npz_keys.add(npz_key)
            if npz_key not in arrays.files:
                raise ValueError(f"realized latent NPZ key is missing: {npz_key}")
            value = np.asarray(arrays[npz_key])
            if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
                raise ValueError(f"realized latent array is invalid: {npz_key}")
            declared_projection = row.get("realized_terminal_projection")
            if not isinstance(declared_projection, list) or len(declared_projection) != value.size:
                raise ValueError(f"RGB report/NPZ projection mismatch: {npz_key}")
            if not np.allclose(value, np.asarray(declared_projection), rtol=0.0, atol=1.0e-6):
                raise ValueError(f"RGB report/NPZ values disagree: {npz_key}")
            refs[key] = f"realized_latents.npz::{npz_key}"

        joined_records: list[dict[str, Any]] = []
        changed = 0
        for row in records:
            key = _candidate_key(row)
            if row.get("realized_latent_ref") is not None:
                raise ValueError(f"refusing to overwrite existing realized_latent_ref for {key}")
            joined = dict(row)
            joined["realized_latent_ref"] = refs[key]
            for field in PROTECTED_FIELDS:
                if joined.get(field) != row.get(field):
                    raise ValueError(f"protected field changed during join: {field} at {key}")
            joined_records.append(joined)
            changed += 1

        copied_npz = output_dir / "realized_latents.npz"
        shutil.copy2(realized_npz, copied_npz)
        metadata = dict(dataset.get("metadata", {}))
        metadata["realized_latent_status"] = "available_from_rgb_successor_v1_provenance_join"
        metadata["realized_latent_join"] = {
            "schema": "gdd_wm_rgb_realized_reference_join.v0.1",
            "source_rgb_report": str(rgb_report.resolve()),
            "source_rgb_report_sha256": _sha256(rgb_report),
            "source_realized_latents": str(realized_npz.resolve()),
            "source_realized_latents_sha256": _sha256(realized_npz),
            "source_candidate_dataset": str(candidate_dataset.resolve()),
            "source_candidate_dataset_sha256": _sha256(candidate_dataset),
            "joined_record_count": changed,
            "key_contract": "(instance_id, candidate_id, split)",
            "projection_key_contract": "projection::<instance_id>/<candidate_id:03d>",
        }
        joined_dataset = {"metadata": metadata, "records": joined_records}
        joined_path = output_dir / "candidate_dataset.json"
        joined_path.write_text(json.dumps(joined_dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        report["status"] = "passed"
        report["summary"] = {
            "candidate_record_count": len(records),
            "rgb_report_row_count": len(rgb_rows),
            "joined_record_count": changed,
            "candidate_key_count": len(candidate_keys),
            "projection_npz_key_count": len(expected_npz_keys),
            "output_candidate_dataset": str(joined_path.resolve()),
            "output_realized_latents": str(copied_npz.resolve()),
        }
        report["source_hashes"] = metadata["realized_latent_join"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report["errors"].append(str(exc))

    (output_dir / "join_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", required=True, type=Path)
    parser.add_argument("--rgb-report", required=True, type=Path)
    parser.add_argument("--realized-npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = join_artifact(args.candidate_dataset, args.rgb_report, args.realized_npz, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
