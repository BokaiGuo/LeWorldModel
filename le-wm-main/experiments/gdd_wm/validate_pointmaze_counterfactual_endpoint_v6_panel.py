"""Validate an immutable V6 panel before resuming its CUDA endpoint runner.

The V6 builder seals a fresh 128/192/64 root panel before the endpoint model
is fitted.  If a host job stops after this expensive build, rerunning the
builder would silently change the experimental object or overwrite evidence.
This validator binds the existing panel to its static preflight ledger and
artifact hashes, so a resume command can execute only the remaining frozen
endpoint-control stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "pointmaze_counterfactual_endpoint_v6_resume_panel_validation.v1"
PANEL_SCHEMA = "pointmaze_temporal_ood_v6_matched_panel.v3"
LEDGER_SCHEMA = "pointmaze_counterfactual_endpoint_v6_static_preflight.v1"
ARTIFACTS = (
    "context",
    "source_dataset",
    "source_arrays",
    "calibration_outcomes",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(panel: Path, name: str) -> Path:
    return panel / {
        "context": "temporal_ood_v6_context.json",
        "source_dataset": "temporal_ood_v6_source_dataset.json",
        "source_arrays": "temporal_ood_v6_source_arrays.npz",
        "calibration_outcomes": "temporal_ood_v6_calibration_outcomes.json",
    }[name]


def validate(*, panel: Path, frozen_preflight: Path) -> dict[str, Any]:
    """Verify one sealed panel without model fitting, CUDA, or outcome access."""

    panel = panel.resolve()
    frozen_preflight = frozen_preflight.resolve()
    if not panel.is_dir():
        raise FileNotFoundError(f"V6 panel directory does not exist: {panel}")
    if not frozen_preflight.is_file():
        raise FileNotFoundError(f"frozen V6 preflight does not exist: {frozen_preflight}")
    ledger = _load(frozen_preflight)
    if ledger.get("schema") != LEDGER_SCHEMA or ledger.get("status") != "passed_static_preflight":
        raise ValueError("resume requires a passed V6 static preflight ledger")
    expected_counts = ledger.get("root_counts")
    if not isinstance(expected_counts, dict) or set(expected_counts) != {"train", "calibration", "outcome"}:
        raise ValueError("static preflight must declare train/calibration/outcome root counts")

    manifest_path = panel / "temporal_ood_v6_manifest.json"
    manifest = _load(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("V6 manifest must bind panel artifacts")
    artifact_checks: dict[str, bool] = {}
    for name in ARTIFACTS:
        descriptor = artifacts.get(name)
        path = _artifact_path(panel, name)
        if not isinstance(descriptor, dict) or not path.is_file():
            raise ValueError(f"V6 manifest is missing required {name} artifact")
        declared_path = Path(str(descriptor.get("path", ""))).resolve()
        declared_hash = descriptor.get("sha256")
        matches = declared_path == path.resolve() and isinstance(declared_hash, str) and _sha256(path) == declared_hash
        artifact_checks[name] = bool(matches)
    if not all(artifact_checks.values()):
        raise ValueError(f"V6 panel artifact hash/path mismatch: {artifact_checks}")
    outcome_descriptor = artifacts.get("outcome_labels")
    outcome_label_path = panel / "temporal_ood_v6_outcome_labels.json"
    if not isinstance(outcome_descriptor, dict) or outcome_descriptor.get("materialized") is not False:
        raise ValueError("V6 resume requires sealed outcome labels in manifest")
    if outcome_label_path.exists():
        raise ValueError("V6 outcome labels are already materialized; refuse a fresh endpoint-control resume")

    context = _load(_artifact_path(panel, "context"))
    metadata = context.get("metadata")
    rows = context.get("records")
    if not isinstance(metadata, dict) or not isinstance(rows, list) or metadata.get("schema") != PANEL_SCHEMA:
        raise ValueError("V6 context must retain the current panel schema and records")
    binding = metadata.get("frozen_preflight_ledger")
    if not isinstance(binding, dict):
        raise ValueError("V6 context is missing frozen-preflight provenance")
    binding_ok = (
        binding.get("schema") == LEDGER_SCHEMA
        and binding.get("status") == "passed_static_preflight"
        and binding.get("sha256") == _sha256(frozen_preflight)
        and binding.get("root_counts") == expected_counts
        and Path(str(binding.get("path", ""))).resolve() == frozen_preflight
    )
    if not binding_ok:
        raise ValueError("V6 context no longer matches its frozen static-preflight ledger")
    roots_by_split: dict[str, set[str]] = {split: set() for split in expected_counts}
    outcome_leak = False
    for row in rows:
        if not isinstance(row, dict) or str(row.get("split")) not in roots_by_split:
            raise ValueError("V6 context contains an invalid split row")
        split = str(row["split"])
        roots_by_split[split].add(str(row.get("instance_id", "")))
        if split == "outcome" and any(row.get(key) is not None for key in (
            "realized_terminal_state", "real_task_cost", "realized_terminal_state_ref",
        )):
            outcome_leak = True
    actual_counts = {split: len(roots) for split, roots in roots_by_split.items()}
    if any(not roots for roots in roots_by_split.values()) or actual_counts != expected_counts:
        raise ValueError(f"V6 context root counts differ from frozen ledger: actual={actual_counts}, expected={expected_counts}")
    if outcome_leak:
        raise ValueError("V6 outcome labels leaked into sealed context")
    if any(roots_by_split[left] & roots_by_split[right] for left in roots_by_split for right in roots_by_split if left < right):
        raise ValueError("V6 context root split overlap is forbidden")

    source = _load(_artifact_path(panel, "source_dataset"))
    source_rows = source.get("records")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("V6 source dataset requires non-empty records")
    source_splits = Counter(str(row.get("split")) for row in source_rows if isinstance(row, dict))
    source_roots = {str(row.get("instance_id", "")) for row in source_rows if isinstance(row, dict)}
    if source_splits != Counter({"train": len(source_rows)}) or source_roots != roots_by_split["train"]:
        raise ValueError("V6 source dataset must contain exactly the sealed train roots")
    labels = _load(_artifact_path(panel, "calibration_outcomes"))
    expected_label_count = len(roots_by_split["calibration"]) * (len(rows) // sum(actual_counts.values()))
    if len(labels) != expected_label_count:
        raise ValueError("V6 calibration labels do not cover the declared fixed candidate bank")

    return {
        "schema": SCHEMA,
        "status": "passed_immutable_v6_resume_panel_validation",
        "panel": str(panel),
        "frozen_preflight": {"path": str(frozen_preflight), "sha256": _sha256(frozen_preflight)},
        "artifact_hashes_verified": artifact_checks,
        "root_counts": actual_counts,
        "source_record_count": len(source_rows),
        "calibration_label_count": len(labels),
        "outcome_labels_sealed": True,
        "claim_boundary": "Immutable-panel validation only; this is not a CUDA run, endpoint-model result, or authorization claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--frozen-preflight", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(panel=args.panel, frozen_preflight=args.frozen_preflight), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
