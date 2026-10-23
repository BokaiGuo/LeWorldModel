"""Audit root-disjoint decision aliasing on one completed PointMaze panel.

Calibration roots define a fit-only representation neighborhood scale.  Their
realized candidate-contrast vectors are then compared with post-execution
outcome roots under the same held candidate bank.  This is an empirical
mechanism audit after outcome execution, never a deployment-time certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.decision_aliasing import (
    calibrate_representation_radius,
    empirical_decision_aliasing_audit,
)


SCHEMA = "pointmaze_decision_aliasing_panel_audit.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_map(payload: dict[str, Any], *, nested_key: str | None = None) -> dict[str, dict[str, Any]]:
    value: Any = payload if nested_key is None else payload.get(nested_key)
    if not isinstance(value, dict) or not value:
        raise ValueError("label payload must contain a non-empty identity-keyed object")
    result = {str(key): dict(row) for key, row in value.items() if isinstance(row, dict)}
    if len(result) != len(value):
        raise ValueError("every label row must be an object")
    return result


def root_table(
    records: list[dict[str, Any]], labels: dict[str, dict[str, Any]], *, representation_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], str]:
    """Convert one root-disjoint split to history representations and contrasts."""

    if representation_key not in {"source_history_model", "source_latent_model"}:
        raise ValueError("representation_key must be source_history_model or source_latent_model")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        root, candidate = str(row.get("instance_id", "")), str(row.get("candidate_id", ""))
        if not root or not candidate:
            raise ValueError("every context record needs instance_id and candidate_id")
        grouped.setdefault(root, []).append(row)
    if not grouped:
        raise ValueError("no context records supplied")
    representations, contrasts, roots = [], [], []
    signature: tuple[str, ...] | None = None
    bank_hash: str | None = None
    for root in sorted(grouped):
        candidates = sorted(grouped[root], key=lambda row: str(row["candidate_id"]))
        candidate_ids = tuple(str(row["candidate_id"]) for row in candidates)
        if len(candidate_ids) < 2 or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("each root needs at least two unique candidates")
        if signature is None:
            signature = candidate_ids
        elif signature != candidate_ids:
            raise ValueError("candidate identities differ across roots")
        current_hashes = {str(row.get("candidate_bank_hash", "")) for row in candidates}
        if len(current_hashes) != 1 or not next(iter(current_hashes)):
            raise ValueError("each root needs one declared candidate_bank_hash")
        current_hash = next(iter(current_hashes))
        if bank_hash is None:
            bank_hash = current_hash
        elif bank_hash != current_hash:
            raise ValueError("candidate bank hash differs across roots")
        history = np.asarray(candidates[0].get(representation_key), dtype=np.float64).reshape(-1)
        if not len(history) or not np.isfinite(history).all():
            raise ValueError(f"{root}: missing finite {representation_key}")
        if any(not np.array_equal(history, np.asarray(row.get(representation_key), dtype=np.float64).reshape(-1)) for row in candidates[1:]):
            raise ValueError(f"{root}: representation must be candidate-invariant")
        costs: list[float] = []
        for row in candidates:
            key = f"{root}/{row['candidate_id']}"
            label = labels.get(key)
            if not isinstance(label, dict) or not np.isfinite(float(label.get("real_task_cost", np.nan))):
                raise ValueError(f"{root}: missing finite realized label for candidate {row['candidate_id']}")
            costs.append(float(label["real_task_cost"]))
        cost = np.asarray(costs, dtype=np.float64)
        representations.append(history)
        contrasts.append(cost - cost.mean())
        roots.append(root)
    assert signature is not None and bank_hash is not None
    return np.stack(representations), np.stack(contrasts), np.asarray(roots, dtype=str), signature, bank_hash


def run(*, context: Path, calibration_labels: Path, outcome_report: Path, output: Path,
        outcome_labels: Path | None = None,
        representation_key: str = "source_history_model", radius_quantile: float = 0.25,
        contrast_gap_threshold: float = 0.05, max_reported_pairs: int = 64) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    context_payload = _load(context)
    report_payload = _load(outcome_report)
    context_records = [dict(row) for row in context_payload.get("records", [])]
    calibration_records = [row for row in context_records if str(row.get("split")) == "calibration"]
    outcome_records = [row for row in context_records if str(row.get("split")) == "outcome"]
    calibration = _label_map(_load(calibration_labels))
    # Current reports embed outcome labels.  The explicit sidecar path exists
    # only to recover older, completed reports that predate that serialization
    # contract; neither form is ever a deployment-time input.
    if outcome_labels is None:
        outcome = _label_map(report_payload, nested_key="post_execution_outcome_labels")
        outcome_label_provenance = {
            "kind": "embedded_report_field",
            "path": str(outcome_report.resolve()),
            "sha256": _sha256(outcome_report),
        }
    else:
        sidecar_payload = _load(outcome_labels)
        outcome = _label_map(sidecar_payload, nested_key="labels")
        outcome_label_provenance = {
            "kind": "post_execution_label_sidecar",
            "path": str(outcome_labels.resolve()),
            "sha256": _sha256(outcome_labels),
        }
    cal_rep, cal_contrast, cal_roots, cal_signature, cal_bank = root_table(calibration_records, calibration, representation_key=representation_key)
    out_rep, out_contrast, out_roots, out_signature, out_bank = root_table(outcome_records, outcome, representation_key=representation_key)
    if cal_signature != out_signature or cal_bank != out_bank:
        raise ValueError("calibration and outcome tables must share candidate identity and bank hash")
    radius = calibrate_representation_radius(cal_rep, cal_roots, quantile=radius_quantile)
    audit = empirical_decision_aliasing_audit(
        cal_rep, cal_contrast, cal_roots, out_rep, out_contrast, out_roots,
        representation_radius=float(radius["representation_radius"]),
        contrast_gap_threshold=float(contrast_gap_threshold),
        max_reported_pairs=int(max_reported_pairs),
    )
    result = {
        "schema": SCHEMA,
        "status": "completed_post_execution_mechanism_audit",
        "representation_key": representation_key,
        "candidate_signature": list(cal_signature), "candidate_bank_hash": cal_bank,
        "radius_calibration": radius, "audit": audit,
        "provenance": {
            "context": str(context.resolve()), "context_sha256": _sha256(context),
            "calibration_labels": str(calibration_labels.resolve()), "calibration_labels_sha256": _sha256(calibration_labels),
            "outcome_report": str(outcome_report.resolve()), "outcome_report_sha256": _sha256(outcome_report),
            "outcome_labels": outcome_label_provenance,
        },
        "claim_boundary": "post-execution root-disjoint PointMaze mechanism audit; not a prospective repair rule, causal-identifiability theorem, or deployment certificate.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--calibration-labels", required=True, type=Path)
    parser.add_argument("--outcome-report", required=True, type=Path)
    parser.add_argument("--outcome-labels", type=Path, default=None,
                        help="post-execution label sidecar for a legacy report without embedded labels")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--representation-key", choices=("source_history_model", "source_latent_model"), default="source_history_model")
    parser.add_argument("--radius-quantile", type=float, default=0.25)
    parser.add_argument("--contrast-gap-threshold", type=float, default=0.05)
    parser.add_argument("--max-reported-pairs", type=int, default=64,
                        help="must cover every outcome root when used for disagreement overlap")
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"status": result["status"], "audit": result["audit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
