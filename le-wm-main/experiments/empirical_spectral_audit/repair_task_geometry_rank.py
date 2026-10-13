"""Recompute learned-record geometry after fixing the full-rank diagnostic bug.

The original panel used a full latent rank for principal-angle alignment.  That
quantity is identically trivial in a full-rank latent space, so this repair
recomputes only the train-split geometry with the predeclared low-rank task
subspace.  Held-out outcomes and all checkpoint selection fields are copied
verbatim; no outcome value is read to choose a rank or a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from experiments.empirical_spectral_audit.build_learned_model_selection_records import (
    TASK_GEOMETRY_RANK,
    _fit_metric,
    _fit_transfer_operator,
    _ridge,
    _rows,
)
from experiments.empirical_spectral_audit.enriched_record_schema import load_payload
from experiments.task_metric_diagnostics import metric_geometry


def repair_record(source: Path, destination: Path) -> dict[str, object]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    record = copy.deepcopy(payload["record"])
    fit_path = Path(record["provenance"]["fit_export"])
    if not fit_path.exists():
        # Older records were written from the outer workspace and stored the
        # repository segment (``le-wm-main/...``) as a relative path.
        fit_path = Path(__file__).resolve().parents[3] / fit_path
    if not fit_path.exists():
        raise FileNotFoundError(f"fit export referenced by {source} is missing: {fit_path}")
    fit_payload = load_payload(fit_path)
    source_latent, predicted, _target, state_delta = _rows(fit_payload, 1)
    centered_source = source_latent - source_latent.mean(0, keepdims=True)
    covariance = centered_source.T @ centered_source / max(len(source_latent) - 1, 1)
    predictor = _ridge(
        centered_source,
        predicted - predicted.mean(0, keepdims=True),
        1e-3,
    )
    metric = _fit_metric(source_latent, state_delta, 1e-3)
    rank = min(TASK_GEOMETRY_RANK, source_latent.shape[1])
    geometry = metric_geometry(
        covariance,
        predictor,
        metric,
        rank=rank,
        horizons=range(1, 5),
    )

    # Preserve the existing transfer operator and all outcome evidence.  The
    # repair changes only the geometry fields whose rank was previously wrong.
    for name, value in geometry.items():
        if name == "reachability_score" or name == "real_predicted_transition_transfer":
            continue
        record["diagnostics"][name] = float(value)
    record.setdefault("protocol", {})["task_geometry_subspace_rank"] = int(rank)
    record.setdefault("provenance", {})["geometry_repair"] = {
        "source_record": str(source),
        "fit_split_only": True,
        "outcome_used_for_repair": False,
        "rank": int(rank),
        "reason": "full_rank_principal_angle_was_trivial",
    }
    repaired = {
        "record_schema": payload["record_schema"],
        "protocol": payload.get("protocol", {}),
        "record": record,
        "claim_boundary": payload.get("claim_boundary", ""),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(repaired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "source": str(source),
        "destination": str(destination),
        "rank": int(rank),
        "fit_export": str(fit_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(args.input_root.rglob("checkpoint_record.json"))
    if not sources:
        raise SystemExit(f"no checkpoint records under {args.input_root}")
    reports = []
    for source in sources:
        relative = source.relative_to(args.input_root)
        reports.append(repair_record(source, args.output_root / relative))
    manifest = {
        "schema": "learned_record_geometry_repair.v1",
        "source_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "record_count": len(reports),
        "task_geometry_subspace_rank": TASK_GEOMETRY_RANK,
        "outcome_used_for_repair": False,
        "records": reports,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in manifest if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
