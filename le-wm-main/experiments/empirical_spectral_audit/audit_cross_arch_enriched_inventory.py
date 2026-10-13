"""Inventory enriched exports and fail closed on incomplete architectures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def classify(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("record_schema") == "enriched_learned_model.v1":
        rows = payload.get("enriched_records", [])
        return {
            "path": str(path),
            "status": "canonical_enriched",
            "architecture": payload.get("architecture"),
            "environment": payload.get("environment"),
            "seed": payload.get("seed"),
            "training_stage": payload.get("training_stage"),
            "latent_dimension": payload.get("latent_dimension"),
            "action_distribution": payload.get("action_distribution"),
            "row_count": len(rows),
        }
    if isinstance(payload.get("enriched_records"), list):
        return {
            "path": str(path),
            "status": "legacy_enriched_requires_migration",
            "row_count": len(payload["enriched_records"]),
        }
    if "operator_fit_vectors" in payload:
        return {"path": str(path), "status": "incomplete_vector_only_export"}
    return {"path": str(path), "status": "unrecognized"}


def audit(paths: list[Path]) -> dict[str, Any]:
    entries = [classify(path) for path in paths]
    canonical = [entry for entry in entries if entry["status"] == "canonical_enriched"]
    arch_by_environment: dict[str, set[str]] = {}
    for entry in canonical:
        arch_by_environment.setdefault(str(entry["environment"]), set()).add(str(entry["architecture"]))
    matched_environment_candidates = {
        environment: sorted(architectures)
        for environment, architectures in sorted(arch_by_environment.items())
        if len(architectures) >= 2
    }
    return {
        "protocol": {
            "canonical_schema": "enriched_learned_model.v1",
            "legacy_vector_only_policy": "incomplete_export; never upgraded without rerunning the model",
            "matched_stratum_requirement": "same environment, training stage, latent dimension, action distribution, prediction quality, and predictor spectrum; at least two architectures and two seeds",
        },
        "inventory": entries,
        "summary": {
            "total_exports": len(entries),
            "canonical_enriched_exports": len(canonical),
            "incomplete_vector_only_exports": sum(entry["status"] == "incomplete_vector_only_export" for entry in entries),
            "architectures": sorted({str(entry["architecture"]) for entry in canonical}),
            "environments": sorted({str(entry["environment"]) for entry in canonical}),
            "cross_architecture_environment_candidates": matched_environment_candidates,
            "status": "strict_matched_stratum_pending" if matched_environment_candidates else "gated_no_cross_architecture_enriched_stratum",
        },
        "claim_boundary": (
            "Canonical enriched exports are present, but no cross-architecture matched learned-model stratum is claimed "
            "unless the inventory contains at least two independently exported architectures in the same environment and "
            "the downstream strict matcher identifies the quality/spectrum-matched pool."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--path", action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.rglob("*.json")) if args.root is not None else []
    paths.extend(args.path or [])
    if not paths:
        parser.error("provide --root or --path")
    report = audit(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
