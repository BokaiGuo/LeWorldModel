"""Audit whether a vector-only DINO temporal artifact can join the enriched panel.

The old DINO temporal artifact stores vectors in evaluator order but no sample
IDs, actions, source states, or target states.  This script performs only a
read-only consistency audit.  It never fabricates missing fields and never
turns vector-only prediction quality into action-conditioned planning evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


HORIZONS = ("1", "2", "4")
REQUIRED_ENRICHED_FIELDS = {
    "sample_id",
    "episode_id",
    "source_frame",
    "source_state_raw",
    "source_latent",
    "action_blocks_raw",
    "latent_by_horizon",
}


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _vector_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = left - right
    return {
        "rows": int(left.shape[0]),
        "dimensions": int(left.shape[1]),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def audit(vector_path: Path, enriched_path: Path) -> dict[str, Any]:
    vector = json.loads(vector_path.read_text())
    enriched = json.loads(enriched_path.read_text())
    protocol = vector.get("protocol", {})
    records = list(enriched.get("enriched_records", []))
    vector_episode_ids = [int(x) for x in protocol.get("episodes_used", [])]
    enriched_episode_ids = [int(row["episode_id"]) for row in records]
    common = min(len(vector_episode_ids), len(records))
    prefix_episode_match = vector_episode_ids[:common] == enriched_episode_ids[:common]
    vector_fields = set(vector.get("operator_fit_vectors", {}))
    vector_has_ids = bool(vector.get("operator_fit_vectors", {}).get("episode_ids"))
    missing_enriched_fields = sorted(REQUIRED_ENRICHED_FIELDS - set(records[0])) if records else sorted(REQUIRED_ENRICHED_FIELDS)

    comparisons: dict[str, Any] = {}
    for horizon in HORIZONS:
        pred = _array(vector["operator_fit_vectors"]["predicted"][horizon])
        target = _array(vector["operator_fit_vectors"]["target"][horizon])
        enriched_pred = _array([row["latent_by_horizon"][horizon]["predicted"] for row in records])
        enriched_target = _array([row["latent_by_horizon"][horizon]["target"] for row in records])
        n = min(common, len(pred), len(target), len(enriched_pred), len(enriched_target))
        comparisons[horizon] = {
            "prefix_rows": int(n),
            "vector_vs_enriched_predicted": _vector_metrics(pred[:n], enriched_pred[:n]),
            "vector_vs_enriched_target": _vector_metrics(target[:n], enriched_target[:n]),
            "vector_only_prediction_mse": float(np.mean((pred - target) ** 2)),
        }

    metadata_compatible = (
        protocol.get("horizons") == [int(h) for h in HORIZONS]
        and int(protocol.get("context_length", -1)) == 3
        and int(protocol.get("frame_stride", -1)) == 5
        and str(protocol.get("action_mode")) == "observed"
        and prefix_episode_match
    )
    formal_status = "available" if metadata_compatible and not missing_enriched_fields and vector_has_ids else "rejected_fail_closed"
    return {
        "schema": "dino_vector_artifact_join_audit.v1",
        "inputs": {
            "vector_artifact": str(vector_path.resolve()),
            "enriched_export": str(enriched_path.resolve()),
        },
        "metadata": {
            "vector_horizons": protocol.get("horizons"),
            "vector_context_length": protocol.get("context_length"),
            "vector_frame_stride": protocol.get("frame_stride"),
            "vector_action_mode": protocol.get("action_mode"),
            "vector_has_sample_ids": vector_has_ids,
            "vector_operator_fields": sorted(vector_fields),
            "vector_episode_count": len(vector_episode_ids),
            "enriched_episode_count": len(enriched_episode_ids),
            "common_prefix_count": common,
            "prefix_episode_match": prefix_episode_match,
            "missing_enriched_fields": missing_enriched_fields,
        },
        "comparisons": comparisons,
        "decision": {
            "formal_action_conditioned_join": formal_status,
            "vector_only_joined_diagnostic": "available" if metadata_compatible and common >= 8 else "rejected_fail_closed",
            "reason": (
                "The vector artifact has no episode/sample IDs and no action/state fields; it can support only a vector-only temporal diagnostic."
                if formal_status != "available"
                else "All required join metadata and enriched fields are present."
            ),
            "claim_boundary": "No action-conditioned adapter, task readout, or planning claim may consume this vector-only artifact.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--enriched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.vector, args.enriched)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
