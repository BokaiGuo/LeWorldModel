"""Shared schema and validation for cross-architecture learned-model exports.

The temporal evaluators historically emitted architecture-specific JSON fields.
This module defines the only export contract accepted by the matched learned
model audit.  It is intentionally strict: an old vector-only export can be
listed in an inventory, but it cannot be upgraded to an enriched export by
filling missing state, action, or source-latent fields with placeholders.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "enriched_learned_model.v1"
REQUIRED_ROW_FIELDS = (
    "sample_id",
    "episode_id",
    "source_frame",
    "source_latent",
    "source_state_raw",
    "source_proprio_raw",
    "action_blocks_raw",
    "action_blocks_normalized",
    "latent_by_horizon",
)

# Counterfactual evaluator exports may additionally record the action blocks
# actually supplied to the predictor.  ``action_blocks_*`` always describe the
# realised trajectory that supplies the physical target, so they cannot by
# themselves establish a zero/shuffled predictor intervention.
PREDICTOR_ACTION_BLOCKS_FIELD = "predictor_action_blocks_normalized"


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is missing")
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be non-empty and finite")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {result.ndim}")
    return result


def validate_row(row: dict[str, Any], horizons: Iterable[int] | None = None) -> dict[str, Any]:
    """Validate one episode/window record without changing its values."""
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"enriched row {row.get('sample_id', '<unknown>')} lacks {missing}")
    if not isinstance(row["sample_id"], str) or not row["sample_id"]:
        raise ValueError("sample_id must be a non-empty string")
    int(row["episode_id"])
    int(row["source_frame"])
    source = _array(row["source_latent"], "source_latent", ndim=1)
    _array(row["source_state_raw"], "source_state_raw", ndim=1)
    _array(row["source_proprio_raw"], "source_proprio_raw", ndim=1)
    raw_actions = _array(row["action_blocks_raw"], "action_blocks_raw", ndim=2)
    normalized_actions = _array(row["action_blocks_normalized"], "action_blocks_normalized", ndim=2)
    if raw_actions.shape != normalized_actions.shape:
        raise ValueError(f"{row['sample_id']} raw/normalized action shape mismatch")
    if PREDICTOR_ACTION_BLOCKS_FIELD in row:
        predictor_actions = _array(
            row[PREDICTOR_ACTION_BLOCKS_FIELD], PREDICTOR_ACTION_BLOCKS_FIELD, ndim=2
        )
        if predictor_actions.shape != normalized_actions.shape:
            raise ValueError(f"{row['sample_id']} predictor/realized action shape mismatch")
    if "predictor_action_mode" in row and (
        not isinstance(row["predictor_action_mode"], str) or not row["predictor_action_mode"]
    ):
        raise ValueError(f"{row['sample_id']} predictor_action_mode must be a non-empty string")
    by_horizon = row["latent_by_horizon"]
    if not isinstance(by_horizon, dict) or not by_horizon:
        raise ValueError(f"{row['sample_id']} has no latent_by_horizon")
    expected = None if horizons is None else {str(int(h)) for h in horizons}
    actual = set(by_horizon)
    if expected is not None and actual != expected:
        raise ValueError(f"{row['sample_id']} horizon mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    for horizon, item in by_horizon.items():
        if not isinstance(item, dict):
            raise ValueError(f"{row['sample_id']} horizon {horizon} is not an object")
        for field in ("predicted", "target", "target_state_raw"):
            _array(item.get(field), f"{row['sample_id']}/{horizon}/{field}", ndim=1)
        predicted = np.asarray(item["predicted"], dtype=np.float64)
        target = np.asarray(item["target"], dtype=np.float64)
        if predicted.shape != target.shape or predicted.shape != source.shape:
            raise ValueError(
                f"{row['sample_id']}/{horizon} latent shape mismatch: "
                f"source={source.shape}, predicted={predicted.shape}, target={target.shape}"
            )
    return row


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a complete export and return it unchanged."""
    if payload.get("record_schema") != SCHEMA:
        raise ValueError(f"expected record_schema={SCHEMA!r}, got {payload.get('record_schema')!r}")
    required_metadata = (
        "architecture", "checkpoint", "environment", "seed", "training_stage",
        "latent_dimension", "action_distribution", "split", "horizons",
        "study_scope", "claim_boundary", "runtime",
    )
    missing = [field for field in required_metadata if field not in payload]
    if missing:
        raise ValueError(f"enriched payload lacks metadata: {missing}")
    horizons = [int(item) for item in payload["horizons"]]
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be a non-empty list of unique integers")
    if int(payload["latent_dimension"]) <= 0:
        raise ValueError("latent_dimension must be positive")
    rows = payload.get("enriched_records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("enriched_records must be a non-empty list")
    sample_ids: set[str] = set()
    for row in rows:
        validate_row(row, horizons)
        if row["sample_id"] in sample_ids:
            raise ValueError(f"duplicate sample_id {row['sample_id']}")
        sample_ids.add(row["sample_id"])
        source_dim = len(row["source_latent"])
        if source_dim != int(payload["latent_dimension"]):
            raise ValueError(
                f"{row['sample_id']} latent dimension {source_dim} != payload {payload['latent_dimension']}"
            )
    return payload


def make_payload(
    *,
    architecture: str,
    checkpoint: str,
    environment: str,
    seed: int | str,
    training_stage: int,
    latent_dimension: int,
    action_distribution: str,
    split: dict[str, Any],
    horizons: Iterable[int],
    enriched_records: list[dict[str, Any]],
    checkpoint_sha256: str | None = None,
    dataset_sha256: str | None = None,
    protocol: dict[str, Any] | None = None,
    study_scope: str = "frozen temporal latent diagnostic",
    claim_boundary: str = (
        "Frozen-checkpoint temporal evidence only; this export does not establish "
        "online planning, causal action effects, or a universal world model."
    ),
    runtime: str = "repo_local_dino_checkpoint_compat.v1",
) -> dict[str, Any]:
    """Create and validate a canonical export payload."""
    payload = {
        "record_schema": SCHEMA,
        "architecture": str(architecture),
        "checkpoint": str(checkpoint),
        "environment": str(environment),
        "seed": str(seed),
        "training_stage": int(training_stage),
        "latent_dimension": int(latent_dimension),
        "action_distribution": str(action_distribution),
        "split": split,
        "horizons": [int(h) for h in horizons],
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
        "study_scope": str(study_scope),
        "claim_boundary": str(claim_boundary),
        "runtime": str(runtime),
        "protocol": protocol or {},
        "enriched_records": enriched_records,
    }
    return validate_payload(payload)


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return validate_payload(payload)

