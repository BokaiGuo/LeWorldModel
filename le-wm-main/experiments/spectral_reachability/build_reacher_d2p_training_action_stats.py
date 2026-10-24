"""Build the Reacher action-normalisation receipt implied by training.

LeWM normalises each physical two-dimensional action *before* the HDF5
dataset reshapes five adjacent controls into the ten-dimensional action token
seen by the checkpoint.  A live calibration panel must therefore not estimate
one independent mean/std pair for each of those five packet positions: that
changes the coordinate system consumed by the frozen action encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


SCHEMA = "reacher_d2p_training_action_stats.v1"
ENVIRONMENT = "swm/ReacherDMControl-v0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_packet_statistics(actions: np.ndarray, *, action_block: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Match ``get_column_normalizer`` then tile its physical action stats.

    ``torch.std`` is deliberate: the original training entry point computes
    this (unbiased) statistic on the raw two-wide HDF5 column before clips are
    packetised.
    """

    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 2 or action_block < 1:
        raise ValueError("expected finite-or-NaN raw Reacher actions with shape [rows, 2]")
    finite = values[np.isfinite(values).all(axis=1)]
    if finite.shape[0] < 2:
        raise ValueError("need at least two finite raw action rows")
    raw = torch.from_numpy(np.ascontiguousarray(finite))
    mean = raw.mean(dim=0)
    std = raw.std(dim=0)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0.0):
        raise ValueError("invalid Reacher training action statistics")
    return (
        mean.repeat(action_block).cpu().numpy(),
        std.repeat(action_block).cpu().numpy(),
    )


def validate_training_semantic_receipt(path: Path, *, environment: str = ENVIRONMENT) -> dict[str, Any]:
    """Fail closed unless a runtime receipt preserves the training coordinate system."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("status") != "training_semantics_validated":
        raise ValueError("Reacher D2P requires a training-semantics action-stat receipt")
    rows = [row for row in payload.get("reports", []) if row.get("environment") == environment]
    if len(rows) != 1:
        raise ValueError("training-semantics receipt must contain exactly one Reacher report")
    row = rows[0]
    if row.get("raw_action_dim") != 2 or row.get("action_block") != 5 or not row.get("all_packet_slots_share_physical_statistics"):
        raise ValueError("training-semantics receipt has an incompatible Reacher packet contract")
    mean = np.asarray(row.get("calibration_action_mean"), dtype=np.float64)
    std = np.asarray(row.get("calibration_action_std"), dtype=np.float64)
    if mean.shape != (10,) or std.shape != (10,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
        raise ValueError("training-semantics receipt has invalid Reacher action statistics")
    if not np.allclose(mean.reshape(5, 2), mean.reshape(5, 2)[0], rtol=0.0, atol=0.0):
        raise ValueError("training-semantics receipt has position-specific action means")
    if not np.allclose(std.reshape(5, 2), std.reshape(5, 2)[0], rtol=0.0, atol=0.0):
        raise ValueError("training-semantics receipt has position-specific action stds")
    return row


def build(*, dataset: Path, source_manifest: Path, output: Path, action_block: int = 5) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable action-stat receipt: {output}")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("status") != "archive_and_hdf5_validated":
        raise ValueError("official Reacher source manifest is not validated")
    if Path(source.get("dataset", "")).resolve() != dataset.resolve():
        raise ValueError("action-stat dataset disagrees with the validated official-source manifest")
    with h5py.File(dataset, "r", swmr=True) as handle:
        if "action" not in handle:
            raise ValueError("official Reacher HDF5 has no action column")
        mean, std = training_packet_statistics(handle["action"][:], action_block=action_block)
    payload = {
        "schema": SCHEMA,
        "status": "training_semantics_validated",
        "source": {
            "official_source_manifest": str(source_manifest.resolve()),
            "official_source_manifest_sha256": _sha256(source_manifest),
            "dataset": str(dataset.resolve()),
            "dataset_sha256": source["dataset_sha256"],
        },
        "reports": [{
            "environment": ENVIRONMENT,
            "status": "completed",
            "raw_action_dim": 2,
            "action_block": action_block,
            "calibration_action_mean": mean.tolist(),
            "calibration_action_std": std.tolist(),
            "normalization_contract": "raw two-dimensional training actions are z-scored before five-step time-major packetisation",
            "all_packet_slots_share_physical_statistics": True,
        }],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-block", type=int, default=5)
    args = parser.parse_args()
    result = build(dataset=args.dataset, source_manifest=args.source_manifest, output=args.output, action_block=args.action_block)
    print(json.dumps({"output": str(args.output), "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
