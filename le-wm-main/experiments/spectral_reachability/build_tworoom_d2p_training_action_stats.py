"""Create a source-locked TwoRoom action-statistics receipt for frozen D2P.

The released LeWM trainer normalised an individual two-dimensional action
before ``HDF5Dataset`` packed five successive controls into the action token.
Consequently, a planning runtime must use the same physical mean/std in every
time slot.  A ten-dimensional calibration fitted after packetisation is a
different coordinate system and is rejected here.
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


SCHEMA = "tworoom_d2p_training_action_stats.v1"
ENVIRONMENT = "swm/TwoRoom-v1"
RAW_ACTION_DIM = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_packet_statistics(actions: np.ndarray, *, action_block: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return the trainer's raw-action z-score stats tiled time-major.

    ``torch.std`` intentionally retains its default unbiased estimator because
    ``utils.get_column_normalizer`` used that exact operation before action
    packetisation.  ``repeat`` on a two-wide vector yields
    ``[mean_x, mean_y, mean_x, mean_y, ...]``.
    """

    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != RAW_ACTION_DIM or action_block < 1:
        raise ValueError("expected finite-or-NaN raw TwoRoom actions with shape [rows, 2]")
    finite = values[np.isfinite(values).all(axis=1)]
    if finite.shape[0] < 2:
        raise ValueError("need at least two finite raw action rows")
    physical = torch.from_numpy(np.ascontiguousarray(finite))
    mean = physical.mean(dim=0)
    std = physical.std(dim=0)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0.0):
        raise ValueError("invalid TwoRoom training action statistics")
    return mean.repeat(action_block).cpu().numpy(), std.repeat(action_block).cpu().numpy()


def validate_training_semantic_receipt(path: Path, *, environment: str = ENVIRONMENT) -> dict[str, Any]:
    """Fail closed on post-packet or position-specific action statistics."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("status") != "training_semantics_validated":
        raise ValueError("TwoRoom D2P requires a training-semantics action-stat receipt")
    rows = [row for row in payload.get("reports", []) if row.get("environment") == environment]
    if len(rows) != 1:
        raise ValueError("training-semantics receipt must contain exactly one TwoRoom report")
    row = rows[0]
    if row.get("raw_action_dim") != RAW_ACTION_DIM or row.get("action_block") != 5:
        raise ValueError("training-semantics receipt has an incompatible TwoRoom packet contract")
    if not row.get("all_packet_slots_share_physical_statistics"):
        raise ValueError("training-semantics receipt does not certify shared TwoRoom packet statistics")
    mean = np.asarray(row.get("calibration_action_mean"), dtype=np.float64)
    std = np.asarray(row.get("calibration_action_std"), dtype=np.float64)
    if mean.shape != (10,) or std.shape != (10,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
        raise ValueError("training-semantics receipt has invalid TwoRoom action statistics")
    if not np.array_equal(mean.reshape(5, 2), np.broadcast_to(mean[:2], (5, 2))):
        raise ValueError("training-semantics receipt has position-specific TwoRoom action means")
    if not np.array_equal(std.reshape(5, 2), np.broadcast_to(std[:2], (5, 2))):
        raise ValueError("training-semantics receipt has position-specific TwoRoom action stds")
    return row


def build(*, dataset: Path, output: Path, action_block: int = 5) -> dict[str, Any]:
    """Bind an immutable runtime receipt to the exact HDF5 training source."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable action-stat receipt: {output}")
    if not dataset.is_file():
        raise FileNotFoundError(f"TwoRoom training HDF5 not found: {dataset}")
    with h5py.File(dataset, "r", swmr=True) as handle:
        if "action" not in handle:
            raise ValueError("TwoRoom HDF5 has no action column")
        mean, std = training_packet_statistics(handle["action"][:], action_block=action_block)
    payload = {
        "schema": SCHEMA,
        "status": "training_semantics_validated",
        "source": {"dataset": str(dataset.resolve()), "dataset_sha256": _sha256(dataset)},
        "reports": [{
            "environment": ENVIRONMENT,
            "status": "completed",
            "raw_action_dim": RAW_ACTION_DIM,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-block", type=int, default=5)
    args = parser.parse_args()
    result = build(dataset=args.dataset, output=args.output, action_block=args.action_block)
    print(json.dumps({"output": str(args.output), "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
