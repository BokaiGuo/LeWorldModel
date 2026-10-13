"""Canonicalize an existing JEPA enriched metrics artifact.

This is a lossless schema migration only.  It accepts an export when every
field can be recovered from the artifact itself (or from the declared source
state convention), and fails closed otherwise.  It never manufactures a
latent or a target from vector-only historical output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.enriched_record_schema import make_payload  # noqa: E402


def _canonical_row(row: dict[str, Any], *, environment: str) -> dict[str, Any]:
    if "source_latent" not in row or not row["source_latent"]:
        raise ValueError("historical vector-only or source-latent-free row cannot be canonicalized")
    if "source_proprio_raw" not in row:
        # PushT's released state chart is [agent_xy, block_xy, sin/cos(theta),
        # agent_velocity_xy].  The proprioceptive input is the agent position
        # and velocity.  This is a declared source convention, not an inferred
        # target or a learned value.
        state = np.asarray(row.get("source_state_raw"), dtype=np.float64)
        if environment.lower() == "pusht" and state.shape == (7,):
            row["source_proprio_raw"] = [float(state[0]), float(state[1]), float(state[5]), float(state[6])]
        else:
            raise ValueError("source_proprio_raw missing and no declared environment convention applies")
    if "action_blocks_normalized" not in row:
        # The PointMaze evaluator's raw action blocks are the normalized action
        # values already consumed by the predictor.  This is recorded explicitly
        # so later matching cannot silently compare incompatible action units.
        if environment.lower() == "pointmaze":
            row["action_blocks_normalized"] = row["action_blocks_raw"]
        else:
            raise ValueError("action_blocks_normalized missing; refusing to guess action units")
    row.setdefault("sample_id", f"{row['episode_id']}:{row.get('window_start', 0)}")
    return row


def canonicalize(
    source: Path,
    *,
    architecture: str,
    checkpoint: str,
    environment: str,
    seed: str,
    training_stage: int,
    action_distribution: str,
    split_name: str | None = None,
    episode_ids: list[int] | None = None,
) -> dict[str, Any]:
    metrics = json.loads(source.read_text())
    rows = metrics.get("enriched_records")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source} does not contain enriched_records")
    rows = [_canonical_row(dict(row), environment=environment) for row in rows]
    horizons = sorted({int(h) for row in rows for h in row["latent_by_horizon"]})
    dimensions = {len(row["source_latent"]) for row in rows}
    if len(dimensions) != 1:
        raise ValueError(f"source latent dimension is not constant: {dimensions}")
    return make_payload(
        architecture=architecture,
        checkpoint=checkpoint,
        environment=environment,
        seed=seed,
        training_stage=training_stage,
        latent_dimension=next(iter(dimensions)),
        action_distribution=action_distribution,
        split={
            "name": split_name or metrics.get("protocol", {}).get("split", "unknown"),
            "episode_ids": sorted(set(int(row["episode_id"]) for row in rows)) if episode_ids is None else episode_ids,
            "provenance": f"lossless migration from {source.name}",
        },
        horizons=horizons,
        enriched_records=rows,
        checkpoint_sha256=metrics.get("protocol", {}).get("checkpoint_sha256"),
        dataset_sha256=metrics.get("protocol", {}).get("dataset_archive_sha256"),
        protocol=metrics.get("protocol", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--training-stage", required=True, type=int)
    parser.add_argument("--action-distribution", required=True)
    args = parser.parse_args()
    payload = canonicalize(
        args.source,
        architecture=args.architecture,
        checkpoint=args.checkpoint,
        environment=args.environment,
        seed=args.seed,
        training_stage=args.training_stage,
        action_distribution=args.action_distribution,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"record_schema": payload["record_schema"], "rows": len(payload["enriched_records"]), "latent_dimension": payload["latent_dimension"]}, indent=2))


if __name__ == "__main__":
    main()

