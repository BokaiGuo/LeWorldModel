"""Fail closed when nominal PointMaze roots duplicate the same observation.

Root identifiers are bookkeeping, not experimental independence. A fixed-bank
calibration or outcome panel must reject two differently named roots that have
the same state, observable history, and goal.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


_SIGNATURE_FIELDS = ("source_state_raw", "source_history_visual_state", "goal_raw")


def _signature(row: dict[str, Any]) -> str:
    missing = [field for field in _SIGNATURE_FIELDS if field not in row]
    if missing:
        raise ValueError(f"root observation is missing identity fields: {missing}")
    return json.dumps(
        {field: row[field] for field in _SIGNATURE_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def assert_unique_root_observations(records: list[dict[str, Any]]) -> dict[str, int]:
    """Validate root-constant observations and cross-root uniqueness per split."""

    by_split_root: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in records:
        try:
            split, root = str(row["split"]), str(row["instance_id"])
        except KeyError as exc:
            raise ValueError("root observation needs split and instance_id") from exc
        by_split_root[(split, root)].add(_signature(row))
    inconsistent = [f"{split}/{root}" for (split, root), values in by_split_root.items() if len(values) != 1]
    if inconsistent:
        raise ValueError(f"candidate rows disagree about one root observation: {sorted(inconsistent)}")

    by_split_signature: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for (split, root), values in by_split_root.items():
        by_split_signature[split][next(iter(values))].append(root)
    duplicates = {
        split: sorted(sorted(roots) for roots in signatures.values() if len(roots) > 1)
        for split, signatures in by_split_signature.items()
    }
    duplicates = {split: groups for split, groups in duplicates.items() if groups}
    if duplicates:
        count = sum(len(groups) for groups in duplicates.values())
        examples = {split: groups[:3] for split, groups in duplicates.items()}
        raise ValueError(
            "nominally distinct roots duplicate one observable context: "
            f"{count} duplicate groups; examples={examples}"
        )
    return {split: len(signatures) for split, signatures in sorted(by_split_signature.items())}
