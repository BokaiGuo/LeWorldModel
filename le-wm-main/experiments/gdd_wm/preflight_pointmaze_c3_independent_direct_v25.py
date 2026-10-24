"""Static C3 admission for V25's fresh, independent zero-motion control."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm import preflight_pointmaze_mixed_velocity_repairability_v19 as base


# Retain the accepted C3 ledger schema so the shared panel builder enforces its
# existing provenance rules; the explicit successor tag below distinguishes
# this repaired protocol from V24.
SCHEMA = panel.C3_STRATIFIED_REPAIR_V24_PREFLIGHT_SCHEMA
POOL_SCHEMA = "pointmaze_mixed_velocity_repairability_pool.v2_independent_direct"
HIGH_RANGE, DIRECT_RANGE = (34_248, 34_696), (34_696, 35_080)
ROUTE_CONTRACT = "visible_motion_ge_0.02: recurrent; visible_motion_lt_0.02: action_only"


def _high_groups() -> list[tuple[int, int]]:
    return [(root, root + 1) for root in range(HIGH_RANGE[0], HIGH_RANGE[1], 2)]


def _assign(seed: int, split_counts: dict[str, int]) -> tuple[dict[int, str], list[int], dict[str, dict[str, int]]]:
    if split_counts != {"train": 256, "calibration": 256, "outcome": 256}:
        raise ValueError("V25 fixes an exactly balanced 256/256/256 root protocol")
    high = _high_groups()
    direct = list(range(DIRECT_RANGE[0], DIRECT_RANGE[1]))
    if len(high) != 224 or len(direct) != 384:
        raise RuntimeError("V25 fresh-pool counts are malformed")
    rng = np.random.default_rng(int(seed))
    high = [high[index] for index in rng.permutation(len(high)).tolist()]
    direct = [direct[index] for index in rng.permutation(len(direct)).tolist()]
    split: dict[int, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for index, name in enumerate(("train", "calibration", "outcome")):
        high_chunk = high[index * 64 : (index + 1) * 64]
        direct_chunk = direct[index * 128 : (index + 1) * 128]
        for left, right in high_chunk:
            split[left] = name
            split[right] = name
        for root in direct_chunk:
            split[root] = name
        counts[name] = {"high_motion": len(high_chunk) * 2, "direct_zero_motion": len(direct_chunk)}
    design = [root for pair in high[192:] for root in pair]
    if len(split) != 768 or len(design) != 64 or set(split) & set(design):
        raise RuntimeError("V25 split construction failed")
    return split, design, counts


def _regime_checks(states: np.ndarray) -> dict[str, bool]:
    high_ok = all(
        float(np.linalg.norm(states[left, panel.SOURCE_FRAME, :2] - states[right, panel.SOURCE_FRAME, :2])) <= 0.025
        and float(np.linalg.norm(states[left, panel.SOURCE_FRAME, 2:])) >= 0.08
        and float(np.linalg.norm(states[right, panel.SOURCE_FRAME, 2:])) >= 0.08
        and float(np.dot(states[left, panel.SOURCE_FRAME, 2:], states[right, panel.SOURCE_FRAME, 2:])) < 0.0
        for left, right in _high_groups()
    )
    direct = states[DIRECT_RANGE[0] : DIRECT_RANGE[1]]
    signatures = [np.ascontiguousarray(row[panel.SOURCE_FRAME - 7 : panel.SOURCE_FRAME + 1]).tobytes() for row in direct]
    return {
        "high_velocity_alias_groups_valid": high_ok,
        "direct_zero_motion_valid": bool(np.all(np.linalg.norm(direct[:, panel.SOURCE_FRAME, 2:], axis=1) <= 0.08)),
        "direct_source_observations_unique": len(set(signatures)) == len(signatures),
    }


def preflight(*, dataset_root: Any, contexts: list[Any], **kwargs: Any) -> dict[str, Any]:
    saved = (base.SCHEMA, base.POOL_SCHEMA, base.HIGH_RANGE, base.DIRECT_RANGE, base._assign, base._regime_checks)
    try:
        base.SCHEMA, base.POOL_SCHEMA = SCHEMA, POOL_SCHEMA
        base.HIGH_RANGE, base.DIRECT_RANGE = HIGH_RANGE, DIRECT_RANGE
        base._assign, base._regime_checks = _assign, _regime_checks
        result = base.preflight(dataset_root=dataset_root, contexts=contexts, **kwargs)
    finally:
        base.SCHEMA, base.POOL_SCHEMA, base.HIGH_RANGE, base.DIRECT_RANGE, base._assign, base._regime_checks = saved
    result["successor_protocol"] = "v25_independent_direct_roots_after_v24_duplicate_root_invalidation"
    result["candidate_bank"]["c3_certificate"] = "stratified_fixed_bank_epsilon_regret_v2_independent_direct"
    result["status"] = "passed_static_preflight" if all(result["checks"].values()) else "rejected_static_preflight"
    return result
