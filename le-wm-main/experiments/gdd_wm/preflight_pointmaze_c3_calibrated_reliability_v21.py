"""Static identity admission for V21 C3; no calibration or test labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm import preflight_pointmaze_mixed_velocity_repairability_v19 as base


SCHEMA = panel.C3_CALIBRATED_RELIABILITY_V21_PREFLIGHT_SCHEMA
HIGH_RANGE, DIRECT_RANGE = (28_040, 28_488), (28_488, 28_872)


def preflight(**kwargs: Any) -> dict[str, Any]:
    saved = (base.SCHEMA, base.HIGH_RANGE, base.DIRECT_RANGE)
    try:
        base.SCHEMA, base.HIGH_RANGE, base.DIRECT_RANGE = SCHEMA, HIGH_RANGE, DIRECT_RANGE
        result = base.preflight(**kwargs)
    finally:
        base.SCHEMA, base.HIGH_RANGE, base.DIRECT_RANGE = saved
    result["claim_boundary"] = (
        "Static V21 C3 admission only. It freezes new identities and the visible-motion route before "
        "the V20 model is replayed, before calibration labels, and before final outcome labels."
    )
    return result
