"""Static admission for V24's motion-stratified repair and calibration test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm import preflight_pointmaze_mixed_velocity_repairability_v19 as base


SCHEMA = panel.C3_STRATIFIED_REPAIR_V24_PREFLIGHT_SCHEMA
HIGH_RANGE, DIRECT_RANGE = (32_456, 32_904), (32_904, 33_288)
REGRET_BUDGET, MOTION_THRESHOLD = 0.12, 0.02
ROUTE_CONTRACT = "visible_motion_ge_0.02: recurrent; visible_motion_lt_0.02: action_only"


def preflight(*, dataset_root: Path, contexts: list[Path], **kwargs: Any) -> dict[str, Any]:
    saved = (base.SCHEMA, base.HIGH_RANGE, base.DIRECT_RANGE)
    try:
        base.SCHEMA, base.HIGH_RANGE, base.DIRECT_RANGE = SCHEMA, HIGH_RANGE, DIRECT_RANGE
        result = base.preflight(dataset_root=dataset_root, contexts=contexts, **kwargs)
    finally:
        base.SCHEMA, base.HIGH_RANGE, base.DIRECT_RANGE = saved
    result["checks"]["fixed_motion_stratified_route"] = MOTION_THRESHOLD == 0.02
    result["checks"]["fixed_nonzero_epsilon_regret_budget"] = REGRET_BUDGET > 0.0
    result["candidate_bank"].update({
        "c3_certificate": "stratified_fixed_bank_epsilon_regret_v1",
        "c3_regret_budget": REGRET_BUDGET, "c3_route_contract": ROUTE_CONTRACT,
        "c3_calibration_strata": ["high_motion", "direct_zero_motion"],
    })
    result["claim_boundary"] = (
        "Static V24 admission only: fresh roots, shared action support, a fixed visual-motion route, "
        "and stratum-specific epsilon-regret calibration. No learned fit or outcome result."
    )
    result["status"] = "passed_static_preflight" if all(result["checks"].values()) else "rejected_static_preflight"
    return result
