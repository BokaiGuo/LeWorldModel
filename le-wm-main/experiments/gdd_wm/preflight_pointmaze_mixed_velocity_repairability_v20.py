"""Static admission for V20's independent, budgeted mixed-mechanism test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm import preflight_pointmaze_mixed_velocity_repairability_v19 as v19


SCHEMA = panel.MIXED_VELOCITY_REPAIRABILITY_V20_PREFLIGHT_SCHEMA
HIGH_RANGE, DIRECT_RANGE = (27_208, 27_656), (27_656, 28_040)
EQUAL_BUDGET_CONTROL = {
    "kind": "history_energy_gate", "direction": "recurrent_when_low", "threshold": 0.02,
    "motion_source": "source_history_visual_state", "fixed_threshold": True,
}


def preflight(
    *, dataset_root: Path, contexts: list[Path], root_seed: int, held_action_seed: int,
    train_action_seed: int, root_id_min: int, root_id_max_exclusive: int,
    split_counts: dict[str, int],
) -> dict[str, Any]:
    """Reuse V19's static checks with a fresh identity range and V20 schema."""
    saved = (v19.SCHEMA, v19.HIGH_RANGE, v19.DIRECT_RANGE)
    try:
        v19.SCHEMA, v19.HIGH_RANGE, v19.DIRECT_RANGE = SCHEMA, HIGH_RANGE, DIRECT_RANGE
        result = v19.preflight(
            dataset_root=dataset_root, contexts=contexts, root_seed=root_seed,
            held_action_seed=held_action_seed, train_action_seed=train_action_seed,
            root_id_min=root_id_min, root_id_max_exclusive=root_id_max_exclusive,
            split_counts=split_counts,
        )
    finally:
        v19.SCHEMA, v19.HIGH_RANGE, v19.DIRECT_RANGE = saved
    candidate = result.get("candidate_bank")
    checks = result.get("checks")
    if not isinstance(candidate, dict) or not isinstance(checks, dict):
        raise RuntimeError("V20 inherited malformed V19 static preflight")
    candidate["equal_budget_control"] = EQUAL_BUDGET_CONTROL
    candidate["repair_budget_fraction"] = 0.5
    checks["equal_budget_inverse_control_declared"] = True
    result["claim_boundary"] = (
        "Static C2 budgeted-route feasibility only; it fixes the high-motion route and "
        "equal-budget inverse control before learned fitting, calibration, or outcome labels."
    )
    return result
