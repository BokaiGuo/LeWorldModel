"""Read-only, equal-repair-budget outcome audit for V20's fresh C2 panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.gdd_wm import audit_pointmaze_mixed_velocity_repairability_v19_outcome as base


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "results" / "pointmaze_mixed_velocity_repairability_v20_20261023_cuda"
OUTPUT = PANEL / "v20_outcome_root_cluster_audit.json"
POLICY = {
    "kind": "history_energy_gate", "direction": "recurrent_when_high", "threshold": 0.02,
    "motion_source": "source_history_visual_state", "fixed_threshold": True,
}
INVERSE_CONTROL_POLICY = {
    "kind": "history_energy_gate", "direction": "recurrent_when_low", "threshold": 0.02,
    "motion_source": "source_history_visual_state", "fixed_threshold": True,
}


def _configure_base() -> dict[str, object]:
    values = {
        "PANEL": base.PANEL, "SOURCE": base.SOURCE, "ARRAYS": base.ARRAYS, "CONTEXT": base.CONTEXT,
        "FROZEN": base.FROZEN, "LABELS": base.LABELS, "SCORE": base.SCORE,
        "MANIFEST": base.MANIFEST, "POLICY": base.POLICY,
    }
    base.PANEL = PANEL
    base.SOURCE = PANEL / "temporal_ood_v6_source_dataset.json"
    base.ARRAYS = PANEL / "temporal_ood_v6_source_arrays.npz"
    base.CONTEXT = PANEL / "temporal_ood_v6_context.json"
    base.FROZEN = PANEL / "temporal_ood_v6_frozen_selection.json"
    base.LABELS = PANEL / "score" / "temporal_ood_v6_outcome_labels.json"
    base.SCORE = PANEL / "score" / "temporal_ood_v6_score_report.json"
    base.MANIFEST = PANEL / "temporal_ood_v6_manifest.json"
    base.POLICY = POLICY
    return values


def _restore_base(values: dict[str, object]) -> None:
    for name, value in values.items():
        setattr(base, name, value)


def audit(*, device: str = "cuda") -> dict[str, Any]:
    values = _configure_base()
    try:
        result = base.audit(device=device, comparison_policy=INVERSE_CONTROL_POLICY)
    finally:
        _restore_base(values)
    control = result.get("equal_budget_control")
    if not isinstance(control, dict):
        raise RuntimeError("V20 audit did not materialize its predeclared equal-budget inverse control")
    if control.get("policy") != INVERSE_CONTROL_POLICY:
        raise ValueError("V20 equal-budget control identity changed")
    if float(result.get("routed_mask_fraction", float("nan"))) != 0.5 or float(control.get("recurrent_fraction", float("nan"))) != 0.5:
        raise ValueError("V20 route/control no longer use the exact 50 percent recurrent budget")
    result["schema"] = "pointmaze_mixed_velocity_repairability_v20_budgeted_outcome_root_cluster_audit.v1"
    result["claim_boundary"] = (
        "Read-only V20 audit of a fresh pre-fixed, 50-percent repair-budget route. It tests whether "
        "visible-motion routing beats action-only and an equal-budget inverse route; it is not C3 "
        "calibration, online MPC, contact transfer, or universal repairability evidence."
    )
    result["outcome_gain_authorized"] = bool(
        result["outcome_gain_authorized"] and result["route_beats_equal_budget_control"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="new immutable V20 audit path")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable V20 outcome audit {output}")
    result = audit()
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "outcome_gain_authorized": result["outcome_gain_authorized"],
        "routed_over_action_only": result["routed_over_action_only"],
        "routed_over_equal_budget_control": result["equal_budget_control"]["routed_over_control"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
