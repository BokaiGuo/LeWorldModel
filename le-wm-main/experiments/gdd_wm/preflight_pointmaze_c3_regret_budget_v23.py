"""Static admission for V23's fixed-bank epsilon-regret C3 test.

V22 showed that a unique-winner certificate has a finer resolution than the
declared 16-way bank can support.  V23 fixes an epsilon-regret certificate
*before* any model fit, calibration labels, or outcome labels are opened.  It
uses a newly fitted, action-supported scorer on entirely fresh roots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm import preflight_pointmaze_mixed_velocity_repairability_v19 as base


SCHEMA = panel.C3_REGRET_BUDGET_V23_PREFLIGHT_SCHEMA
HIGH_RANGE, DIRECT_RANGE = (30_664, 31_112), (31_112, 31_496)
REGRET_BUDGET = 0.12


def preflight(*, dataset_root: Path, contexts: list[Path], **kwargs: Any) -> dict[str, Any]:
    """Run V19's causal/root-disjoint admission under V23's frozen ranges."""

    saved = (base.SCHEMA, base.HIGH_RANGE, base.DIRECT_RANGE)
    try:
        base.SCHEMA, base.HIGH_RANGE, base.DIRECT_RANGE = SCHEMA, HIGH_RANGE, DIRECT_RANGE
        result = base.preflight(dataset_root=dataset_root, contexts=contexts, **kwargs)
    finally:
        base.SCHEMA, base.HIGH_RANGE, base.DIRECT_RANGE = saved
    result["checks"]["fixed_nonzero_epsilon_regret_budget"] = REGRET_BUDGET > 0.0
    result["candidate_bank"]["c3_certificate"] = "fixed_bank_epsilon_regret_v1"
    result["candidate_bank"]["c3_regret_budget"] = REGRET_BUDGET
    result["claim_boundary"] = (
        "Static V23 C3 admission only: fresh roots, shared action support, and a predeclared "
        "epsilon-regret certificate. No fit, calibration label, outcome label, or planning claim."
    )
    result["status"] = "passed_static_preflight" if all(result["checks"].values()) else "rejected_static_preflight"
    return result
