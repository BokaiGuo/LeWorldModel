"""V15 static C1 admission for goal-centred radial velocity aliases."""

from __future__ import annotations

from typing import Any

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v14_velocity_alias as v14


SCHEMA = panel.PAIRWISE_CONTRAST_V15_PREFLIGHT_SCHEMA


def preflight(**kwargs: Any) -> dict[str, Any]:
    return v14.preflight(
        **kwargs, schema=SCHEMA,
        candidate_builder=panel.velocity_matched_braking_candidate_banks,
        candidate_design="velocity_matched_braking_pairs",
        require_goal_centered_radial_pool=True,
    )
