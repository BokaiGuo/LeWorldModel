"""Freeze V18's fresh-root, fixed-policy C1 confirmation contract."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v13_position_alias_preflight as prior
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed as v18


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "point_maze_v61_goal_radial_visible_high_velocity_confirmation_20261023"
OUTPUT = ROOT / "results" / "pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed_static_preflight_20261023.json"
ROOT_RANGE = (25_544, 26_376)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable V18 ledger exists: {OUTPUT}")
    contexts = prior._prior_contexts()
    result = v18.preflight(
        dataset_root=DATASET, contexts=list(contexts),
        root_seed=202611292, held_action_seed=202611293, train_action_seed=202611294,
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1], split_counts=SPLIT_COUNTS,
    )
    result["extension"] = {"visible_high_velocity_confirmation_pool": str((DATASET / "velocity_matched_manifest.json").resolve())}
    result["prior_context_count"] = len(contexts)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "status": result["status"], "checks": result["checks"]}, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
