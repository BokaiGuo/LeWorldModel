"""Host-only immutable static admission wrapper for V15."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v13_position_alias_preflight as v13
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v15_radial_velocity_alias as v15


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "point_maze_v58_goal_radial_velocity_history_20261023"
OUTPUT = ROOT / "results" / "pointmaze_pairwise_contrast_v15_radial_velocity_alias_static_preflight_20261023.json"
ROOT_RANGE = (23_048, 23_880)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611252, 202611253, 202611254


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to reopen immutable V15 static audit: {OUTPUT}")
    contexts = v13._prior_contexts()
    result = v15.preflight(
        dataset_root=DATASET, contexts=list(contexts), root_seed=ROOT_SEED,
        held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1], split_counts=SPLIT_COUNTS,
    )
    result["extension"] = {"goal_radial_velocity_pool": str((DATASET / "velocity_matched_manifest.json").resolve())}
    result["prior_context_count"] = len(contexts)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "status": result["status"], "checks": result["checks"]}, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
