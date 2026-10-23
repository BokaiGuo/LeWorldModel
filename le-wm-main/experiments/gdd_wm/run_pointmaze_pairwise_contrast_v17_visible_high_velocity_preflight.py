"""Freeze V17's fresh high-velocity, renderer-visible C1 panel."""
from __future__ import annotations
import json
from pathlib import Path
from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v13_position_alias_preflight as v13
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v16_visible_radial_velocity_alias as v16

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "point_maze_v60_goal_radial_visible_high_velocity_history_20261023"
OUTPUT = ROOT / "results" / "pointmaze_pairwise_contrast_v17_visible_high_velocity_static_preflight_20261023.json"
ROOT_RANGE = (24_712, 25_544)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}

def main() -> int:
    if OUTPUT.exists(): raise FileExistsError(f"immutable V17 ledger exists: {OUTPUT}")
    contexts = v13._prior_contexts()
    result = v16.preflight(dataset_root=DATASET, contexts=list(contexts), root_seed=202611272,
        held_action_seed=202611273, train_action_seed=202611274,
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1], split_counts=SPLIT_COUNTS)
    result["extension"] = {"visible_high_velocity_pool": str((DATASET / "velocity_matched_manifest.json").resolve())}
    result["prior_context_count"] = len(contexts)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"], "output": str(OUTPUT)}, indent=2))
    return 0 if result["status"] == "passed_static_preflight" else 2
if __name__ == "__main__": raise SystemExit(main())
