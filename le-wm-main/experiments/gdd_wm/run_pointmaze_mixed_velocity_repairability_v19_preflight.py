"""Write the immutable V19 C2 static ledger without launching a learned fit."""

from __future__ import annotations

import json

from experiments.decision_benchmark import run_pointmaze_pairwise_contrast_v11_cuda as runner
from experiments.gdd_wm import preflight_pointmaze_mixed_velocity_repairability_v19 as v19


DATASET = runner.ROOT / "data" / "point_maze_v63_c2_mixed_velocity_repairability_20261023"
OUTPUT = runner.ROOT / "results" / "pointmaze_mixed_velocity_repairability_v19_static_preflight_20261023.json"
ROOT_RANGE = (26_376, 27_208)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611303, 202611304, 202611305


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable V19 static ledger {OUTPUT}")
    contexts = runner._prior_contexts()
    result = v19.preflight(
        dataset_root=DATASET, contexts=list(contexts), root_seed=ROOT_SEED,
        held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1], split_counts=SPLIT_COUNTS,
    )
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": result["checks"], "output": str(OUTPUT)}, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
