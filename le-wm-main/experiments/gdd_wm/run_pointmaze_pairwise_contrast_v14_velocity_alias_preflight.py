"""Host-only immutable V14 static admission wrapper.

This deliberately runs before a CUDA job.  It performs only MuJoCo candidate
design on the new velocity-matched pool; a rejected ledger is scientific
evidence of an unavailable mechanism, not a reason to train a larger model.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.decision_benchmark import run_pointmaze_pairwise_contrast_v11_cuda as v11
from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v13_position_alias_preflight as v13
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v14_velocity_alias as v14


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "point_maze_v57_velocity_matched_history_20261023"
OUTPUT = ROOT / "results" / "pointmaze_pairwise_contrast_v14_velocity_alias_static_preflight_20261023.json"
ROOT_RANGE = (22_216, 23_048)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611241, 202611242, 202611243


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to reopen immutable V14 static audit: {OUTPUT}")
    # V11 freezes its own historical context set once its ledger exists, while
    # a new panel must discover all prior panels before it freezes its set.
    contexts = v13._prior_contexts()
    result = v14.preflight(
        dataset_root=DATASET, contexts=list(contexts), root_seed=ROOT_SEED,
        held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1], split_counts=SPLIT_COUNTS,
    )
    result["extension"] = {"velocity_matched_pool": str((DATASET / "velocity_matched_manifest.json").resolve())}
    result["prior_context_count"] = len(contexts)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "status": result["status"], "checks": result["checks"]}, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
