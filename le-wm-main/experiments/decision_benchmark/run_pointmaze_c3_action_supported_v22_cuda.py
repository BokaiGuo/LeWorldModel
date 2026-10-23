"""Run V22 C3 with V20's exact declared action support on fresh roots."""
from __future__ import annotations

from experiments.decision_benchmark import run_pointmaze_c3_calibrated_reliability_v21_cuda as runner
from experiments.gdd_wm import preflight_pointmaze_c3_action_supported_v22 as preflight


def main() -> int:
    root = runner.ROOT
    runner.DATASET = root / "data" / "point_maze_v69_c3_action_supported_20261023"
    runner.OUTPUT = root / "results" / "pointmaze_c3_action_supported_v22_20261023_cuda"
    runner.PREFLIGHT = root / "results" / "pointmaze_c3_action_supported_v22_static_preflight_20261023.json"
    runner.ROOT_RANGE, runner.SPLIT_COUNTS = (28_872, 29_704), {"train": 256, "calibration": 256, "outcome": 256}
    # Exact V20 held/shared candidate bank.  The roots, pool construction and
    # calibration/outcome partitions remain entirely new.
    runner.ROOT_SEED, runner.HELD_ACTION_SEED, runner.TRAIN_ACTION_SEED = 202611603, 202611404, 202611405
    runner.PREFLIGHT_MODULE = preflight
    runner.POOL_MODULE = "experiments.decision_benchmark.build_pointmaze_v22_c3_action_supported_pool"
    import json
    import numpy as np
    payload = json.loads((runner.V20_PANEL / "temporal_ood_v6_context.json").read_text(encoding="utf-8"))
    root_id = sorted({str(row["instance_id"]) for row in payload["records"] if row["split"] == "outcome"})[0]
    rows = sorted((row for row in payload["records"] if str(row["instance_id"]) == root_id), key=lambda row: int(row["candidate_id"]))
    runner.FIXED_HELD_ACTIONS = np.asarray([row["action_sequence"] for row in rows], dtype=np.float64)
    runner.ALLOWED_PRIOR_ACTION_HASHES = {str(row["candidate_action_hash"]) for row in rows}
    return runner.main()


if __name__ == "__main__": raise SystemExit(main())
