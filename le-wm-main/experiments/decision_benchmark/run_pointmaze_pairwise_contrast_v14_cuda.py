"""Run V14 only after its grouped velocity-alias static ledger passed."""

from __future__ import annotations

import json

from experiments.decision_benchmark import run_pointmaze_pairwise_contrast_v11_cuda as runner
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v14_velocity_alias as v14


BASE_DATASET = runner.ROOT / "data" / "point_maze_v57_velocity_matched_history_20261023"
DATASET = BASE_DATASET
PREFLIGHT = runner.ROOT / "results" / "pointmaze_pairwise_contrast_v14_velocity_alias_static_preflight_20261023.json"
OUTPUT = runner.ROOT / "results" / "pointmaze_pairwise_contrast_v14_velocity_alias_20261023_cuda"
ROOT_RANGE = (22_216, 23_048)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611241, 202611242, 202611243
EPOCHS = 160
RUN_LABEL = "V14"
RUN_SCHEMA = "pointmaze_pairwise_contrast_v14_velocity_alias_runtime.v1"
RUNTIME_FILENAME = "v14_cuda_runtime.json"
CANDIDATE_DESIGN = "velocity_matched_acceleration_pairs"


def _ensure_velocity_pool() -> dict[str, object]:
    manifest_path = DATASET / "velocity_matched_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"V14 needs its immutable velocity pool: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema": "pointmaze_velocity_matched_state_pool.v1", "seed": 202611241,
        "new_root_id_range": [ROOT_RANGE[0], ROOT_RANGE[1] - 1], "pair_count": 416,
    }
    mismatch = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatch:
        raise RuntimeError(f"V14 velocity-matched pool contract mismatch: {mismatch}")
    return {"resumed_velocity_matched_pool": True, "manifest": str(manifest_path.resolve())}


def _configure_runner() -> None:
    runner.BASE_DATASET = BASE_DATASET
    runner.DATASET = DATASET
    runner.PREFLIGHT = PREFLIGHT
    runner.OUTPUT = OUTPUT
    runner.ROOT_RANGE = ROOT_RANGE
    runner.SPLIT_COUNTS = SPLIT_COUNTS
    runner.EXTENSION_SEED = 202611241
    runner.ROOT_SEED, runner.HELD_ACTION_SEED, runner.TRAIN_ACTION_SEED = ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED
    runner.EPOCHS = EPOCHS
    runner.RUN_LABEL = RUN_LABEL
    runner.RUN_SCHEMA = RUN_SCHEMA
    runner.RUNTIME_FILENAME = RUNTIME_FILENAME
    runner.PREFLIGHT_MODULE = v14
    runner.CANDIDATE_DESIGN = CANDIDATE_DESIGN
    runner._ensure_extension = _ensure_velocity_pool


def main() -> int:
    _configure_runner()
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
