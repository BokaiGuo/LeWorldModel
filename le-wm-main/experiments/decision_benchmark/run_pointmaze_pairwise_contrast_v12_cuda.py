"""Run V12's independent, hard balanced-reversal C1 confirmation on CUDA.

The immutable static ledger is created first on fresh roots. This wrapper only
binds that ledger to the shared C1 training/score implementation; it cannot
fall back to CPU and it never reuses V11's actions, roots, or outcomes.
"""

from __future__ import annotations

from experiments.decision_benchmark import run_pointmaze_pairwise_contrast_v11_cuda as runner
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v12 as v12


BASE_DATASET = runner.ROOT / "data" / "point_maze_v54_pairwise_contrast_v11_extension_20261022"
DATASET = runner.ROOT / "data" / "point_maze_v55_pairwise_contrast_v12_balance_extension_20261022"
PREFLIGHT = runner.ROOT / "results" / "pointmaze_pairwise_contrast_v12_balance_static_preflight_20261022.json"
OUTPUT = runner.ROOT / "results" / "pointmaze_pairwise_contrast_v12_balance_20261022_cuda"
ROOT_RANGE = (20_552, 21_384)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}
EXTENSION_SEED = 202611211
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611212, 202611213, 202611214
EPOCHS = 160
RUN_LABEL = "V12"
RUN_SCHEMA = "pointmaze_pairwise_contrast_v12_runtime.v1"
RUNTIME_FILENAME = "v12_cuda_runtime.json"
CANDIDATE_DESIGN = "pairwise_contrast_balanced_reversal_interaction_pairs"


def _configure_runner() -> None:
    """Inject V12's immutable contract only at execution time, never import."""
    runner.BASE_DATASET = BASE_DATASET
    runner.DATASET = DATASET
    runner.PREFLIGHT = PREFLIGHT
    runner.OUTPUT = OUTPUT
    runner.ROOT_RANGE = ROOT_RANGE
    runner.SPLIT_COUNTS = SPLIT_COUNTS
    runner.EXTENSION_SEED = EXTENSION_SEED
    runner.ROOT_SEED, runner.HELD_ACTION_SEED, runner.TRAIN_ACTION_SEED = ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED
    runner.EPOCHS = EPOCHS
    runner.RUN_LABEL = RUN_LABEL
    runner.RUN_SCHEMA = RUN_SCHEMA
    runner.RUNTIME_FILENAME = RUNTIME_FILENAME
    runner.PREFLIGHT_MODULE = v12
    runner.CANDIDATE_DESIGN = CANDIDATE_DESIGN


def main() -> int:
    _configure_runner()
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
