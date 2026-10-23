"""Fresh V18 confirmation of V17's pre-declared recurrent policy on host CUDA.

V18 fixes V17's policy configuration before touching its new calibration or
outcome roots. It retrains that one configuration on fresh train roots, uses
all fresh calibration roots for authorization only, then opens outcome labels
once. No V18 hyperparameter or policy search is permitted.
"""

from __future__ import annotations

import json

from experiments.decision_benchmark import run_pointmaze_pairwise_contrast_v11_cuda as runner
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed as preflight


BASE_DATASET = runner.ROOT / "data" / "point_maze_v61_goal_radial_visible_high_velocity_confirmation_20261023"
DATASET = BASE_DATASET
PREFLIGHT = runner.ROOT / "results" / "pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed_static_preflight_20261023.json"
OUTPUT = runner.ROOT / "results" / "pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed_20261023_cuda"
ROOT_RANGE = (25_544, 26_376)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611292, 202611293, 202611294
EPOCHS = 160
RUN_LABEL = "V18"
RUN_SCHEMA = "pointmaze_pairwise_contrast_v18_visible_high_velocity_fixed_runtime.v1"
RUNTIME_FILENAME = "v18_cuda_runtime.json"
CANDIDATE_DESIGN = "velocity_matched_braking_pairs"
FIXED_POLICY_REFERENCE = "V17: hidden=32, contrast_weight=1.0, temperature=1.0, source_history_model"


def _pool() -> dict[str, object]:
    manifest = json.loads((DATASET / "velocity_matched_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "seed": 202611291,
        "new_root_id_range": [25544, 26375],
        "velocity_magnitude": 1.5,
        "target_radius": 0.30,
        "history_qpos_bounds": [0.05, 3.95],
    }
    mismatch = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatch:
        raise RuntimeError(f"V18 fresh pool contract mismatch: {mismatch}")
    return {"resumed_v18_pool": True, "manifest": str((DATASET / "velocity_matched_manifest.json").resolve())}


def _fixed_calibration(source, source_arrays, context, calibration_labels, output, *, epochs, device, manifest):
    return interaction.calibrate_fixed_policy(
        source, source_arrays, context, calibration_labels, output,
        hidden=32, contrast_weight=1.0, temperature=1.0,
        history_source="source_history_model", epochs=epochs, device=device,
        manifest=manifest, protocol_reference=FIXED_POLICY_REFERENCE,
    )


def main() -> int:
    runner.BASE_DATASET = BASE_DATASET
    runner.DATASET = DATASET
    runner.PREFLIGHT = PREFLIGHT
    runner.OUTPUT = OUTPUT
    runner.ROOT_RANGE = ROOT_RANGE
    runner.SPLIT_COUNTS = SPLIT_COUNTS
    runner.EXTENSION_SEED = 202611291
    runner.ROOT_SEED, runner.HELD_ACTION_SEED, runner.TRAIN_ACTION_SEED = ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED
    runner.EPOCHS = EPOCHS
    runner.RUN_LABEL = RUN_LABEL
    runner.RUN_SCHEMA = RUN_SCHEMA
    runner.RUNTIME_FILENAME = RUNTIME_FILENAME
    runner.PREFLIGHT_MODULE = preflight
    runner.CANDIDATE_DESIGN = CANDIDATE_DESIGN
    runner._ensure_extension = _pool
    runner.interaction.calibrate_pairwise_contrast = _fixed_calibration
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
