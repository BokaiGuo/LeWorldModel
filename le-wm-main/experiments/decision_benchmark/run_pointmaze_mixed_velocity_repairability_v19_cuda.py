"""Run V19's prospective mixed-mechanism C2 routing test on host CUDA.

The policy is fixed before V19 calibration/outcome labels: use the V17 C1
history model only when the causal renderer-derived XY displacement is at
least 0.02; otherwise retain action-only.  Calibration may authorize or reject
that one policy, but may not alter its threshold, feature, direction, or model
configuration.
"""

from __future__ import annotations

import hashlib
import json

from experiments.decision_benchmark import run_pointmaze_pairwise_contrast_v11_cuda as runner
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.gdd_wm import preflight_pointmaze_mixed_velocity_repairability_v19 as preflight


BASE_DATASET = runner.ROOT / "data" / "point_maze_v63_c2_mixed_velocity_repairability_20261023"
DATASET = BASE_DATASET
PREFLIGHT = runner.ROOT / "results" / "pointmaze_mixed_velocity_repairability_v19_static_preflight_20261023.json"
OUTPUT = runner.ROOT / "results" / "pointmaze_mixed_velocity_repairability_v19_20261023_cuda"
ROOT_RANGE = (26_376, 27_208)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611303, 202611304, 202611305
EPOCHS = 160
RUN_LABEL = "V19"
RUN_SCHEMA = "pointmaze_mixed_velocity_repairability_v19_runtime.v1"
RUNTIME_FILENAME = "v19_cuda_runtime.json"
CANDIDATE_DESIGN = "velocity_matched_braking_pairs"
FIXED_POLICY_REFERENCE = "V17 fixed history model plus V19 static visual-motion route"
MOTION_POLICY = {
    "kind": "history_energy_gate", "direction": "recurrent_when_high", "threshold": 0.02,
    "motion_source": "source_history_visual_state", "fixed_threshold": True,
}
CLAIM_BOUNDARY = (
    "C2 mixed-mechanism prospective-repairability test only: a pre-fixed "
    "visual-motion route is evaluated on sealed held-out outcomes. Not C3 "
    "calibration, online MPC, contact transfer, or universal repairability evidence."
)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pool() -> dict[str, object]:
    path = DATASET / "mixed_velocity_repairability_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "pointmaze_mixed_velocity_repairability_pool.v1",
        "high_root_range": [26376, 26823], "direct_root_range": [26824, 27207],
        "route_contract": "causal_visual_motion_norm_ge_0p02_routes_history_else_action_only",
    }
    mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatch or payload.get("states_sha256") != _sha256(DATASET / "states.pth"):
        raise RuntimeError(f"V19 mixed-pool contract mismatch: {mismatch}")
    return {"resumed_v19_pool": True, "manifest": str(path.resolve())}


def _fixed_calibration(source, source_arrays, context, calibration_labels, output, *, epochs, device, manifest):
    return interaction.calibrate_fixed_policy(
        source, source_arrays, context, calibration_labels, output,
        hidden=32, contrast_weight=1.0, temperature=1.0,
        history_source="source_history_model", epochs=epochs, device=device,
        manifest=manifest, protocol_reference=FIXED_POLICY_REFERENCE, policy=MOTION_POLICY,
    )


def main() -> int:
    runner.BASE_DATASET = BASE_DATASET
    runner.DATASET = DATASET
    runner.PREFLIGHT = PREFLIGHT
    runner.OUTPUT = OUTPUT
    runner.ROOT_RANGE = ROOT_RANGE
    runner.SPLIT_COUNTS = SPLIT_COUNTS
    runner.EXTENSION_SEED = 202611301
    runner.ROOT_SEED, runner.HELD_ACTION_SEED, runner.TRAIN_ACTION_SEED = ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED
    runner.EPOCHS = EPOCHS
    runner.RUN_LABEL = RUN_LABEL
    runner.RUN_SCHEMA = RUN_SCHEMA
    runner.RUNTIME_FILENAME = RUNTIME_FILENAME
    runner.PREFLIGHT_MODULE = preflight
    runner.CANDIDATE_DESIGN = CANDIDATE_DESIGN
    runner.HISTORY_GATE = True
    runner.CLAIM_BOUNDARY = CLAIM_BOUNDARY
    runner._ensure_extension = _pool
    runner.interaction.calibrate_pairwise_contrast = _fixed_calibration
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
