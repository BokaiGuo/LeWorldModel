"""Run V20's independent, equal-repair-budget C2 replication on host CUDA.

The deployed route is fixed before calibration/outcome labels: recurrent
repair is permitted only for visible-motion roots.  The outcome audit compares
it with the inverse visual-motion route, which invokes the same recurrent
model on exactly the complementary 50% of roots.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.decision_benchmark import run_pointmaze_pairwise_contrast_v11_cuda as runner
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.gdd_wm import preflight_pointmaze_mixed_velocity_repairability_v20 as preflight


_BASE_PRIOR_CONTEXTS = runner._prior_contexts

BASE_DATASET = runner.ROOT / "data" / "point_maze_v65_c2_budgeted_mixed_velocity_20261023"
DATASET = BASE_DATASET
PREFLIGHT = runner.ROOT / "results" / "pointmaze_mixed_velocity_repairability_v20_static_preflight_20261023.json"
OUTPUT = runner.ROOT / "results" / "pointmaze_mixed_velocity_repairability_v20_20261023_cuda"
ROOT_RANGE = (27_208, 28_040)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611403, 202611404, 202611405
EPOCHS = 160
RUN_LABEL = "V20"
RUN_SCHEMA = "pointmaze_mixed_velocity_repairability_v20_runtime.v1"
RUNTIME_FILENAME = "v20_cuda_runtime.json"
CANDIDATE_DESIGN = "velocity_matched_braking_pairs"
FIXED_POLICY_REFERENCE = "V20 fixed visible-motion repair under a 50 percent repair budget"
MOTION_POLICY = {
    "kind": "history_energy_gate", "direction": "recurrent_when_high", "threshold": 0.02,
    "motion_source": "source_history_visual_state", "fixed_threshold": True,
}
INVERSE_CONTROL_POLICY = {
    "kind": "history_energy_gate", "direction": "recurrent_when_low", "threshold": 0.02,
    "motion_source": "source_history_visual_state", "fixed_threshold": True,
}
CLAIM_BOUNDARY = (
    "C2 independent, budgeted prospective-repairability replication only: a fixed visible-motion "
    "route is tested against an equal-budget inverse route and action-only. Not C3 calibration, "
    "online MPC, contact transfer, or universal repairability evidence."
)


def _sha256(path: Path) -> str:
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
        "high_root_range": [27208, 27655], "direct_root_range": [27656, 28039],
        "route_contract": "causal_visual_motion_norm_ge_0p02_routes_history_else_action_only",
        "equal_budget_control": "causal_visual_motion_norm_lt_0p02_routes_history_else_action_only",
    }
    mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatch or payload.get("states_sha256") != _sha256(DATASET / "states.pth"):
        raise RuntimeError(f"V20 mixed-pool contract mismatch: {mismatch}")
    return {"resumed_v20_pool": True, "manifest": str(path.resolve())}


def _prior_contexts() -> tuple[Path, ...]:
    v19 = runner.ROOT / "results" / "pointmaze_mixed_velocity_repairability_v19_20261023_cuda" / "temporal_ood_v6_context.json"
    if not v19.is_file():
        raise FileNotFoundError("V20 requires V19's sealed context for action/root freshness exclusion")
    return tuple(sorted({*_BASE_PRIOR_CONTEXTS(), v19.resolve()}))


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
    runner.EXTENSION_SEED = 202611401
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
    runner._prior_contexts = _prior_contexts
    runner.interaction.calibrate_pairwise_contrast = _fixed_calibration
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
