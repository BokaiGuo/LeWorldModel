"""Construct a fresh mixed-velocity pool for V21's C3 calibration test.

No learned score or task outcome is read here.  V21 is deliberately sourced
after V20's complete root range, so the frozen V20 repair model can be tested
without reusing any of its fit, calibration, or outcome roots.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.decision_benchmark import build_pointmaze_velocity_matched_state_pool as pool


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "point_maze_v65_c2_budgeted_mixed_velocity_20261023"
HIGH = ROOT / "data" / "point_maze_v66_c3_high_part_20261023"
OUTPUT = ROOT / "data" / "point_maze_v67_c3_calibrated_reliability_20261023"
HIGH_PAIR_COUNT, DIRECT_PAIR_COUNT = 224, 192
HIGH_SEED, DIRECT_SEED = 202611501, 202611502


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_high() -> dict[str, object]:
    return {
        "source_root": str(INPUT.resolve()), "seed": HIGH_SEED,
        "pair_count": HIGH_PAIR_COUNT, "regime": "opposed_velocity",
        "velocity_magnitude": 1.5, "target_goal": [0.8, 0.8],
        "target_radius": 0.30, "history_qpos_bounds": [0.05, 3.95],
        "new_root_id_range": [28040, 28487],
    }


def _expected_direct() -> dict[str, object]:
    return {
        "source_root": str(HIGH.resolve()), "seed": DIRECT_SEED,
        "pair_count": DIRECT_PAIR_COUNT, "regime": "zero_velocity",
        "velocity_magnitude": 0.0, "target_goal": [0.8, 0.8],
        "target_radius": 0.30, "history_qpos_bounds": [0.05, 3.95],
        "new_root_id_range": [28488, 28871],
    }


def _verify(path: Path, expected: dict[str, object]) -> dict[str, Any]:
    manifest = path / "velocity_matched_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing V21 pool component manifest: {manifest}")
    payload = _read(manifest)
    mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatch:
        raise RuntimeError(f"V21 pool component contract mismatch: {mismatch}")
    return payload


def build() -> dict[str, Any]:
    high = _verify(HIGH, _expected_high()) if HIGH.exists() else pool.build_pool(
        INPUT, HIGH, pair_count=HIGH_PAIR_COUNT, seed=HIGH_SEED, target_goal=(0.8, 0.8),
        target_radius=0.30, velocity_magnitude=1.5, history_qpos_bounds=(0.05, 3.95), regime="opposed_velocity",
    )
    direct = _verify(OUTPUT, _expected_direct()) if OUTPUT.exists() else pool.build_pool(
        HIGH, OUTPUT, pair_count=DIRECT_PAIR_COUNT, seed=DIRECT_SEED, target_goal=(0.8, 0.8),
        target_radius=0.30, velocity_magnitude=0.0, history_qpos_bounds=(0.05, 3.95), regime="zero_velocity",
    )
    path = OUTPUT / "mixed_velocity_repairability_manifest.json"
    expected = {
        "schema": "pointmaze_mixed_velocity_repairability_pool.v1", "input_root": str(INPUT.resolve()),
        "high_component": str(HIGH.resolve()), "high_root_range": [28040, 28487],
        "direct_root_range": [28488, 28871], "high_pair_count": HIGH_PAIR_COUNT,
        "direct_pair_count": DIRECT_PAIR_COUNT,
        "route_contract": "causal_visual_motion_norm_ge_0p02_routes_history_else_action_only",
    }
    payload = {**expected, "high_manifest_sha256": _sha256(HIGH / "velocity_matched_manifest.json"),
               "direct_manifest_sha256": _sha256(OUTPUT / "velocity_matched_manifest.json"),
               "states_sha256": _sha256(OUTPUT / "states.pth"), "components": {"high": high, "direct": direct},
               "claim_boundary": "Pool construction only: no fit, calibration outcome, final outcome, or C3 claim."}
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("existing V21 pool manifest differs from its frozen construction")
    else:
        path.write_text(encoded, encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
