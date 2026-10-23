"""Fresh-root mixed pool for C3 with V20's frozen action support."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.decision_benchmark import build_pointmaze_velocity_matched_state_pool as pool

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "point_maze_v67_c3_calibrated_reliability_20261023"
HIGH = ROOT / "data" / "point_maze_v68_c3_action_supported_high_20261023"
OUTPUT = ROOT / "data" / "point_maze_v69_c3_action_supported_20261023"
HIGH_PAIR_COUNT, DIRECT_PAIR_COUNT = 224, 192
HIGH_SEED, DIRECT_SEED = 202611601, 202611602


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _high() -> dict[str, object]:
    return {"source_root": str(INPUT.resolve()), "seed": HIGH_SEED, "pair_count": HIGH_PAIR_COUNT,
            "regime": "opposed_velocity", "velocity_magnitude": 1.5, "target_goal": [0.8, 0.8],
            "target_radius": 0.30, "history_qpos_bounds": [0.05, 3.95], "new_root_id_range": [28872, 29319]}


def _direct() -> dict[str, object]:
    return {"source_root": str(HIGH.resolve()), "seed": DIRECT_SEED, "pair_count": DIRECT_PAIR_COUNT,
            "regime": "zero_velocity", "velocity_magnitude": 0.0, "target_goal": [0.8, 0.8],
            "target_radius": 0.30, "history_qpos_bounds": [0.05, 3.95], "new_root_id_range": [29320, 29703]}


def _verify(path: Path, expected: dict[str, object]) -> dict[str, Any]:
    payload = _read(path / "velocity_matched_manifest.json")
    mismatch = {k: (payload.get(k), v) for k, v in expected.items() if payload.get(k) != v}
    if mismatch: raise RuntimeError(f"V22 pool component mismatch: {mismatch}")
    return payload


def build() -> dict[str, Any]:
    high = _verify(HIGH, _high()) if HIGH.exists() else pool.build_pool(INPUT, HIGH, pair_count=HIGH_PAIR_COUNT, seed=HIGH_SEED, target_goal=(0.8, 0.8), target_radius=.30, velocity_magnitude=1.5, history_qpos_bounds=(.05, 3.95), regime="opposed_velocity")
    direct = _verify(OUTPUT, _direct()) if OUTPUT.exists() else pool.build_pool(HIGH, OUTPUT, pair_count=DIRECT_PAIR_COUNT, seed=DIRECT_SEED, target_goal=(0.8, 0.8), target_radius=.30, velocity_magnitude=0.0, history_qpos_bounds=(.05, 3.95), regime="zero_velocity")
    path = OUTPUT / "mixed_velocity_repairability_manifest.json"
    payload = {"schema": "pointmaze_mixed_velocity_repairability_pool.v1", "input_root": str(INPUT.resolve()), "high_component": str(HIGH.resolve()), "high_root_range": [28872, 29319], "direct_root_range": [29320, 29703], "high_pair_count": HIGH_PAIR_COUNT, "direct_pair_count": DIRECT_PAIR_COUNT, "route_contract": "causal_visual_motion_norm_ge_0p02_routes_history_else_action_only", "high_manifest_sha256": _sha256(HIGH / "velocity_matched_manifest.json"), "direct_manifest_sha256": _sha256(OUTPUT / "velocity_matched_manifest.json"), "states_sha256": _sha256(OUTPUT / "states.pth"), "components": {"high": high, "direct": direct}, "claim_boundary": "Pool construction only; C3 action support is decided by the static preflight."}
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded: raise RuntimeError("existing V22 pool manifest differs")
    if not path.exists(): path.write_text(encoded, encoding="utf-8")
    return payload


if __name__ == "__main__": print(json.dumps(build(), indent=2, sort_keys=True))
