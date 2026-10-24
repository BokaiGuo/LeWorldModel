"""Build V25's fresh PointMaze pool after V24's duplicate-zero-root defect.

V25 never reuses V24 panel roots.  Its direct zero-motion component is built
only by the post-fix generator, which emits two independently sampled source
contexts instead of treating +0 and -0 velocity as different observations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.decision_benchmark import build_pointmaze_velocity_matched_state_pool as pool
from experiments.decision_benchmark import extend_pointmaze_state_action_pool as extension


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "point_maze_v75_c3_stratified_repair_20261023"
EXTENSION = ROOT / "data" / "point_maze_v76_c3_independent_direct_extension_20261023"
HIGH = ROOT / "data" / "point_maze_v77_c3_independent_direct_high_20261023"
OUTPUT = ROOT / "data" / "point_maze_v78_c3_independent_direct_20261023"
EXTENSION_SEED, HIGH_SEED, DIRECT_SEED = 202611901, 202611902, 202611903
EXTENSION_EPISODES, HIGH_PAIR_COUNT, DIRECT_PAIR_COUNT = 960, 224, 192


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extension() -> dict[str, Any]:
    expected = {"source_root": str(INPUT.resolve()), "seed": EXTENSION_SEED, "new_episode_count": EXTENSION_EPISODES,
                "base_root_count": 33_288, "extended_root_count": 34_248, "new_root_id_range": [33_288, 34_247]}
    manifest = EXTENSION / "extension_manifest.json"
    if EXTENSION.exists():
        payload = _read(manifest)
        mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatch:
            raise RuntimeError(f"V25 extension contract mismatch: {mismatch}")
        return payload
    return extension.build_extension(INPUT, EXTENSION, episode_count=EXTENSION_EPISODES, seed=EXTENSION_SEED)


def _high() -> dict[str, object]:
    return {"source_root": str(EXTENSION.resolve()), "seed": HIGH_SEED, "pair_count": HIGH_PAIR_COUNT,
            "regime": "opposed_velocity", "velocity_magnitude": 1.5, "target_goal": [0.8, 0.8],
            "target_radius": .30, "history_qpos_bounds": [.05, 3.95], "new_root_id_range": [34_248, 34_695]}


def _direct() -> dict[str, object]:
    return {"source_root": str(HIGH.resolve()), "seed": DIRECT_SEED, "pair_count": DIRECT_PAIR_COUNT,
            "regime": "zero_velocity", "velocity_magnitude": 0.0, "target_goal": [0.8, 0.8],
            "target_radius": .30, "history_qpos_bounds": [.05, 3.95], "new_root_id_range": [34_696, 35_079]}


def _verify(path: Path, expected: dict[str, object]) -> dict[str, Any]:
    payload = _read(path / "velocity_matched_manifest.json")
    mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatch:
        raise RuntimeError(f"V25 pool component mismatch: {mismatch}")
    if payload.get("regime") == "zero_velocity" and payload.get("unique_source_signature_count") != payload.get("new_episode_count"):
        raise RuntimeError("V25 direct control lacks one unique source observation per nominal root")
    return payload


def build() -> dict[str, Any]:
    extended = _extension()
    high = _verify(HIGH, _high()) if HIGH.exists() else pool.build_pool(
        EXTENSION, HIGH, pair_count=HIGH_PAIR_COUNT, seed=HIGH_SEED, target_goal=(.8, .8), target_radius=.30,
        velocity_magnitude=1.5, history_qpos_bounds=(.05, 3.95), regime="opposed_velocity",
    )
    direct = _verify(OUTPUT, _direct()) if OUTPUT.exists() else pool.build_pool(
        HIGH, OUTPUT, pair_count=DIRECT_PAIR_COUNT, seed=DIRECT_SEED, target_goal=(.8, .8), target_radius=.30,
        velocity_magnitude=0.0, history_qpos_bounds=(.05, 3.95), regime="zero_velocity",
    )
    path = OUTPUT / "mixed_velocity_repairability_manifest.json"
    payload = {
        "schema": "pointmaze_mixed_velocity_repairability_pool.v2_independent_direct",
        "input_root": str(EXTENSION.resolve()), "extension": extended, "high_component": str(HIGH.resolve()),
        "high_root_range": [34_248, 34_695], "direct_root_range": [34_696, 35_079],
        "high_pair_count": HIGH_PAIR_COUNT, "direct_independent_root_count": int(direct["new_episode_count"]),
        "route_contract": "causal_visual_motion_norm_ge_0p02_routes_history_else_action_only",
        "equal_budget_control": "independent_zero_motion_source_contexts_route_action_only",
        "high_manifest_sha256": _sha256(HIGH / "velocity_matched_manifest.json"),
        "direct_manifest_sha256": _sha256(OUTPUT / "velocity_matched_manifest.json"),
        "states_sha256": _sha256(OUTPUT / "states.pth"), "components": {"high": high, "direct": direct},
        "claim_boundary": "Fresh-pool construction only; no model fit, calibration, outcome label, or planning claim.",
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("existing V25 pool manifest differs")
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
