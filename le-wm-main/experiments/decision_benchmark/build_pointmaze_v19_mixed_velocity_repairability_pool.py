"""Build the fresh mixed-mechanism pool for the V19 C2 routing test.

The high-motion half is the renderer-visible velocity-alias mechanism that
requires history beyond the source position.  The direct half has an identical
current-state interface but zero source velocity, so a route can legally
retain its action-only fallback.  The mechanism label is never supplied to a
learned scorer; V19 may only derive the route from causal visual motion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.decision_benchmark import build_pointmaze_velocity_matched_state_pool as pool


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "point_maze_v61_goal_radial_visible_high_velocity_confirmation_20261023"
HIGH = ROOT / "data" / "point_maze_v62_c2_mixed_velocity_high_part_20261023"
OUTPUT = ROOT / "data" / "point_maze_v63_c2_mixed_velocity_repairability_20261023"
HIGH_PAIR_COUNT = 224
DIRECT_PAIR_COUNT = 192
HIGH_SEED = 202611301
DIRECT_SEED = 202611302


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
        "new_root_id_range": [26376, 26823],
    }


def _expected_direct() -> dict[str, object]:
    return {
        "source_root": str(HIGH.resolve()), "seed": DIRECT_SEED,
        "pair_count": DIRECT_PAIR_COUNT, "regime": "zero_velocity",
        "velocity_magnitude": 0.0, "target_goal": [0.8, 0.8],
        "target_radius": 0.30, "history_qpos_bounds": [0.05, 3.95],
        "new_root_id_range": [26824, 27207],
    }


def _verify(path: Path, expected: dict[str, object]) -> dict[str, Any]:
    manifest = path / "velocity_matched_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing mixed-pool component manifest: {manifest}")
    payload = _read(manifest)
    mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatch:
        raise RuntimeError(f"mixed-pool component contract mismatch: {mismatch}")
    return payload


def build() -> dict[str, Any]:
    if HIGH.exists():
        high = _verify(HIGH, _expected_high())
    else:
        high = pool.build_pool(
            INPUT, HIGH, pair_count=HIGH_PAIR_COUNT, seed=HIGH_SEED,
            target_goal=(0.8, 0.8), target_radius=0.30, velocity_magnitude=1.5,
            history_qpos_bounds=(0.05, 3.95), regime="opposed_velocity",
        )
    if OUTPUT.exists():
        direct = _verify(OUTPUT, _expected_direct())
    else:
        direct = pool.build_pool(
            HIGH, OUTPUT, pair_count=DIRECT_PAIR_COUNT, seed=DIRECT_SEED,
            target_goal=(0.8, 0.8), target_radius=0.30, velocity_magnitude=0.0,
            history_qpos_bounds=(0.05, 3.95), regime="zero_velocity",
        )
    path = OUTPUT / "mixed_velocity_repairability_manifest.json"
    expected = {
        "schema": "pointmaze_mixed_velocity_repairability_pool.v1",
        "input_root": str(INPUT.resolve()), "high_component": str(HIGH.resolve()),
        "high_root_range": [26376, 26823], "direct_root_range": [26824, 27207],
        "high_pair_count": HIGH_PAIR_COUNT, "direct_pair_count": DIRECT_PAIR_COUNT,
        "route_contract": "causal_visual_motion_norm_ge_0p02_routes_history_else_action_only",
    }
    payload = {
        **expected,
        "high_manifest_sha256": _sha256(HIGH / "velocity_matched_manifest.json"),
        "direct_manifest_sha256": _sha256(OUTPUT / "velocity_matched_manifest.json"),
        "states_sha256": _sha256(OUTPUT / "states.pth"),
        "components": {"high": high, "direct": direct},
        "claim_boundary": "State-pool construction only; it does not fit a model, use calibration/outcome labels, or establish C2 repairability.",
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("existing V19 mixed-pool manifest differs from the frozen construction")
    else:
        path.write_text(encoded, encoding="utf-8")
    return payload


def main() -> int:
    print(json.dumps(build(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
