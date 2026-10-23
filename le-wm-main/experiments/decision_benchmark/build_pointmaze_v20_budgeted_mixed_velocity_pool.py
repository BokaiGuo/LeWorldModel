"""Build an independent, mixed-mechanism pool for V20's budgeted C2 test.

V20 is not a retry of V19's outcome panel: it uses fresh generated roots and
will compare a predeclared high-motion repair route with an equal-budget
inverse-motion control.  The pool builder never reads model scores, outcome
labels, or terminal task costs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.decision_benchmark import build_pointmaze_velocity_matched_state_pool as pool


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "point_maze_v63_c2_mixed_velocity_repairability_20261023"
HIGH = ROOT / "data" / "point_maze_v64_c2_budgeted_high_part_20261023"
OUTPUT = ROOT / "data" / "point_maze_v65_c2_budgeted_mixed_velocity_20261023"
HIGH_PAIR_COUNT, DIRECT_PAIR_COUNT = 224, 192
HIGH_SEED, DIRECT_SEED = 202611401, 202611402


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
        "new_root_id_range": [27208, 27655],
    }


def _expected_direct() -> dict[str, object]:
    return {
        "source_root": str(HIGH.resolve()), "seed": DIRECT_SEED,
        "pair_count": DIRECT_PAIR_COUNT, "regime": "zero_velocity",
        "velocity_magnitude": 0.0, "target_goal": [0.8, 0.8],
        "target_radius": 0.30, "history_qpos_bounds": [0.05, 3.95],
        "new_root_id_range": [27656, 28039],
    }


def _verify(path: Path, expected: dict[str, object]) -> dict[str, Any]:
    manifest = path / "velocity_matched_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing V20 pool component manifest: {manifest}")
    payload = _read(manifest)
    mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatch:
        raise RuntimeError(f"V20 pool component contract mismatch: {mismatch}")
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
        "high_root_range": [27208, 27655], "direct_root_range": [27656, 28039],
        "high_pair_count": HIGH_PAIR_COUNT, "direct_pair_count": DIRECT_PAIR_COUNT,
        "route_contract": "causal_visual_motion_norm_ge_0p02_routes_history_else_action_only",
        "equal_budget_control": "causal_visual_motion_norm_lt_0p02_routes_history_else_action_only",
    }
    payload = {
        **expected,
        "high_manifest_sha256": _sha256(HIGH / "velocity_matched_manifest.json"),
        "direct_manifest_sha256": _sha256(OUTPUT / "velocity_matched_manifest.json"),
        "states_sha256": _sha256(OUTPUT / "states.pth"), "components": {"high": high, "direct": direct},
        "claim_boundary": "State-pool construction only; no fit, calibration outcome, or C2 claim.",
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("existing V20 mixed-pool manifest differs from its frozen construction")
    else:
        path.write_text(encoded, encoding="utf-8")
    return payload


def main() -> int:
    print(json.dumps(build(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
