"""Sweep RGB-only approach controls before action-effect probing.

This audit asks whether the low support coverage in the Cube transfer panel is
caused by a poor RGB approach prelude.  It uses only rendered front/side RGB
features at the decision point; simulator contact/state labels are not read.
Calibration roots for the pixel-to-action Jacobian are disjoint from the roots
used by the sweep.  The output is a routing diagnostic, not a planning claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.audit_cube_rgb_observability import _make_env
from experiments.run_cube_rgb_visual_servo import _capture_initial, _reset
from experiments.run_live_cube_rgb_macro_ranker import _fit_rgb_pixel_action_mapping
from experiments.train_cube_multiview_spatial_adapter import _joint_features, _prepare_contact


def _distance_summary(values: list[float], threshold: float) -> dict[str, object]:
    distances = np.asarray(values, dtype=np.float64)
    return {
        "root_count": int(len(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p90": float(np.quantile(distances, 0.9)),
        "eligible_count": int(np.sum(distances <= float(threshold))),
        "eligible_rate": float(np.mean(distances <= float(threshold))),
        "threshold": float(threshold),
        "values": distances.tolist(),
    }


def run(
    *, output: Path, seed: int, roots: int, tasks: list[int], descent_steps: int,
    approach_steps: list[int], gains: list[float], close_steps: int,
    calibration_roots: int, probe_steps: int, threshold: float,
) -> dict[str, object]:
    if roots < 1 or calibration_roots < 1 or not tasks:
        raise ValueError("roots, calibration_roots, and tasks must be non-empty")
    if any(step < 0 for step in approach_steps) or close_steps < 0:
        raise ValueError("approach and close steps must be non-negative")
    if any(not np.isfinite(gain) or not 0.0 < gain <= 1.0 for gain in gains):
        raise ValueError("approach gains must lie in (0, 1]")
    if threshold < 0.0 or not np.isfinite(threshold):
        raise ValueError("threshold must be finite and non-negative")

    print("[purpose] RGB-only approach sweep: measure whether pre-contact routing can recover support.", flush=True)
    mapping, mapping_report = _fit_rgb_pixel_action_mapping(
        seed=int(seed) + 100000,
        tasks=[int(task) for task in tasks],
        roots=int(calibration_roots),
        descent_steps=int(descent_steps),
        probe_steps=int(probe_steps),
    )
    initials = [
        _capture_initial(int(seed) + index, int(tasks[index % len(tasks)]))
        for index in range(int(roots))
    ]
    rows: list[dict[str, object]] = []
    for steps in approach_steps:
        for gain in gains:
            distances: list[float] = []
            side_distances: list[float] = []
            for initial in initials:
                env = _make_env()
                try:
                    obs, _info, _goal = _reset(env, initial)
                    obs = _prepare_contact(
                        env, obs, descent_steps=int(descent_steps),
                        approach_steps=int(steps), close_steps=int(close_steps),
                        approach_gain=float(gain), pixel_action_mapping=mapping,
                    )
                    features = _joint_features(env, np.asarray(obs, dtype=np.uint8))
                    distances.append(float(np.linalg.norm(features[:2] - features[2:4])))
                    side_distances.append(float(np.linalg.norm(features[5:7] - features[7:9])))
                finally:
                    env.close()
            rows.append({
                "approach_steps": int(steps),
                "approach_gain": float(gain),
                "front_distance": _distance_summary(distances, threshold),
                "side_distance": _distance_summary(side_distances, threshold),
            })
            print(
                f"steps={steps} gain={gain:.3g} "
                f"front_mean={np.mean(distances):.5f} "
                f"eligible={np.mean(np.asarray(distances) <= threshold):.3f}",
                flush=True,
            )
    payload = {
        "schema": "cube_rgb_approach_sweep.v1",
        "claim_boundary": "RGB-only contact-proxy routing diagnostic; no contact-label or planning claim",
        "protocol": {
            "seed": int(seed), "roots": int(roots), "tasks": [int(task) for task in tasks],
            "root_disjoint": True,
            "calibration_seed_offset": 100000,
            "calibration_roots": int(calibration_roots),
            "descent_steps": int(descent_steps), "close_steps": int(close_steps),
            "probe_steps": int(probe_steps), "eligibility_threshold": float(threshold),
            "features": "front/side RGB object and effector centroids plus object area",
        },
        "pixel_action_mapping": mapping_report,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261925)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--descent-steps", type=int, default=15)
    parser.add_argument("--approach-steps", type=int, nargs="+", default=[0, 8, 16, 24, 32])
    parser.add_argument("--gains", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--close-steps", type=int, default=5)
    parser.add_argument("--calibration-roots", type=int, default=8)
    parser.add_argument("--probe-steps", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()
    payload = run(
        output=args.output, seed=args.seed, roots=args.roots, tasks=args.tasks,
        descent_steps=args.descent_steps, approach_steps=args.approach_steps,
        gains=args.gains, close_steps=args.close_steps,
        calibration_roots=args.calibration_roots, probe_steps=args.probe_steps,
        threshold=args.threshold,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
