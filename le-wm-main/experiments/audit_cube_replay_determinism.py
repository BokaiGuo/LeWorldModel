"""Audit deterministic replay of RGB cube action sequences.

This diagnostic does not train or rank actions.  It repeats the exact same
root initialization and action sequence and compares RGB-derived feature
traces.  It is required before interpreting small held-root ranking changes
as learned-model effects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_cube_rgb_visual_servo import _capture_initial
from experiments.evaluate_cube_support_preserving_two_pulse import _run_sequence


def _restore_arrays(value: Any) -> Any:
    """Restore JSON-serialized numeric arrays expected by CubeEnv."""
    if isinstance(value, dict):
        return {key: _restore_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        array = np.asarray(value)
        if array.dtype != object and array.ndim > 0:
            return array
        return [_restore_arrays(item) for item in value]
    return value


def _max_pairwise_difference(values: list[np.ndarray]) -> float:
    if len(values) < 2:
        return 0.0
    reference = np.asarray(values[0], dtype=np.float64)
    return float(max(np.max(np.abs(np.asarray(value, dtype=np.float64) - reference)) for value in values[1:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--initials-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pulse-steps", type=int, default=4)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda":
        require_cuda_allocation("cuda")
    summary = json.loads(args.summary_json.read_text())
    protocol = summary["protocol"]
    mapping_data = (protocol.get("approach_mapping") or {}).get("mapping")
    pixel_mapping = np.asarray(mapping_data, dtype=np.float32) if mapping_data is not None else None
    if args.initials_json is None:
        seeds = [int(value) for value in summary["test_root_seeds"]]
        roots = [
            _capture_initial(seed, int(args.tasks[index % len(args.tasks)]))
            for index, seed in enumerate(seeds)
        ]
    else:
        manifest = json.loads(args.initials_json.read_text())
        roots = [_restore_arrays(root) for root in manifest["roots"]]
        seeds = [int(root["seed"]) for root in roots]
    sequences = {
        "zero": np.zeros((2, 2), dtype=np.float32),
        "direct_diagonal": np.asarray([[0.35355339, 0.35355339]] * 2, dtype=np.float32),
        "axis_composed": np.asarray([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32),
    }
    rows: list[dict[str, Any]] = []
    for index, (seed, initial) in enumerate(zip(seeds, roots)):
        task = int(initial["task_id"])
        for name, sequence in sequences.items():
            source_values: list[np.ndarray] = []
            successor_values: list[np.ndarray] = []
            intermediate_values: list[np.ndarray] = []
            for _repeat in range(int(args.repeats)):
                source, successor, intermediate = _run_sequence(
                    initial, sequence,
                    descent_steps=int(protocol["descent_steps"]),
                    pulse_steps=int(args.pulse_steps),
                    approach_steps=int(protocol["approach_steps"]),
                    close_steps=int(protocol["close_steps"]),
                    approach_gain=float(protocol["approach_gain"]),
                    pixel_mapping=pixel_mapping,
                )
                source_values.append(source)
                successor_values.append(successor)
                if intermediate:
                    intermediate_values.append(intermediate[0])
            rows.append({
                "seed": seed,
                "task": task,
                "sequence": name,
                "repeats": int(args.repeats),
                "source_max_abs_diff": _max_pairwise_difference(source_values),
                "successor_max_abs_diff": _max_pairwise_difference(successor_values),
                "intermediate_max_abs_diff": _max_pairwise_difference(intermediate_values),
                "source_features": np.asarray(source_values[0], dtype=np.float32).tolist(),
                "successor_features": np.asarray(successor_values[0], dtype=np.float32).tolist(),
                "intermediate_features": (
                    np.asarray(intermediate_values[0], dtype=np.float32).tolist()
                    if intermediate_values else None
                ),
            })
    all_differences = np.asarray([
        max(row["source_max_abs_diff"], row["successor_max_abs_diff"], row["intermediate_max_abs_diff"])
        for row in rows
    ], dtype=np.float64)
    report = {
        "schema": "cube_replay_determinism.v1",
        "protocol": {
            "root_disjoint": True,
            "repeats_per_root_sequence": int(args.repeats),
            "feature_source": "RGB front/side geometry chart",
            "privileged_state_used_by_controller": False,
        },
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "max_abs_feature_difference": float(np.max(all_differences)) if len(all_differences) else 0.0,
            "median_max_abs_feature_difference": float(np.median(all_differences)) if len(all_differences) else 0.0,
            "deterministic_at_float_tolerance": bool(np.all(all_differences <= 1.0e-7)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
