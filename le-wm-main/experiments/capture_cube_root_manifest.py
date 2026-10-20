"""Capture reusable Cube root initial conditions for cross-process replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.run_cube_rgb_visual_servo import _capture_initial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [
        _capture_initial(int(seed), int(args.tasks[index % len(args.tasks)]))
        for index, seed in enumerate(args.seeds)
    ]
    payload = {
        "schema": "cube_root_manifest.v1",
        "root_count": len(roots),
        "roots": roots,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, default=lambda value: value.tolist()) + "\n"
    )
    print(json.dumps({"schema": payload["schema"], "root_count": len(roots)}))


if __name__ == "__main__":
    main()
