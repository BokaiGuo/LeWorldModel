"""Run the locked Cube ceiling + RGB panel + failure-cause bundle.

The bundle is the executable protocol boundary for the next GPU run.  It
locks the seed, task schedule, held-out root count, and output directory for
all three artifacts.  It refuses CPU fallback before producing partial
scientific output when CUDA is requested.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    print("[cube-bundle]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _resume_if_partial(path: Path) -> list[str]:
    """Return ``--resume`` only for a known partial artifact.

    A completed artifact must be rerun under an explicitly new output
    directory; passing ``--resume`` to a completed ceiling would correctly
    reject it rather than silently mixing protocols.
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return ["--resume"] if '"schema": "cube_planner_ceiling.v2.partial"' in text else []


def _panel_resume_if_partial(path: Path) -> list[str]:
    if not path.exists():
        return []
    for artifact in path.glob("*.json"):
        try:
            if '"schema": "live_cube_rgb_macro_ranker.v1"' in artifact.read_text(encoding="utf-8"):
                return ["--resume"]
        except OSError:
            continue
    return []


def build_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    python = sys.executable
    output = Path(args.output)
    tasks = [str(int(task)) for task in args.tasks]
    ceiling = [
        python, "-m", "experiments.audit_cube_planner_ceiling",
        "--output", str(output / "ceiling.json"),
        "--seed", str(args.seed),
        "--test-roots", str(args.test_roots),
        "--macro-length", str(args.macro_length),
        "--expanded-macro-length", str(args.expanded_macro_length),
        "--goal-tolerance-m", str(args.goal_tolerance_m),
        "--descent-steps", str(args.descent_steps),
        "--approach-steps", str(args.approach_steps),
        "--approach-calibration-roots", str(args.calibration_roots),
        "--probe-steps", str(args.probe_steps),
        "--macro-cycles", str(args.macro_cycles),
        "--tasks", *tasks,
    ]
    panel = [
        python, "-m", "experiments.run_live_cube_rgb_macro_panel",
        "--output", str(output / "panel"),
        "--device", args.device,
        "--seed", str(args.seed),
        "--calibration-roots", str(args.calibration_roots),
        "--test-roots", str(args.test_roots),
        "--macro-cycles", str(args.macro_cycles),
        "--action-block", str(args.action_block),
        "--approach-steps", str(args.approach_steps),
        "--descent-steps", str(args.descent_steps),
        "--probe-steps", str(args.probe_steps),
        "--tasks", *tasks,
    ]
    ceiling.extend(_resume_if_partial(output / "ceiling.json"))
    panel.extend(_panel_resume_if_partial(output / "panel"))
    cause = [
        python, "-m", "experiments.summarize_cube_failure_cause",
        "--ceiling", str(output / "ceiling.json"),
        "--panel", str(output / "panel" / "panel.json"),
        "--output", str(output / "failure_cause.json"),
    ]
    return {"ceiling": ceiling, "panel": panel, "failure_cause": cause}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261900)
    parser.add_argument("--calibration-roots", type=int, default=40)
    parser.add_argument("--test-roots", type=int, default=8)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--macro-length", type=int, default=20)
    parser.add_argument("--expanded-macro-length", type=int, default=40)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.05)
    parser.add_argument("--macro-cycles", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=1)
    parser.add_argument("--approach-steps", type=int, default=8)
    parser.add_argument("--descent-steps", type=int, default=15)
    parser.add_argument("--probe-steps", type=int, default=4)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; bundle refuses CPU fallback")
    if args.calibration_roots < 8 or args.test_roots < 1:
        raise ValueError("calibration-roots must be >=8 and test-roots must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    commands = build_commands(args)
    manifest = {
        "schema": "cube_diagnostic_bundle.v1",
        "protocol": {
            "seed": args.seed,
            "tasks": [int(task) for task in args.tasks],
            "calibration_roots": args.calibration_roots,
            "test_roots": args.test_roots,
            "device": args.device,
            "root_schedule": "seed + 1000 + index for held-out roots",
            "claim_boundary": "ceiling and paired RGB diagnostics; not universal planning evidence",
        },
        "commands": commands,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    _run(commands["ceiling"])
    _run(commands["panel"])
    _run(commands["failure_cause"])
    manifest["status"] = "complete"
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
