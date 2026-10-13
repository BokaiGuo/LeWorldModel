"""Build or watch a compact status snapshot for a spectral-transfer run."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="run output directory")
    parser.add_argument("--watch", action="store_true", help="keep updating run_status.json")
    parser.add_argument("--interval", type=float, default=30.0)
    return parser.parse_args()


def count_chunks(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob("chunk_*.npz"))


def last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def build_status(output: Path) -> dict[str, Any]:
    phase1 = last_jsonl(output / "progress.jsonl")
    phase2 = last_jsonl(output / "planning_progress.jsonl")
    chunks = count_chunks(output / "chunks")
    planning_chunks = count_chunks(output / "planning_chunks")
    stage_times = {
        "phase1_latent_audit": mtime(output / "progress.jsonl"),
        "phase2_planning_alignment": mtime(output / "planning_progress.jsonl"),
        "phase3_adapter_intervention": mtime(output / "intervention_adapter.json"),
        "summary_completed": mtime(output / "summary.md"),
    }
    stage = max(stage_times, key=stage_times.get)
    if stage_times[stage] <= 0:
        stage = "not_started"

    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": "observed",
        "output": str(output),
        "phase1": phase1,
        "phase2": phase2,
        "chunks": chunks,
        "planning_chunks": planning_chunks,
        "artifacts": {
            "latents": (output / "latents.npz").exists(),
            "lift_metrics": (output / "lift_metrics.json").exists(),
            "planning_alignment": (output / "planning_alignment.json").exists(),
            "intervention_adapter": (output / "intervention_adapter.json").exists(),
            "summary": (output / "summary.md").exists(),
        },
    }


def write_status(output: Path) -> dict[str, Any]:
    status = build_status(output)
    (output / "run_status.json").write_text(json.dumps(status, indent=2, sort_keys=True))
    with (output / "run_events.jsonl").open("a") as f:
        f.write(json.dumps(status, sort_keys=True) + "\n")
    return status


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    while True:
        status = write_status(output)
        print(json.dumps(status, sort_keys=True), flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
