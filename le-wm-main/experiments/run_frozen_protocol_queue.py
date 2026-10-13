"""Resume a pre-frozen strong-claim protocol without changing its commands.

This runner is intentionally a queue, not a sweep generator: it consumes an
existing manifest, verifies completion from declared artifacts, and executes
only missing immutable commands in their manifest order.  It cannot add a
seed, checkpoint epoch, selector, or metric after outcomes are observed.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def arguments(command: str) -> list[str]:
    values = shlex.split(command)
    if len(values) < 2 or values[1] not in {
        "experiments/run_causal_visual_pusht_world_model.py",
        "experiments/audit_causal_visual_checkpoint.py",
        "experiments/audit_visual_causal_interventions.py",
    }:
        raise ValueError(f"manifest command is outside the frozen protocol surface: {command!r}")
    return values


def value_after(values: list[str], flag: str) -> str:
    try:
        return values[values.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"missing {flag} in frozen command") from exc


def complete(values: list[str], stage: str) -> bool:
    output = Path(value_after(values, "--output"))
    if stage == "training":
        snapshots_start = values.index("--snapshot-epochs") + 1
        snapshots: list[int] = []
        for token in values[snapshots_start:]:
            if token.startswith("--"):
                break
            snapshots.append(int(token))
        required = [output / "causal_visual_world_model.pt", output / "split_manifest.json"]
        required.extend(output / "snapshots" / f"epoch_{epoch:04d}.pt" for epoch in snapshots)
        return all(path.is_file() and path.stat().st_size > 0 for path in required)
    if stage == "selection":
        target = output / "checkpoint_record.json"
        if not (target.is_file() and target.stat().st_size > 0):
            return False
        try:
            payload = json.loads(target.read_text())
            protocol = payload["protocol"]
            interface = protocol["interface"]
            quality = protocol["prediction_quality"]
            return (
                payload.get("record_schema") == "real_checkpoint_selection.v1"
                and "direct_utility" in interface
                and quality.get("metric") == "planner_matched_open_loop_rollout_mse"
                and quality.get("diagnostic_roots_disjoint_from_planning_test") is True
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
    elif stage == "causal_ood":
        target = output / "causal_intervention_audit.json"
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return target.is_file() and target.stat().st_size > 0


def inputs_ready(values: list[str], stage: str) -> bool:
    """Fail closed if an ordered downstream stage lacks its frozen checkpoint."""
    if stage not in {"selection", "causal_ood"}:
        return True
    checkpoint = Path(value_after(values, "--checkpoint"))
    return checkpoint.is_file() and checkpoint.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("training", "selection", "causal_ood"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-runs", type=int, default=0, help="0 executes every missing command")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_runs < 0:
        raise ValueError("max-runs must be nonnegative")
    manifest = json.loads(args.manifest.read_text())
    commands = manifest.get("commands", {}).get(args.stage)
    if not isinstance(commands, list) or not commands:
        raise ValueError(f"manifest has no commands for stage {args.stage!r}")
    manifest_hash = manifest.get("sha256_before_execution")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise ValueError("manifest is missing its pre-execution SHA-256")
    log_dir = args.manifest.parent / "queue_logs" / args.stage
    pending: list[tuple[int, list[str]]] = []
    for index, command in enumerate(commands):
        values = arguments(command)
        if not complete(values, args.stage):
            pending.append((index, values))
    limit = len(pending) if args.max_runs == 0 else min(args.max_runs, len(pending))
    print(json.dumps({
        "stage": args.stage, "manifest_sha256": manifest_hash, "total": len(commands),
        "completed": len(commands) - len(pending), "pending": len(pending), "scheduled": limit,
        "dry_run": bool(args.dry_run),
    }, sort_keys=True), flush=True)
    if args.dry_run:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    for index, values in pending[:limit]:
        if not inputs_ready(values, args.stage):
            raise RuntimeError(
                f"frozen {args.stage} command {index} is blocked because its checkpoint is missing; "
                "do not substitute another checkpoint or skip the unit"
            )
        log_path = log_dir / f"command_{index:03d}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write("$ " + shlex.join(values) + "\n")
            log.flush()
            outcome = subprocess.run(values, cwd=args.repo_root, stdout=log, stderr=subprocess.STDOUT, check=False)
            log.write(f"[exit={outcome.returncode}]\n")
        if outcome.returncode != 0:
            raise SystemExit(outcome.returncode)


if __name__ == "__main__":
    main()
