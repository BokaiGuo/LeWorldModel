"""Resumable GPU queue for the pre-registered task-metric variant audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


VARIANTS = ("scalar_legacy", "phase_legacy", "mode_uncertain")
SEEDS = (8411, 8412, 8413, 8414, 8415)
SNAPSHOTS = (10, 20, 30, 50, 75, 100)


def complete(path: Path) -> bool:
    target = path / "checkpoint_record.json"
    try:
        payload = json.loads(target.read_text())
        diagnostics = payload["record"]["diagnostics"]
        variants = diagnostics["task_metric_variants"]
        required = {
            "default_task_space", "agent_weight_low", "agent_weight_high",
            "block_xy_only", "orientation_only", "block_xy_plus_orientation", "action_effect",
            "stable_block_xy_focus", "stable_orientation_focus", "stable_block_xy_plus_orientation",
            "stable_action_effect",
        }
        return required.issubset(variants) and all(
            "task_metric_schur_offdiagonal_energy" in variants[name] for name in required
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def command(root: Path, output: Path, variant: str, seed: int, snapshot: int) -> tuple[list[str], Path]:
    checkpoint = root / "strong_claim_protocol_v2_20261009" / "checkpoints" / variant / f"seed_{seed}" / "snapshots" / f"epoch_{snapshot:04d}.pt"
    target = output / "task_metric_records" / variant / f"seed_{seed}" / f"epoch_{snapshot:04d}"
    args = [
        sys.executable, "experiments/audit_causal_visual_checkpoint.py",
        "--checkpoint", str(checkpoint),
        "--checkpoint-label", f"{variant}/seed_{seed}/epoch_{snapshot}",
        "--group", f"PushT/in_distribution/{variant}/seed_{seed}",
        "--output", str(target),
        "--diagnostic-roots", "240", "--planning-episodes", "32",
        "--planning-steps", "100", "--planning-horizon", "6",
        "--cem-samples", "192", "--cem-iterations", "4", "--device", "cuda",
    ]
    return args, target


def run_one(item: tuple[list[str], Path]) -> tuple[str, int, str]:
    args, target = item
    target.mkdir(parents=True, exist_ok=True)
    log = target / "task_metric_variant_audit.log"
    with log.open("w") as stream:
        result = subprocess.run(args, cwd=Path(__file__).resolve().parents[1], stdout=stream, stderr=subprocess.STDOUT, check=False)
    return str(target), int(result.returncode), str(log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output-name", default="alignment_universal_selection_metric_variants_20261011")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.root / args.output_name
    jobs = []
    skipped = 0
    for variant in VARIANTS:
        for seed in SEEDS:
            for snapshot in SNAPSHOTS:
                item = command(args.root, output, variant, seed, snapshot)
                if complete(item[1]):
                    skipped += 1
                else:
                    jobs.append(item)
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_one, item) for item in jobs]
        for future in as_completed(futures):
            target, returncode, log = future.result()
            print(json.dumps({"target": target, "returncode": returncode, "log": log}), flush=True)
            if returncode:
                failures.append((target, returncode, log))
    summary = {"total": 90, "skipped": skipped, "run": len(jobs), "failures": failures, "output": str(output)}
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
