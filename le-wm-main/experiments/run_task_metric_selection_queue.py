"""Run the frozen task-metric checkpoint audit as a resumable parallel queue."""

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
    if not target.is_file():
        return False
    try:
        payload = json.loads(target.read_text())
        record = payload["record"]
        diagnostics = record["diagnostics"]
        factors = record["model_factors"]
        return all(
            name in diagnostics
            for name in (
                "task_metric_alignment",
                "task_metric_commutator",
                "task_metric_transient_amplification",
                "predictor_spectrum",
            )
        ) and factors["latent_dimension"] == 96 and "training_stage" in factors
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def command(root: Path, output: Path, variant: str, seed: int, snapshot: int) -> tuple[list[str], Path]:
    checkpoint = root / "strong_claim_protocol_v2_20261009" / "checkpoints" / variant / f"seed_{seed}" / "snapshots" / f"epoch_{snapshot:04d}.pt"
    target = output / "task_metric_records" / variant / f"seed_{seed}" / f"epoch_{snapshot:04d}"
    args = [
        sys.executable,
        "experiments/audit_causal_visual_checkpoint.py",
        "--checkpoint", str(checkpoint),
        "--checkpoint-label", f"{variant}/seed_{seed}/epoch_{snapshot}",
        "--group", f"PushT/in_distribution/{variant}/seed_{seed}",
        "--output", str(target),
        "--diagnostic-roots", "240",
        "--planning-episodes", "32",
        "--planning-steps", "100",
        "--planning-horizon", "6",
        "--cem-samples", "192",
        "--cem-iterations", "4",
        "--device", "cuda",
    ]
    return args, target


def run_one(item: tuple[list[str], Path]) -> tuple[str, int, str]:
    args, target = item
    target.mkdir(parents=True, exist_ok=True)
    log = target / "task_metric_audit.log"
    with log.open("w") as stream:
        result = subprocess.run(args, cwd=Path(__file__).resolve().parents[1], stdout=stream, stderr=subprocess.STDOUT, check=False)
    return str(target), int(result.returncode), str(log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root
    output = root / "alignment_universal_selection_audit_20261011"
    jobs = []
    skipped = 0
    for variant in VARIANTS:
        for seed in SEEDS:
            for snapshot in SNAPSHOTS:
                command_args, target = command(root, output, variant, seed, snapshot)
                if complete(target):
                    skipped += 1
                else:
                    jobs.append((command_args, target))
    failures: list[tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_one, item) for item in jobs]
        for future in as_completed(futures):
            target, returncode, log = future.result()
            print(json.dumps({"target": target, "returncode": returncode, "log": log}), flush=True)
            if returncode != 0:
                failures.append((target, returncode, log))
    summary = {"total": len(VARIANTS) * len(SEEDS) * len(SNAPSHOTS), "skipped": skipped, "run": len(jobs), "failures": failures}
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
