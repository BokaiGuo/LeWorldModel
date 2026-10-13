"""Run the external-evidence evaluator over the complete learned snapshot panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.empirical_spectral_audit.evaluate_learned_snapshot_evidence import evaluate_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--planning-seed", type=int, default=20261013)
    parser.add_argument("--planning-episodes", type=int, default=12)
    parser.add_argument("--planning-steps", type=int, default=100)
    parser.add_argument("--cem-samples", type=int, default=96)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    snapshots = sorted(args.checkpoint_root.glob("*/seed_*/snapshots/epoch_*.pt"))
    if not snapshots:
        raise ValueError(f"no snapshots found below {args.checkpoint_root}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for snapshot in snapshots:
        relative = snapshot.relative_to(args.checkpoint_root)
        target = args.output / relative.parent / snapshot.stem
        evidence = target / "external_evidence.json"
        if args.resume and evidence.exists():
            try:
                report = json.loads(evidence.read_text())
                rows.append({
                    "checkpoint": report["checkpoint"],
                    "architecture": report["architecture"],
                    "seed": report["seed"],
                    "training_stage": report["training_stage"],
                    "action_inverse_risk": report["action_risk"]["action_inverse_risk"],
                    "action_identifiability": report["action_risk"]["action_identifiability"],
                    "planning_risk": report["planning_consequence"]["planning_risk"],
                    "strict_success_rate": report["planning_consequence"]["strict_success_rate"],
                    "evidence": str(evidence),
                    "status": "resumed",
                })
                continue
            except (KeyError, json.JSONDecodeError):
                pass
        try:
            report = evaluate_snapshot(
                snapshot,
                target,
                device=args.device,
                planning_seed=args.planning_seed,
                planning_episodes=args.planning_episodes,
                planning_steps=args.planning_steps,
                cem_samples=args.cem_samples,
                cem_iterations=args.cem_iterations,
            )
            rows.append({
                "checkpoint": report["checkpoint"],
                "architecture": report["architecture"],
                "seed": report["seed"],
                "training_stage": report["training_stage"],
                "action_inverse_risk": report["action_risk"]["action_inverse_risk"],
                "action_identifiability": report["action_risk"]["action_identifiability"],
                "planning_risk": report["planning_consequence"]["planning_risk"],
                "strict_success_rate": report["planning_consequence"]["strict_success_rate"],
                "evidence": str(target / "external_evidence.json"),
                "status": "complete",
            })
        except Exception as exc:  # preserve a resumable failure ledger
            failures.append({"checkpoint": str(snapshot), "error": repr(exc)})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        (args.output / "panel_progress.json").write_text(json.dumps({"rows": rows, "failures": failures}, indent=2) + "\n")
    manifest = {
        "schema": "learned_snapshot_external_evidence_panel.v1",
        "protocol": {
            "checkpoint_count": len(snapshots),
            "completed_count": len(rows),
            "failure_count": len(failures),
            "planning_seed": args.planning_seed,
            "planning_episodes": args.planning_episodes,
            "planning_steps": args.planning_steps,
            "cem_samples": args.cem_samples,
            "cem_iterations": args.cem_iterations,
            "runtime_successor_verification": False,
            "test_planning_used_for_snapshot_selection": False,
        },
        "rows": rows,
        "failures": failures,
        "claim_boundary": (
            "The panel contains independent action-intervention and fresh RGB-only planning evidence for immutable "
            "PushT checkpoints. It is not cross-environment evidence."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["protocol"], indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
