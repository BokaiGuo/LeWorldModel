"""Run the root-disjoint PointMaze external-evidence evaluator over a panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_pointmaze_adapter_external_evidence import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--external-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-roots", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    checkpoint_root = args.panel_root / "checkpoints"
    export_root = args.panel_root / "exports"
    checkpoints = sorted(checkpoint_root.glob("*/seed_*/stage_*.pt"))
    if not checkpoints:
        raise ValueError(f"no adapter checkpoints under {checkpoint_root}")
    reports, failures = [], []
    for checkpoint in checkpoints:
        base = checkpoint.parent.parent.name
        seed = checkpoint.parent.name.removeprefix("seed_")
        stage = checkpoint.stem.removeprefix("stage_")
        export = export_root / f"{base}_seed{seed}_stage{stage}"
        fit_export = export / "fit.json"
        outcome_export = export / "outcome.json"
        target = args.output / base / f"seed_{seed}" / f"stage_{stage}"
        evidence_path = target / "external_evidence.json"
        if args.resume and evidence_path.exists():
            try:
                reports.append(json.loads(evidence_path.read_text()))
                continue
            except json.JSONDecodeError:
                pass
        try:
            reports.append(evaluate(
                checkpoint,
                fit_export,
                outcome_export,
                args.external_export,
                target,
                device=args.device,
                max_roots=args.max_roots,
            ))
        except Exception as exc:  # keep the panel audit complete and visible
            failures.append({"checkpoint": str(checkpoint), "error": repr(exc)})
    manifest = {
        "schema": "pointmaze_external_evidence_panel.v2",
        "reports": [str(Path(item["protocol"].get("evidence_path", ""))) for item in reports],
        "report_count": len(reports),
        "failure_count": len(failures),
        "failures": failures,
        "protocol": {
            "external_export": str(args.external_export.resolve()),
            "root_disjoint": True,
            "action_intervention": "+a/-a/zero diagonal fan repeated over four five-step blocks",
            "horizon_alignment": "simulator labels and predictor outputs at blocks 1, 2, and 4",
            "simulator_labels": "released legacy PointMaze XML replayed with modern mujoco, offline only",
            "outcome_used_for_selection": False,
        },
        "summary": [
            {
                "checkpoint": item["checkpoint"],
                "architecture": item["architecture"],
                "seed": item["seed"],
                "training_stage": item["training_stage"],
                "paired_roots": item["action_risk"]["paired_roots"],
                "action_inverse_risk": item["action_risk"]["action_inverse_risk"],
                "action_identifiability": item["action_risk"]["action_identifiability"],
                "planning_risk": item["planning_consequence"]["planning_risk"],
            }
            for item in reports
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    # The evidence path is produced by the evaluator, rather than inferred
    # from architecture labels (which include an environment prefix).
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report_count": len(reports), "failure_count": len(failures)}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
