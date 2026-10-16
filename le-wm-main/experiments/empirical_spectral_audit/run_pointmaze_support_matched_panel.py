"""Run the corrected PointMaze support-matched panel on CUDA.

The input panel is the previously frozen 2-architecture x 5-seed x 5-stage
checkpoint list.  This runner changes only the audited replay protocol: the
repaired external export and the five-physics-step PointMaze block clock.
Checkpoint, split, and fit/outcome artifacts remain immutable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .evaluate_pointmaze_support_matched_external_evidence import evaluate
except ImportError:  # direct-script compatibility
    from evaluate_pointmaze_support_matched_external_evidence import evaluate


def _fit_outcome_for(checkpoint: Path) -> tuple[Path, Path]:
    stage = checkpoint.stem.removeprefix("stage_")
    adapter_dir = checkpoint.parent.parent.name
    seed = checkpoint.parent.name.removeprefix("seed_")
    export_root = checkpoint.parents[3] / "exports"
    stem = f"{adapter_dir}_seed{seed}_stage{stage}"
    export_dir = export_root / stem
    return export_dir / "fit.json", export_dir / "outcome.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--external-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-roots", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

    checkpoints = sorted(args.checkpoint_root.glob("*/seed_*/stage_*.pt"))
    if not checkpoints:
        raise ValueError(f"no checkpoints found below {args.checkpoint_root}")
    reports = []
    failures = []
    for checkpoint in checkpoints:
        fit_export, outcome_export = _fit_outcome_for(checkpoint)
        report_dir = args.output / checkpoint.parent.parent.name / checkpoint.parent.name / checkpoint.stem
        try:
            if not fit_export.exists() or not outcome_export.exists():
                raise FileNotFoundError(f"missing fit/outcome export for {checkpoint}")
            report = evaluate(
                checkpoint,
                fit_export,
                outcome_export,
                args.external_export,
                report_dir,
                device=args.device,
                sequence_scale=args.sequence_scale,
                max_roots=args.max_roots,
            )
            reports.append(str((report_dir / "support_matched_external_evidence.json").resolve()))
            print(json.dumps({
                "checkpoint": str(checkpoint),
                "action_inverse_risk": report["action_risk"]["action_inverse_risk"],
                "planning_risk": report["planning_consequence"]["planning_risk"],
            }))
        except Exception as exc:  # preserve a complete panel audit
            failures.append({"checkpoint": str(checkpoint), "error": repr(exc)})
            print(json.dumps(failures[-1]))
    manifest = {
        "schema": "pointmaze_support_matched_external_panel.v2",
        "reports": reports,
        "failures": failures,
        "protocol": {
            "checkpoint_root": str(args.checkpoint_root.resolve()),
            "external_export": str(args.external_export.resolve()),
            "checkpoint_immutable": True,
            "root_disjoint": True,
            "device": args.device,
            "sequence_scale": args.sequence_scale,
            "max_roots": args.max_roots,
            "steps_per_action": 5,
            "replay_labels_repaired": True,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"reports": len(reports), "failures": len(failures)}))


if __name__ == "__main__":
    main()
