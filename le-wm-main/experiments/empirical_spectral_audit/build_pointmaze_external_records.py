"""Join PointMaze adapter exports with root-disjoint external evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_learned_model_selection_records import build_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    evidence_paths = sorted(args.evidence_root.rglob("external_evidence.json"))
    if not evidence_paths:
        raise ValueError(f"no external evidence under {args.evidence_root}")
    records, failures = [], []
    for index, evidence_path in enumerate(evidence_paths):
        evidence = json.loads(evidence_path.read_text())
        checkpoint = Path(evidence["checkpoint"])
        base = checkpoint.parent.parent.name
        seed = checkpoint.parent.name.removeprefix("seed_")
        stage = checkpoint.stem.removeprefix("stage_")
        export = checkpoint.parents[3] / "exports" / f"{base}_seed{seed}_stage{stage}"
        fit = export / "fit.json"
        outcome = export / "outcome.json"
        target = args.output / f"candidate_{index:03d}" / "checkpoint_record.json"
        if args.resume and target.exists():
            try:
                json.loads(target.read_text())
                records.append(str(target))
                continue
            except json.JSONDecodeError:
                pass
        try:
            record = build_record(fit, outcome, external_evidence_path=evidence_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            # build_record already returns the canonical outer
            # real_checkpoint_selection.v1 payload.  Do not wrap it again:
            # the benchmark loader expects diagnostics at payload["record"].
            target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            records.append(str(target))
        except Exception as exc:
            failures.append({"evidence": str(evidence_path), "fit": str(fit), "outcome": str(outcome), "error": repr(exc)})
    manifest = {
        "schema": "pointmaze_external_records.v1",
        "records": records,
        "failures": failures,
        "protocol": {"independent_action_intervention": True, "independent_pointmaze_progress": True},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record_count": len(records), "failure_count": len(failures)}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
