"""Join snapshot diagnostic exports with independent external evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.empirical_spectral_audit.build_learned_model_selection_records import build_record


def _key(path: Path) -> tuple[str, str, str]:
    match = re.match(r"(.+)_seed(\d+)_stage(\d+)_(?:fit|outcome)(?:_pool256)?\.json$", path.name)
    if match is None:
        raise ValueError(f"cannot parse snapshot export name: {path}")
    return match.group(1), match.group(2), str(int(match.group(3)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-suffix", default="_fit_pool256.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    fit_paths = sorted(args.export_root.glob(f"*{args.fit_suffix}"))
    if not fit_paths:
        raise ValueError(f"no fit exports under {args.export_root}")
    outcome_paths = {}
    for path in args.export_root.glob("*_outcome.json"):
        outcome_paths[_key(path)] = path
    evidence_paths = {}
    for path in args.evidence_root.glob("*/seed_*/snapshots/epoch_*/external_evidence.json"):
        payload = json.loads(path.read_text())
        key = (payload["architecture"], str(payload["seed"]), str(payload["training_stage"]))
        evidence_paths[key] = path
        # The diagnostic enriched exports retain the checkpoint architecture
        # prefix, while the external evidence names the adapter architecture.
        evidence_paths[(f"causal_visual_{payload['architecture']}", key[1], key[2])] = path
        checkpoint_parts = Path(payload["checkpoint"]).parts
        if "checkpoints" in checkpoint_parts:
            checkpoint_arch = checkpoint_parts[checkpoint_parts.index("checkpoints") + 1]
            evidence_paths[(checkpoint_arch, key[1], key[2])] = path
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    for index, fit_path in enumerate(fit_paths):
        key = _key(fit_path)
        outcome_path = outcome_paths.get(key)
        evidence_path = evidence_paths.get(key)
        target = args.output / f"candidate_{index:03d}" / "checkpoint_record.json"
        if args.resume and target.exists():
            try:
                json.loads(target.read_text())
                records.append(str(target))
                continue
            except json.JSONDecodeError:
                pass
        if outcome_path is None or evidence_path is None:
            failures.append({"fit": str(fit_path), "missing_outcome": outcome_path is None, "missing_evidence": evidence_path is None})
            continue
        try:
            payload = build_record(
                fit_path,
                outcome_path,
                fit_horizon=1,
                outcome_horizon=4,
                external_evidence_path=evidence_path,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            records.append(str(target))
        except Exception as exc:
            failures.append({"fit": str(fit_path), "outcome": str(outcome_path), "evidence": str(evidence_path), "error": repr(exc)})
    manifest = {
        "schema": "real_checkpoint_selection_external_panel.v1",
        "records": records,
        "failures": failures,
        "protocol": {
            "record_count": len(records),
            "failure_count": len(failures),
            "independent_action_risk": True,
            "independent_planning_consequence": True,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["protocol"], indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
