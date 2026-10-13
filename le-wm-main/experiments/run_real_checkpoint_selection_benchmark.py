"""Aggregate immutable real-checkpoint records into a held-out selection report.

Inputs are JSON files emitted by ``audit_causal_visual_checkpoint.py``.  The
runner refuses to infer missing fields, because a selection study with mixed
coordinate gauges, planning interfaces, or data splits is not interpretable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .real_checkpoint_selection_protocol import evaluate_records
except ImportError:  # direct script execution
    from real_checkpoint_selection_protocol import evaluate_records  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="append", required=True, type=Path, help="record JSON; repeatable")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20261009)
    parser.add_argument("--mse-relative-tolerance", type=float, default=0.05)
    return parser.parse_args()


def load_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("record_schema") != "real_checkpoint_selection.v1":
        raise ValueError(f"{path} is not a real_checkpoint_selection.v1 record")
    record = payload.get("record")
    if not isinstance(record, dict):
        raise ValueError(f"{path} has no record object")
    required_protocol = {
        "diagnostic_split": "frozen before checkpoint selection",
        "heldout_planning": "disjoint from diagnostic split",
        "runtime_successor_verification": False,
    }
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{path} has no protocol object")
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"{path} violates required protocol field {key!r}: {protocol.get(key)!r}")
    if protocol.get("coordinate_gauge") != "decoded_observable_pose_raw":
        raise ValueError(f"{path} has incompatible coordinate gauge: {protocol.get('coordinate_gauge')!r}")
    return record


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real learned-checkpoint model-selection benchmark",
        "",
        "All selectors use only the declared diagnostic split. Planning loss is held out.",
        "",
        "## Mean held-out planning regret",
        "",
        "| Selector | Mean regret | 95% CI | Groups |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, item in report["summary"]["selection_regret"].items():
        lines.append(
            f"| {name} | {item['mean']:.6g} | [{item['ci95_low']:.6g}, {item['ci95_high']:.6g}] | {item['n_groups']} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        f"{report['claim_boundary']}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = [load_record(path) for path in args.record]
    report = evaluate_records(records, args.bootstrap_seed, args.mse_relative_tolerance)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "selection_report.md").write_text(markdown(report))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
