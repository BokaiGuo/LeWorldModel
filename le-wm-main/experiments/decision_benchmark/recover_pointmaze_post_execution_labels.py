"""Recover missing post-execution PointMaze labels from a frozen outcome context.

This migration tool is for a completed panel whose report predates the
``post_execution_outcome_labels`` serialization contract.  It refuses to
modify the original report and emits a separately hashed sidecar.  The
deterministic MuJoCo replay occurs only after the original outcome report
exists, so it cannot affect scorer fitting, routing, or authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.gdd_wm.run_pointmaze_planning_shaped_control import _fresh_terminal
from experiments.gdd_wm.run_pointmaze_planning_shaped_repairability import _post_execution_labels


SCHEMA = "pointmaze_post_execution_label_recovery.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*, context: Path, completed_report: Path, output: Path) -> dict[str, Any]:
    """Write identity-bound labels for a report whose execution already completed."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    report = _load(completed_report)
    if report.get("status") != "completed_execution_grounded_repairability_control":
        raise ValueError("completed_report does not certify a completed repairability execution")
    if isinstance(report.get("post_execution_outcome_labels"), dict):
        raise ValueError("completed_report already embeds outcome labels; recovery is unnecessary")
    payload = _load(context)
    rows = [dict(row) for row in payload.get("records", []) if str(row.get("split")) == "outcome"]
    if not rows:
        raise ValueError("context has no outcome rows")
    labels = _post_execution_labels(rows, _fresh_terminal(rows))
    result = {
        "schema": SCHEMA,
        "status": "completed_post_execution_label_recovery",
        "labels": labels,
        "provenance": {
            "context": str(context.resolve()), "context_sha256": _sha256(context),
            "completed_report": str(completed_report.resolve()), "completed_report_sha256": _sha256(completed_report),
        },
        "claim_boundary": "post-execution deterministic label recovery for a completed legacy PointMaze report; not a new experiment, deployment-time signal, or model-selection input.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--completed-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"status": result["status"], "label_count": len(result["labels"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
