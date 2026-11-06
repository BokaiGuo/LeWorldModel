"""Reconcile equal-slot cost fields from the saved prediction call logs."""
from __future__ import annotations

import json
from pathlib import Path

from experiments import pc_retention_equal_slot_cost_dev_20261104_codex as panel


OUT = panel.OUT / "cost_ledger_reconciled_20261104.json"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source_audit = json.loads((panel.OUT / "audit.json").read_text())
    if source_audit["status"] != "passed" or len(source_audit["rows"]) != 80:
        raise ValueError("source audit incomplete")
    rows = []
    for root, _task in enumerate(panel.tasks()):
        for arm in panel.ARMS:
            record = json.loads((panel.OUT / f"plan_{root:02d}_{arm}.json").read_text())
            calls = list(
                panel.prior.old.previous.fixed.read_calls(
                    panel.OUT / f"plan_{root:02d}_{arm}_calls"
                )
            )
            full_samples = sum(len(actions) for actions, _values, _meta in calls if actions.shape[1] == 15)
            resource = record["result"].get("resource") or {}
            rows.append(
                {
                    "root": root,
                    "arm": arm,
                    "prediction_tokens": record["tokens"],
                    "prediction_calls_from_journal": len(calls),
                    "full_samples_from_journal": full_samples,
                    "saved_full_samples": resource.get("full_samples"),
                    "saved_inner_wall_seconds": resource.get("wall_seconds"),
                    "outer_elapsed_seconds": record["elapsed_seconds"],
                    "timing_comparable": False,
                    "full_sample_count_reconciled": full_samples
                    == resource.get("full_samples", full_samples),
                }
            )
    result = {
        "schema": "pc_retention_cost_reconciliation.v1",
        "status": "passed",
        "rows": rows,
        "full_sample_definition": "all actions in prediction journal with horizon 15, including final paired scores",
        "timing_definition": "outer elapsed is retained; saved inner planner timing is not cross-arm comparable",
        "gpu_or_memory_measured": False,
        "full_cost_certified": False,
        "scope": "saved equal-slot development panel; read-only reconciliation",
    }
    panel.prior.journal.write_json(OUT, result)
    print(json.dumps({
        "status": result["status"],
        "rows": len(rows),
        "full_sample_mismatches": sum(not row["full_sample_count_reconciled"] for row in rows),
        "timing_comparable": False,
        "full_cost_certified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
