"""Read-only cost-ledger audit for the sealed equal-slot retention panel."""
import json
import statistics
from pathlib import Path

from experiments import pc_retention_equal_slot_cost_dev_20261104_codex as panel


OUTPUT = panel.OUT / "cost_ledger_audit.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    panel.prior.seal.verify(manifest["dependencies"])
    audit = json.loads((panel.OUT / "audit.json").read_text())
    if audit["status"] != "passed" or len(audit["rows"]) != 80:
        raise ValueError("source audit is incomplete")
    rows = []
    for root, task in enumerate(panel.tasks()):
        for arm in panel.ARMS:
            record = json.loads((panel.OUT / f"plan_{root:02d}_{arm}.json").read_text())
            if record["status"] != "completed":
                raise ValueError("failed source plan")
            resource = record["result"].get("resource", {})
            if arm == "global_cem":
                calls = list(panel.prior.old.previous.fixed.read_calls(
                    panel.OUT / f"plan_{root:02d}_{arm}_calls"))
                prediction_calls = len(calls)
                full_samples = sum(len(actions) for actions, _, _ in calls
                                   if actions.shape[1] == 15)
                values = {"prediction_tokens": record["result"]["prediction_tokens"],
                          "prediction_calls": prediction_calls,
                          "full_samples": full_samples, "svd_calls": 0,
                          "svd_seconds": 0.0, "wall_seconds": record["elapsed_seconds"],
                          "token_cap": panel.CAP}
            else:
                required = ("prediction_tokens", "prediction_calls", "full_samples",
                            "svd_calls", "svd_seconds", "wall_seconds", "token_cap")
                if any(key not in resource for key in required):
                    raise ValueError("incomplete compiler resource ledger")
                values = resource
            if values["prediction_tokens"] != record["tokens"] or values["token_cap"] != panel.CAP:
                raise ValueError("resource/token binding mismatch")
            rows.append({
                "root": root, "arm": arm, "tokens": record["tokens"],
                "prediction_calls": values["prediction_calls"],
                "full_samples": values["full_samples"],
                "svd_calls": values["svd_calls"],
                "svd_seconds": values["svd_seconds"],
                "planner_wall_seconds": values["wall_seconds"],
                "journal_wall_seconds": record["elapsed_seconds"],
            })
    summary = {}
    for arm in panel.ARMS:
        group = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "roots": len(group),
            "tokens": sum(row["tokens"] for row in group),
            "prediction_calls": sum(row["prediction_calls"] for row in group),
            "full_samples": sum(row["full_samples"] for row in group),
            "svd_calls": sum(row["svd_calls"] for row in group),
            "svd_seconds": sum(row["svd_seconds"] for row in group),
            "planner_wall_seconds_sum": sum(row["planner_wall_seconds"] for row in group),
            "planner_wall_seconds_mean": statistics.mean(row["planner_wall_seconds"] for row in group),
            "journal_wall_seconds_sum": sum(row["journal_wall_seconds"] for row in group),
        }
    result = {
        "status": "passed",
        "source_audit": str(panel.OUT / "audit.json"),
        "rows": rows,
        "summary": summary,
        "measured": ["prediction_tokens", "prediction_calls", "full_samples",
                     "svd_calls", "svd_seconds", "planner_wall_seconds",
                     "journal_wall_seconds"],
        "not_measured": ["GPU identity", "GPU utilization", "peak GPU memory",
                         "encoding time", "CPU preprocessing time", "synchronization time",
                         "model loading time", "peak host memory", "FLOPs"],
        "full_cost_certified": False,
        "scope": "same-environment read-only ledger; development only",
    }
    panel.prior.journal.write_json(OUTPUT, result)
    print(json.dumps({"status": result["status"], "summary": summary,
                      "full_cost_certified": False}, indent=2))


if __name__ == "__main__":
    main()
