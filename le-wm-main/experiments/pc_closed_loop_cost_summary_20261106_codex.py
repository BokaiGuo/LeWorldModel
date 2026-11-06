"""Summarize development timing runs, their task pairing and trace parity."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results"
RUNS = {
    "cem_105x6": ("cem_r2", "cem_run_r2", "cem_trace_r2", "cem_audit_r2"),
    "prefix": ("prefix", "prefix_run", "prefix_trace", "prefix_audit"),
    "cem_300x10": ("cem300x10", "cem300x10_run", "cem300x10_trace", "cem300x10_audit"),
    "cem_300x30": ("cem300x30", "cem300x30_run", "cem300x30_trace", "cem300x30_audit"),
}


def path(stem, extension):
    return ROOT / f"pc_closed_cost_{stem}_20261106_codex{extension}"


def lines(file):
    return [json.loads(row) for row in file.read_text().splitlines()]


def main():
    out = ROOT / "pc_closed_cost_summary_20261106_codex.json"
    if out.exists():
        raise FileExistsError(out)
    output, starts = {}, {}
    for name, (profile, run, trace, audit) in RUNS.items():
        timing = json.loads(path(profile, ".json").read_text())["records"]
        metric = json.loads((path(run, "") / "eval_metrics.json").read_text())
        physical = lines(path(trace, ".jsonl"))
        audits = lines(path(audit, ".jsonl"))
        starts[name] = {str(row["reset_seed"]): row["before"] for row in physical
                        if row["step"] == 1}
        if len(starts[name]) != 10:
            raise ValueError(f"expected 10 initial states for {name}")
        if any(abs(float(x)) > 1.000001 for row in physical for x in row["action"]):
            raise ValueError(f"illegal executed action: {name}")
        environments = [env for record in audits for env in record["environments"]]
        scores = sum(env["scored_model_candidates"] for env in environments)
        output[name] = {
            "successes": sum(metric["episode_successes"]),
            "episodes": len(metric["episode_successes"]),
            "scored_candidates_total": scores,
            "planning_contexts": len(environments),
            "executed_physical_steps": len(physical),
            "timing": timing,
        }
    pair = all(value == starts["cem_105x6"] for value in starts.values())
    key = lambda row: (row["reset_seed"], row["step"])
    instrumented = sorted(lines(path("cem_trace_parity", ".jsonl")), key=key)
    original = sorted(lines(ROOT / "pc_strong_cem_105x6_solver42_dev1106.trace.jsonl"), key=key)
    with out.open("x", encoding="utf-8") as stream:
        json.dump({"runs": output, "initial_physical_snapshots_equal": pair,
                   "cem_instrumented_vs_prior_trace_exact": instrumented == original,
                   "parity_trace_rows": len(instrumented),
                   "parity_scope": "CEM 105x6, task seed 42, solver seed 42, reset seed 771001 only",
                   "cost_panel_reset_seed": 871001}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
