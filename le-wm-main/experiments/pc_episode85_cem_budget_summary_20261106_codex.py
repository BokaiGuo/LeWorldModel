"""Audit paired development CEM budget runs and produce a non-confirmatory summary."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results"
STEMS = {
    "105x6": "pc_episode_expansion_cem_task85_solver42_dev1106",
    "300x10": "pc_episode_expansion_cem300x10_task85_solver42_dev1106_gpu1",
    "300x30": "pc_episode_expansion_cem300x30_task85_solver42_dev1106_gpu1",
}
OUT = ROOT / "pc_episode85_cem_budget_summary_20261106_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    output, successes, snapshots = {}, {}, {}
    for label, stem in STEMS.items():
        metrics = json.loads((ROOT / stem / "eval_metrics.json").read_text())
        flags = metrics["episode_successes"]
        if len(flags) != 50 or any(not isinstance(x, bool) for x in flags):
            raise ValueError("unexpected episode outcomes")
        successes[label] = flags
        trace = [json.loads(line) for line in (ROOT / f"{stem}.trace.jsonl").open()]
        first = {}
        max_abs, count = 0., 0
        for row in trace:
            if row["step"] == 1:
                if row["reset_seed"] in first:
                    raise ValueError("duplicate reset")
                first[row["reset_seed"]] = row["before"]
            max_abs = max(max_abs, max(abs(float(a)) for a in row["action"]))
            count += 1
        if set(first) != set(range(871001, 871051)) or max_abs > 1.000001:
            raise ValueError("missing task initialization or illegal action")
        snapshots[label] = first
        audits = [json.loads(line) for line in (ROOT / f"{stem}.audit.jsonl").open()]
        environments = [env for record in audits for env in record["environments"]]
        output[label] = {"success": sum(flags), "task_successes": flags,
                         "planning_calls": len(environments),
                         "full_chain_scores": sum(env["scored_model_candidates"] for env in environments),
                         "solver_seconds": sum(record["elapsed_seconds"] for record in audits),
                         "subprocess_seconds": float((ROOT / f"{stem}.wall.txt").read_text()),
                         "physical_steps": count, "max_abs_physical_action": max_abs}
    for label in ("300x10", "300x30"):
        for seed in snapshots["105x6"]:
            if not np.allclose(snapshots["105x6"][seed], snapshots[label][seed], atol=1e-4, rtol=0):
                raise ValueError(f"initial state mismatch for {label} task {seed}")
        low, high = successes["105x6"], successes[label]
        output[label]["paired_vs_105x6"] = {
            "both_success": sum(a and b for a, b in zip(low, high, strict=True)),
            "105x6_only": sum(a and not b for a, b in zip(low, high, strict=True)),
            "higher_only": sum(not a and b for a, b in zip(low, high, strict=True)),
            "both_fail": sum(not a and not b for a, b in zip(low, high, strict=True))}
    with OUT.open("x") as stream:
        json.dump({"scope": "same exposed 50-task development panel; no noninferiority claim",
                   "initial_snapshots_match": True, "results": output}, stream, indent=2)
        stream.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
