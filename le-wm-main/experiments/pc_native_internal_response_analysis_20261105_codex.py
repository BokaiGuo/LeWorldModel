"""Summarize fixed-budget development runs and validate audit counters."""
import json
import re
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    paths = {
        "native": root / "pc_native_internal_control_seed42_20261105_codex",
        "response": root / "pc_native_internal_response_seed42_20261105_codex",
        "random": root / "pc_native_internal_random_seed42_20261105_codex",
    }
    report = {"scope": "seed 42 development, not full cost or confirmation", "arms": {}}
    outcomes = {}
    for name, directory in paths.items():
        metrics = json.loads((directory / "eval_metrics.json").read_text())
        outcomes[name] = np.array(metrics["episode_successes"], dtype=bool)
        if len(outcomes[name]) != 50:
            raise ValueError(f"incomplete metrics: {name}")
        log = (directory / "pusht_results.txt").read_text()
        match = re.findall(r"evaluation_time:\s*([0-9.]+)", log)
        if len(match) != 1:
            raise ValueError(f"ambiguous elapsed time: {name}")
        result = {"successes": int(outcomes[name].sum()), "evaluation_seconds": float(match[0])}
        if name != "native":
            records = [json.loads(line) for line in directory.with_suffix(".jsonl").read_text().splitlines()]
            iterations = [i for record in records for i in record["iterations"]]
            if any(i["scored_per_environment"] != 300 for i in iterations):
                raise ValueError(f"budget mismatch: {name}")
            result.update(iterations=len(iterations),
                          probe_elites=sum(sum(i["probe_elites"]) for i in iterations),
                          proposal_elites=sum(sum(i["proposal_elites"]) for i in iterations))
        report["arms"][name] = result
    report["paired"] = {}
    for left, right in (("native", "response"), ("native", "random"), ("response", "random")):
        a, b = outcomes[left], outcomes[right]
        report["paired"][f"{left}_vs_{right}"] = {
            "left_only": int((a & ~b).sum()), "right_only": int((~a & b).sum()),
            "both_success": int((a & b).sum()), "both_fail": int((~a & ~b).sum()),
        }
    output = root / "pc_native_internal_response_analysis_seed42_20261105_codex.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
