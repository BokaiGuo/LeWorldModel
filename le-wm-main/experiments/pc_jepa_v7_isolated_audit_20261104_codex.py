"""Audit the new isolated run against the old exposed-input engineering run."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/pc_jepa_pusht_v7_planning_dev_20261104_codex.json"
BASE = ROOT / "results/pc_jepa_v7_isolated_input_dev_20261104_codex"
NEW = BASE / "output/result.json"
AUDIT = BASE / "output/audit.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maxdiff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return float(np.max(np.abs(a - b))) if a.size else 0.


def main():
    old, new = json.loads(OLD.read_text()), json.loads(NEW.read_text())
    if len(new["forbidden_path_probes"]) != 5 or not all(
        x in ("FileNotFoundError", "PermissionError") for x in new["forbidden_path_probes"].values()
    ):
        raise ValueError("isolation probe failure")
    arms = {}
    for name in ("v7_response", "smooth_cem"):
        x, y = old["results"][name], new["results"][name]
        if len(x["evaluations"]) != len(y["evaluations"]):
            raise ValueError("evaluation count mismatch")
        differences = {field: max(maxdiff(i[field], j[field]) for i, j in
                                   zip(x["evaluations"], y["evaluations"]))
                       for field in ("actions", "positions", "scores")}
        arms[name] = {"old_score": x["decision"]["selected"]["terminal_score"],
                      "new_score": y["decision"]["selected"]["terminal_score"],
                      "old_meter": x["meter"], "new_meter": y["meter"],
                      "evaluation_batches": len(y["evaluations"]),
                      "max_absolute_differences": differences}
        if any(differences.values()) or arms[name]["old_score"] != arms[name]["new_score"]:
            raise ValueError(f"engineering regression mismatch: {name}")
    record = {"old_result_sha256": sha(OLD), "isolated_result_sha256": sha(NEW),
              "audit_script_sha256": sha(Path(__file__)),
              "chart_max_absolute_difference": maxdiff(old["chart_goal"], new["chart_goal"]),
              "forbidden_path_probes": new["forbidden_path_probes"], "arms": arms,
              "scope": "exposed-root model-only engineering equivalence; not physical or independent confirmation"}
    if record["chart_max_absolute_difference"] != 0:
        raise ValueError("chart mismatch")
    with AUDIT.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
