"""Tie-aware addendum; preserve the original index-based audit."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_geometric_contact_equivalence_dev_20261106_codex import PREFIX, sha


def main():
    source = PREFIX.with_name(PREFIX.name + "_audit.json")
    output = PREFIX.with_name(PREFIX.name + "_tie_audit.json")
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads(source.read_text())
    rows = []
    for root in audit["rows"]:
        white = [c for c in root["conditions"] if c["appearance"] == "white"]
        rows.append({"root": root["root"],
                     "model_minus_zero_mean_regret": float(np.mean([
                         c["model_physical_regret"] - c["zero_physical_regret"] for c in white])),
                     "tie_aware_best_counts": {
                         method: sum(c[f"{method}_physical_regret"] <= 1e-9 for c in white)
                         for method in ("model", "encoded", "zero")}})
    result = {"scope": "posthoc correction for index-based tie misclassification, not new selection rule",
              "tolerance": 1e-9, "source_sha256": sha(source), "code_sha256": sha(__file__),
              "roots": len(rows), "conditions": 2 * len(rows),
              "tie_aware_best_counts": {
                  method: sum(r["tie_aware_best_counts"][method] for r in rows)
                  for method in ("model", "encoded", "zero")},
              "roots_model_improves_zero": sum(r["model_minus_zero_mean_regret"] < -1e-9 for r in rows),
              "roots_model_worsens_zero": sum(r["model_minus_zero_mean_regret"] > 1e-9 for r in rows),
              "rows": rows}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
