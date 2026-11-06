"""Compute protocol-required predicted/encoded pair agreement from saved latents."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_pusht_scored_success_failure_visual_audit_dev_20261105_codex as run


def main():
    run.verify()
    audit = json.loads((run.OUT / "audit.json").read_text())
    if audit["status"] != "passed" or audit["report_sha256"] != run.sha(run.OUT / "report.json"):
        raise ValueError("saved scoring audit mismatch")
    result = []
    for arm in ("response", "cem"):
        path = run.OUT / f"root_11_{arm}/output/latents.npz"
        with np.load(path, allow_pickle=False) as f:
            pred, actual, goal = (f[k].astype(np.float64) for k in ("predicted", "actual", "goal"))
        p = ((pred-goal)**2).mean(axis=(1, 2))
        a = ((actual-goal)**2).mean(axis=(1, 2))
        i, j = np.triu_indices(len(p), 1)
        product = (p[i]-p[j]) * (a[i]-a[j])
        result.append({"arm": arm, "agree": int(np.sum(product > 0)), "disagree": int(np.sum(product < 0)),
                       "tied": int(np.sum(product == 0)), "latents_sha256": run.sha(path)})
    run.save(run.OUT / "pair_order.json", {"status": "completed", "rows": result,
                                           "script_sha256": run.sha(Path(__file__)),
                                           "audit_sha256": run.sha(run.OUT / "audit.json")})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
