"""Reconstruct cheap direction and all optimizer requests from sealed journals."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_cheap_direction_edit_dev_20261105_codex as run


def main():
    out = run.OUT
    audit = json.loads((out / "audit.json").read_text())
    if audit["status"] != "passed" or audit["summary_sha256"] != run.prior.digest(out / "summary.json"):
        raise ValueError("prediction audit mismatch")
    manifest = json.loads((out / "manifest.json").read_text())
    run.fixed.prior.prior.prior.seal.verify(manifest["dependencies"])
    rows = json.loads((out / "summary.json").read_text())["rows"]
    count = 0
    for row in rows:
        if not row["arm"].startswith("cheap"):
            continue
        root, repeat, arm = row["root"], row["repeat"], row["arm"]
        log = out / f"root{root:02d}_rep{repeat}_{arm}_calls"
        source = json.loads((run.fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        parent = run.fixed.starts_from_result(source["result"])[0][0]
        counter = [0]

        def predict(actions):
            stem = f"{counter[0]:06d}"
            with np.load(log / f"{stem}_actions.npz", allow_pickle=False) as f:
                np.testing.assert_array_equal(actions, f["actions"])
            with np.load(log / f"{stem}_result.npz", allow_pickle=False) as f:
                result = f["features"].copy(), f["scores"].copy()
            counter[0] += 1
            return result

        result = run.cheap_arm(predict, parent, manifest["tasks"][root]["goal"],
                               row["seed"], arm.endswith("_total"))
        if counter[0] != row["calls"] or result != row["result"]:
            raise ValueError(f"replay differs {root} {repeat} {arm}")
        count += 1
    if count != 64:
        raise ValueError("missing cheap runs")
    output = {"status": "passed", "rows": count, "prediction_audit_sha256": run.prior.digest(out / "audit.json"),
              "auditor_sha256": run.prior.digest(Path(__file__)),
              "scope": "same direction and optimizer implementations; request reconstruction, not independent algorithm"}
    with (out / "direction_replay_audit.json").open("x") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
