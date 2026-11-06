"""Audit every persisted v25 mechanism endpoint call against alternate geometry."""
import json

import numpy as np

from experiments import pc_v25_e0_prospective_same_chain_dev_20261104_codex as panel


def main():
    out = panel.OUT / "interventions_geometry_audit.json"
    if out.exists():
        raise FileExistsError(out)
    manifest, plans, data = (json.loads((panel.OUT / name).read_text())
                             for name in ("manifest.json", "plans.json", "interventions.json"))
    panel.seal.verify(manifest["dependencies"])
    if data["plans_sha256"] != panel.seal.digest(panel.OUT / "plans.json"):
        raise ValueError("plan hash mismatch")
    rows = []
    for plan, entry in zip(plans["rows"], data["rows"], strict=True):
        root = plan["root"]
        if entry["root"] != root:
            raise ValueError("root mismatch")
        if entry["status"] == "plan_failed":
            rows.append({"root": root, "status": "plan_failed"})
            continue
        directory = panel.OUT / f"intervention_root{root:02d}_calls"
        info = panel.journal_audit.audit_journal(directory)
        if info["tokens"] != entry["journal_tokens"] or len(info["calls"]) != entry["journal_calls"]:
            raise ValueError("journal meter mismatch")
        if entry["status"] != "completed":
            rows.append({"root": root, "status": "failed", "tokens": info["tokens"]})
            continue
        if entry["data"]["selected_chain"] != plan["result"]["selected"]:
            raise ValueError("intervention chain differs from final selected chain")
        calls = list(panel.fixed.read_calls(directory))
        samples = 0
        for actions, positions, scores in calls:
            trajectory, _ = panel.OLD.previous.independent(plan["task"]["start"], actions)
            np.testing.assert_allclose(trajectory[:, -1, :2], positions, atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, np.linalg.norm(positions-plan["task"]["goal"], axis=1),
                                       atol=1e-12, rtol=0)
            samples += len(actions)
        rows.append({"root": root, "status": "passed", "calls": len(calls),
                     "samples": samples, "tokens": info["tokens"]})
    panel.seal.verify(manifest["dependencies"])
    panel.journal.write_json(out, {"status": "passed", "rows": rows,
        "interventions_sha256": panel.seal.digest(panel.OUT / "interventions.json"),
        "scope": "alternate implementation of same physical geometry; not independent replication"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
