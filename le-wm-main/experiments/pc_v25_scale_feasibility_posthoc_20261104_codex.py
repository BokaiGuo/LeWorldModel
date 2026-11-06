"""Post-hoc privileged witness of already-sealed candidate-scale tasks."""
import json
from pathlib import Path
import sys

import numpy as np

from experiments import pc_v25_candidate_scale_e0_dev_20261104_codex as source

OUT = source.ROOT / "results/pc_v25_candidate_scale_feasibility_posthoc_20261104_codex.json"
PROTOCOL = source.ROOT.parent / "docs/PC_V25_SCALE_TASK_FEASIBILITY_POSTHOC_PROTOCOL_20261104_CODEX.md"


def calculate(task):
    records = []
    for item in source.OLD.exact.wall.feasibility(task):
        eligible = bool(item["within_horizon"] and item["steps_needed"] <= 15)
        row = {"height": item["height"], "steps_needed": item["steps_needed"], "eligible": eligible}
        if eligible:
            actions = (np.asarray(item["actions"][:15], np.float64)
                       / source.OLD.exact.wall.HIGH).astype(np.float32)
            if actions.shape != (15, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
                raise ValueError("invalid witness action")
            positions = source.OLD.exact.endpoints(task, actions[None])
            path, blocked = source.OLD.previous.independent(task["start"], actions[None])
            np.testing.assert_allclose(path[:, -1, :2], positions, rtol=0, atol=1e-12)
            distance = float(np.linalg.norm(positions[0] - task["goal"]))
            row.update(actions=actions.tolist(), path=path[0].tolist(), blocked=blocked[0].tolist(),
                       distance=distance, success=distance < .05)
        records.append(row)
    return records


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest_path = source.OUT / "manifest.json"
    audit_path = source.OUT / "audit.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    source.seal.verify(manifest["dependencies"])
    if (manifest["tasks"] != source.tasks() or audit["status"] != "passed"
            or len(audit["rows"]) != 2 * len(source.tasks())
            or any(row["status"] != "completed" for row in audit["rows"])):
        raise ValueError("source panel is not completed and audited")
    dependencies = source.seal.capture([Path(__file__), PROTOCOL, manifest_path, audit_path])
    rows = []
    print("Checking two fixed privileged geometric witnesses on each sealed development task.", flush=True)
    for root, task in enumerate(source.tasks()):
        witnesses = calculate(task)
        rows.append({"root": root, "task": task, "witnesses": witnesses,
                     "feasibility": "witnessed" if any(w.get("success", False) for w in witnesses)
                     else "unknown"})
    source.seal.verify(manifest["dependencies"])
    source.seal.verify(dependencies)
    source.journal.write_json(OUT, {"dependencies": dependencies, "rows": rows,
        "scope": "post-hoc privileged diagnostic, not method performance or PC0 budget certification"})
    print(f"witnessed={sum(row['feasibility'] == 'witnessed' for row in rows)}/{len(rows)}")


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("no arguments")
    main()
