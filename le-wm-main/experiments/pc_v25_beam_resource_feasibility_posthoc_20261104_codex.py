"""Post-hoc privileged feasibility witnesses for the frozen beam panel."""
import json
import traceback
from pathlib import Path

from experiments import pc_v25_beam_resource_curve_dev_20261104_codex as panel
from experiments import pc_v25_scale_feasibility_posthoc_20261104_codex as feasibility

OUT = panel.OUT / "feasibility_posthoc.json"
PROTOCOL = panel.ROOT.parent / "docs/PC_V25_BEAM_RESOURCE_FEASIBILITY_POSTHOC_PROTOCOL_20261104_CODEX.md"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest_path = panel.OUT / "manifest.json"
    audit_path = panel.OUT / "audit.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    panel.seal.verify(manifest["dependencies"])
    if (manifest["tasks"] != panel.tasks() or audit["status"] != "passed"
            or len(audit["rows"]) != len(panel.tasks()) * len(panel.ARMS) * len(panel.CAPS)):
        raise ValueError("source panel is not fully audited")
    deps = panel.seal.capture([Path(__file__), PROTOCOL, Path(feasibility.__file__),
                               manifest_path, audit_path])
    rows = []
    print("Checking fixed privileged geometric witnesses; no planner reruns.", flush=True)
    for root, task in enumerate(panel.tasks()):
        try:
            records = feasibility.calculate(task)
            row = {"root": root, "task": task, "witnesses": records,
                   "feasibility": "witnessed" if any(x.get("success") for x in records) else "unknown"}
        except Exception:
            row = {"root": root, "task": task, "feasibility": "unknown", "error": traceback.format_exc()}
        rows.append(row)
    panel.seal.verify(manifest["dependencies"])
    panel.seal.verify(deps)
    panel.journal.write_json(OUT, {"dependencies": deps, "rows": rows,
        "scope": "privileged post-hoc witness; not a budget-certified oracle or method input"})
    print(f"witnessed={sum(x['feasibility'] == 'witnessed' for x in rows)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
