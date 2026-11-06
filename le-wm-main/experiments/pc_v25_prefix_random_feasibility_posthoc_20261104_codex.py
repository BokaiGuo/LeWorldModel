"""Post-hoc geometric witnesses for the prefix-random development panel."""
import json
from pathlib import Path

from experiments import pc_v25_prefix_random_refinement_e0_dev_20261104_codex as panel
from experiments import pc_v25_scale_feasibility_posthoc_20261104_codex as geometry

base = panel.previous
OUT = base.ROOT / "results/pc_v25_prefix_random_feasibility_posthoc_20261104_codex.json"
PROTOCOL = base.ROOT.parent / "docs/PC_V25_PREFIX_RANDOM_FEASIBILITY_POSTHOC_PROTOCOL_20261104_CODEX.md"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    audit = json.loads((panel.OUT / "audit.json").read_text())
    base.seal.verify(manifest["dependencies"])
    if (audit["status"] != "passed" or len(audit["rows"]) != 16
            or any(row["status"] != "completed" for row in audit["rows"])):
        raise ValueError("panel not completely audited")
    deps = base.seal.capture([Path(__file__), PROTOCOL, Path(geometry.__file__),
                              panel.OUT / "manifest.json", panel.OUT / "audit.json"])
    print("Checking fixed privileged paths on sealed prefix-random tasks.", flush=True)
    rows = []
    for root, task in enumerate(panel.tasks()):
        witnesses = geometry.calculate(task)
        rows.append({"root": root, "task": task, "witnesses": witnesses,
                     "feasibility": "witnessed" if any(w.get("success", False) for w in witnesses) else "unknown"})
    base.seal.verify(manifest["dependencies"])
    base.seal.verify(deps)
    base.journal.write_json(OUT, {"dependencies": deps, "rows": rows,
        "scope": "post-hoc privileged witness; not method performance or PC0 budget certification"})
    print(f"witnessed={sum(row['feasibility'] == 'witnessed' for row in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
