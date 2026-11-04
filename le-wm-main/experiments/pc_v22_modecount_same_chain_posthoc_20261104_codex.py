"""Apply the existing mechanism intervention to all sealed two-mode chains."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_v22_modecount_e0_dev_20261104_codex as panel
from experiments import pc_v13_e0_prospective_mechanism_dev_20261104_codex as mechanism
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = panel.ROOT / "results/pc_v22_modecount_same_chain_posthoc_20261104_codex"
PROTOCOL = panel.ROOT.parent / "docs/PC_V22_MODECOUNT_SAME_CHAIN_POSTHOC_PROTOCOL_20261104_CODEX.md"


def source_rows():
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    plans = json.loads((panel.OUT / "plans.json").read_text())
    audit = json.loads((panel.OUT / "audit.json").read_text())
    seal.verify(manifest["dependencies"])
    if (audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(panel.OUT / "plans.json")
            or plans["manifest_sha256"] != seal.digest(panel.OUT / "manifest.json")):
        raise ValueError("unsealed source panel")
    rows = [row for row in plans["rows"] if row["arm"] == "v22_mode2"]
    if len(rows) != 8 or [row["root"] for row in rows] != list(range(8)):
        raise ValueError("missing or reordered panel roots")
    return rows


def adapt(row):
    return {"root": row["root"], "task": row["task"], "status": row["status"],
            "result": None if row["result"] is None else row["result"]["result"]}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    source = source_rows()
    dependencies = seal.capture([Path(__file__), PROTOCOL, Path(panel.__file__), Path(mechanism.__file__),
        Path(mechanism.intervention.__file__), Path(mechanism.intervention.local.__file__),
        Path(mechanism.intervention.core.__file__), Path(mechanism.exact.__file__),
        Path(mechanism.exact.wall.__file__), Path(mechanism.exact.wall.physics.__file__),
        panel.OUT / "manifest.json", panel.OUT / "plans.json", panel.OUT / "audit.json"])
    OUT.mkdir()
    mechanism.writer(OUT / "manifest.json", {"dependencies": dependencies,
        "scope": "post-hoc all-root development diagnostic"})
    rows = []
    for source_row in source:
        row = adapt(source_row)
        try:
            if row["status"] != "completed":
                raise ValueError("source planner failed")
            data, arrays = mechanism.calculate(row)
            path = OUT / f"intervention_root{row['root']:02d}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            rows.append({"root": row["root"], "status": "completed", "data": data,
                         "arrays": path.name, "arrays_sha256": seal.digest(path)})
        except Exception:
            rows.append({"root": row["root"], "status": "failed", "error": traceback.format_exc()})
        print(f"root={row['root']} status={rows[-1]['status']}", flush=True)
    seal.verify(dependencies)
    mechanism.writer(OUT / "interventions.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"),
                                                 "rows": rows})


def audit():
    output = OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    source = source_rows()
    manifest = json.loads((OUT / "manifest.json").read_text())
    recorded = json.loads((OUT / "interventions.json").read_text())
    seal.verify(manifest["dependencies"])
    if recorded["manifest_sha256"] != seal.digest(OUT / "manifest.json") or len(recorded["rows"]) != 8:
        raise ValueError("intervention binding mismatch")
    for source_row, row in zip(source, recorded["rows"], strict=True):
        if source_row["root"] != row["root"]:
            raise ValueError("root mismatch")
        if row["status"] != "completed":
            continue
        path = OUT / row["arrays"]
        if seal.digest(path) != row["arrays_sha256"]:
            raise ValueError("changed arrays")
        rebuilt, arrays = mechanism.calculate(adapt(source_row))
        if rebuilt != row["data"]:
            raise ValueError("intervention reconstruction mismatch")
        with np.load(path, allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
    seal.verify(manifest["dependencies"])
    mechanism.writer(output, {"status": "passed", "interventions_sha256": seal.digest(OUT / "interventions.json"),
                              "attempted": len(recorded["rows"]),
                              "completed": sum(row["status"] == "completed" for row in recorded["rows"]),
                              "scope": "same-code deterministic reconstruction, not independent replication"})
    print("All available interventions reconstructed.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
