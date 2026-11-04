"""Bind wall-branch instability to the actual retained successor indices."""
import json
from pathlib import Path

from experiments import pc_v8_v7_e0_fd_branch_stability_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = source.OUT.with_name(source.OUT.stem + "_retained_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    data = json.loads(source.OUT.read_text())
    seal.verify(data["dependencies"])
    plans = {"v8": json.loads(source.geometry.v8.RESULT.read_text())["rows"],
             "v7": json.loads(source.geometry.panel.OUT.read_text())["rows"]}
    detail = {(r["arm"], r["root"], r["index"]): r for r in data["rows"] if r["layer"] == "successor"}
    if len(detail) != sum(len(r["result"]["v7"]["successor_responses"]) for r in plans["v8"]) + \
            sum(len(r["result"]["successor_responses"]) for r in plans["v7"]):
        raise ValueError("duplicate or missing successor")
    rows = []
    for arm, panel in plans.items():
        for row in panel:
            compiled = row["result"]["v7"] if arm == "v8" else row["result"]
            retained = compiled["retained"]
            if len(retained) != len(set(retained)):
                raise ValueError("duplicate retained index")
            unstable = [i for i in retained if detail[(arm, row["root"], i)]["unstable_coordinates"]]
            rows.append({"arm": arm, "root": row["root"], "retained": retained, "unstable_retained": unstable})
    summary = {arm: {"retained": sum(len(r["retained"]) for r in rows if r["arm"] == arm),
                     "unstable_retained": sum(len(r["unstable_retained"]) for r in rows if r["arm"] == arm),
                     "roots_with_unstable_retained": sum(bool(r["unstable_retained"])
                                                          for r in rows if r["arm"] == arm)}
               for arm in plans}
    source.geometry.panel.source.write(OUT, {"source_sha256": seal.digest(source.OUT),
        "script_sha256": seal.digest(Path(__file__)), "rows": rows, "summary": summary,
        "scope": "saved retained indices joined to actual .05 branch replay"})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
