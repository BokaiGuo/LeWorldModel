"""Post-hoc privileged feasibility and score ranking, not planner injection."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v5_e0_prospective_mechanism_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PROTOCOL = source.ROOT.parent / "docs/PC_V5_E0_PRIVILEGED_FEASIBILITY_DIAG_PROTOCOL_20261104_CODEX.md"
OUTPUT = source.PREFIX.with_name(source.PREFIX.name + "_privileged_feasibility_diag.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    deps = seal.capture([Path(__file__), PROTOCOL, source.PLANS, source.AUDIT,
                         Path(source.__file__), Path(source.exact.__file__),
                         Path(source.exact.wall.__file__), Path(source.exact.wall.physics.__file__)])
    plans = json.loads(source.PLANS.read_text())
    audit = json.loads(source.AUDIT.read_text())
    if audit["plans_sha256"] != source.sha(source.PLANS):
        raise ValueError("source audit mismatch")
    rows = []
    for row in plans["rows"]:
        task = row["task"]
        goal = np.asarray(task["goal"])
        chain = np.asarray(row["result"]["selected"]["chain"], dtype=np.float32)
        selected = float(np.linalg.norm(source.exact.endpoints(task, chain[None])[0] - goal))
        np.testing.assert_allclose(selected, row["result"]["selected"]["terminal_score"], rtol=0, atol=1e-12)
        witnesses = []
        for witness in source.exact.wall.feasibility(task):
            entry = {"height": witness["height"], "steps_needed": witness["steps_needed"],
                     "eligible": bool(witness["within_horizon"] and witness["steps_needed"] <= 15)}
            if entry["eligible"]:
                action = np.asarray(witness["actions"][:15], dtype=float) / source.exact.wall.HIGH
                if not np.isfinite(action).all() or np.any(np.abs(action) > 1 + 1e-12):
                    raise ValueError("illegal witness")
                raw = action.astype(np.float32)
                distance = float(np.linalg.norm(source.exact.endpoints(task, raw[None])[0] - goal))
                entry.update(actions=action.tolist(), float32_actions=raw.tolist(),
                             terminal_distance=distance, successful=distance < .05,
                             beats_selected=distance < selected)
            witnesses.append(entry)
        rows.append({"root": row["root"], "task": task, "selected_distance": selected,
                     "witnesses": witnesses})
    seal.verify(deps)
    source.write(OUTPUT, {"dependencies": deps, "scope": "privileged post-hoc scoring; no planner injection",
                          "rows": rows})
    print(f"Saved privileged feasibility diagnostics for {len(rows)} roots.", flush=True)


if __name__ == "__main__":
    main()
