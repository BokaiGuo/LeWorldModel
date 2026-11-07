"""Evaluate already-sealed RGB goal scores against prior physical replay."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments.pc_pusht_rgb_goal_rotation_score_20261107_codex import OUT
from experiments.pc_pusht_shared_pool_real_endpoint_20261107_codex import OUT as PHYSICAL, summarize

REPORT = OUT.parent / f"{OUT.name}_evaluation.json"


def main():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    sealed = json.loads(OUT.with_suffix(".json").read_text())
    if file_hash(OUT.with_suffix(".npz")) != sealed["arrays_sha256"]:
        raise ValueError("sealed arrays changed")
    prior = json.loads(Path(next(p for p in sealed["sources"] if p.endswith("pc_pusht_all_shared_prefix_rescore_20261107_codex.json"))).read_text())
    rows, physical_hashes = [], {}
    with np.load(OUT.with_suffix(".npz"), allow_pickle=False) as scores:
        for row in sealed["rows"]:
            task = row["task"]
            stem = PHYSICAL.parent / f"{PHYSICAL.name}_task{task}"
            meta = json.loads(stem.with_suffix(".json").read_text())
            if file_hash(stem.with_suffix(".npz")) != meta["arrays_sha256"]:
                raise ValueError("physical endpoint arrays changed")
            if meta["model_sha256"] != sealed["model_sha256"]:
                raise ValueError("physical endpoint model drift")
            if meta["hashes"] != [c["sha256"] for c in prior["plans"][task]["candidates"]]:
                raise ValueError("candidate identities differ")
            physical_hashes[str(stem.with_suffix(".json"))] = file_hash(stem.with_suffix(".json"))
            physical_hashes[str(stem.with_suffix(".npz"))] = file_hash(stem.with_suffix(".npz"))
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as data:
                ever, final = data["ever"].copy(), data["final"].copy()
                summary = {}
                for variant in ("original", "tolerant"):
                    values = scores[f"{variant}_{task}"]
                    if int(np.argmin(values)) != row[f"{variant}_winner"]:
                        raise ValueError("sealed winner mismatch")
                    summary[variant] = summarize(values, ever, final)
            rows.append({"task": task, "abstained": row["detection"]["abstained"], "summary": summary})
    lookup = {r["task"]: r for r in rows}
    exits = {label: bool(lookup[1]["summary"]["tolerant"][f"winner_{label}"]
                         and lookup[0]["summary"]["tolerant"][f"winner_{label}"])
             for label in ("ever", "final")}
    with REPORT.open("x") as f:
        json.dump({"scope": "exposed fixed-pool development, no gate claim",
                   "protocol_exit_by_success_definition": exits,
                   "exit_note": "protocol did not specify ever versus final; both are reported",
                   "sealed_json_sha256": file_hash(OUT.with_suffix(".json")),
                   "sealed_arrays_sha256": file_hash(OUT.with_suffix(".npz")),
                   "evaluator_sha256": file_hash(Path(__file__)),
                   "physical_sources": physical_hashes, "rows": rows}, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"exit_by_success_definition": exits, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
