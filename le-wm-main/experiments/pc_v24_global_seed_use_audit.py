"""Check v24's declared extra bank proposal reaches actual CEM seed batch."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v24_global_seed_e0_dev_20261104_codex as panel
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = panel.ROOT / "results/pc_v24_global_seed_use_audit_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest_path, plans_path, audit_path = (panel.OUT / x for x in
                                              ("manifest.json", "plans.json", "audit.json"))
    manifest, plans, audit = (json.loads(p.read_text()) for p in
                              (manifest_path, plans_path, audit_path))
    seal.verify(manifest["dependencies"])
    if (audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(plans_path)
            or plans["manifest_sha256"] != seal.digest(manifest_path)):
        raise ValueError("source binding mismatch")
    rows = []
    selected_rows = [r for r in plans["rows"] if r["arm"] == "v24"]
    if len(selected_rows) != 8:
        raise ValueError("unexpected roots")
    for row in selected_rows:
        if row["status"] != "completed" or seal.digest(panel.OUT / row["trace"]) != row["trace_sha256"]:
            raise ValueError("source failure or trace changed")
        result, calls = row["result"]["result"], row["result"]["calls"]
        proposal, child = result["global_proposals"], result["child"]
        starts = child["refinement_starts"]
        indices = proposal["warm_indices"]
        if len(indices) != 1 or len(starts) != 4:
            raise ValueError("unexpected global seed count")
        positions = [i for i, start in enumerate(starts) if start["source"] == "global_warm_start"]
        if len(positions) != 1:
            raise ValueError("warm start was not used exactly once")
        generation_count = sum(len(start["generations"]) for start in starts)
        seed_batch = len(calls) - generation_count - 2
        with np.load(panel.OUT / row["trace"], allow_pickle=False) as data:
            seeds = data[f"actions_{seed_batch}"]
            scores = data[f"scores_{seed_batch}"]
            np.testing.assert_array_equal(seeds[positions[0]],
                                          np.asarray(proposal["actions"][indices[0]], np.float32))
            np.testing.assert_allclose(scores[positions[0]], proposal["scores"][indices[0]],
                                       atol=1e-12, rtol=0)
        rows.append({"root": row["root"], "bank_index": indices[0], "seed_batch": seed_batch,
                     "warm_samples": starts[positions[0]]["samples"],
                     "warm_best_score": starts[positions[0]]["best_score"]})
    deps = seal.capture([Path(__file__), panel.PROTOCOL, manifest_path, plans_path, audit_path,
                         *[panel.OUT / r["trace"] for r in selected_rows]])
    seal.verify(manifest["dependencies"])
    with OUT.open("x") as handle:
        json.dump({"status": "passed", "scope": "same prediction trace and deterministic environment",
                   "dependencies": deps, "rows": rows}, handle, allow_nan=False)
        handle.write("\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
