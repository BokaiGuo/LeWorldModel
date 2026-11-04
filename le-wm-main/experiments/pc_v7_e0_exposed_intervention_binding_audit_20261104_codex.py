"""Audit v7 intervention rows against sealed v5 per-root artifact bindings."""

import json
from pathlib import Path

import numpy as np

from experiments import pc_v7_e0_exposed_intervention_dev_20261104_codex as run

OUTPUT = run.OUTPUT.with_name(run.OUTPUT.stem + "_binding_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(run.OUTPUT.read_text())
    plans = json.loads(run.plans.OUT.read_text())
    old_result_path = run.original_v2.PREFIX.with_name(run.original_v2.PREFIX.name + "_result.json")
    old_audit_path = run.original_v2.PREFIX.with_name(run.original_v2.PREFIX.name + "_audit.json")
    old_result, old_audit = (json.loads(path.read_text()) for path in (old_result_path, old_audit_path))
    if (run.plans.source.sha(run.plans.OUT) != data["v7_plans_sha256"]
            or old_audit["result_sha256"] != run.original.sha(old_result_path)):
        raise ValueError("plan or old index binding mismatch")
    if not (len(data["rows"]) == len(plans["rows"]) == len(old_result["rows"]) == 18):
        raise ValueError("incorrect panel length")
    for root, (row, plan, binding) in enumerate(zip(data["rows"], plans["rows"], old_result["rows"], strict=True)):
        meta = run.original_v2.PREFIX.with_name(run.original_v2.PREFIX.name + f"_root{root:02d}.json")
        arrays = run.original_v2.PREFIX.with_name(run.original_v2.PREFIX.name + f"_root{root:02d}.npz")
        if not (row["root"] == plan["root"] == binding["root"] == root
                and row["reference_metadata_sha256"] == binding["metadata_sha256"] == run.original.sha(meta)
                and row["reference_arrays_sha256"] == binding["arrays_sha256"] == run.original.sha(arrays)):
            raise ValueError(f"root {root} artifact binding mismatch")
        np.testing.assert_array_equal(np.asarray(row["selected_chain"], dtype=np.float32),
                                      np.asarray(plan["result"]["selected"]["chain"], dtype=np.float32))
    run.plans.source.write(OUTPUT, {"intervention_sha256": run.plans.source.sha(run.OUTPUT),
        "old_result_sha256": run.original.sha(old_result_path), "verified_roots": 18,
        "scope": "hash and v7 selected-chain bindings; same-backend intervention recomputation"})
    print("Verified 18 v7 selected-chain and historical artifact bindings.", flush=True)


if __name__ == "__main__":
    main()
