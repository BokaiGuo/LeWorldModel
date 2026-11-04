"""Recalculate established interventions against sealed v7 selected chains."""

import json
from pathlib import Path

import numpy as np

from experiments import pc_v5_e0_prospective_interventions_dev_20261104_codex as original
from experiments import pc_v5_e0_prospective_interventions_v2_dev_20261104_codex as original_v2
from experiments import pc_v7_e0_exposed_panel_dev_20261104_codex as plans
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = plans.source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V7_E0_EXPOSED_INTERVENTION_DEV_PROTOCOL_20261104_CODEX.md"
OUTPUT = ROOT / "results/pc_v7_e0_exposed_intervention_dev_20261104_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    v7 = json.loads(plans.OUT.read_text())
    audit = json.loads(plans.AUDIT.read_text())
    if audit["plans_sha256"] != plans.source.sha(plans.OUT):
        raise ValueError("v7 plan audit mismatch")
    old_result = original_v2.PREFIX.with_name(original_v2.PREFIX.name + "_result.json")
    old_audit_path = original_v2.PREFIX.with_name(original_v2.PREFIX.name + "_audit.json")
    old_audit = json.loads(old_audit_path.read_text())
    if old_audit["result_sha256"] != original.sha(old_result):
        raise ValueError("old intervention audit mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, plans.OUT, plans.AUDIT, old_result,
                         old_audit_path, Path(original.__file__), Path(original_v2.__file__), Path(original.local.__file__),
                         Path(original.core.__file__), Path(original.exact.__file__),
                         Path(original.exact.wall.__file__)])
    rows = []
    print("Recalculating same-chain interventions on the exposed v7 panel.", flush=True)
    for row in v7["rows"]:
        root = row["root"]
        prior_meta = original_v2.PREFIX.with_name(original_v2.PREFIX.name + f"_root{root:02d}.json")
        prior_arrays = original_v2.PREFIX.with_name(original_v2.PREFIX.name + f"_root{root:02d}.npz")
        saved = json.loads(prior_meta.read_text())
        if saved["arrays_sha256"] != original.sha(prior_arrays):
            raise ValueError("prior intervention array hash mismatch")
        old_local = original.local

        class LocalAdapter:
            @staticmethod
            def build(input_row):
                value = old_local.build(input_row)
                value["controls"]["selected"]["candidate_index"] = -1
                return value

        try:
            original.local = LocalAdapter
            rebuilt, arrays = original.calculate(row)
        finally:
            original.local = old_local
        rebuilt["arrays_sha256"] = saved["arrays_sha256"]
        if rebuilt != saved:
            raise ValueError(f"intervention result changed for root {root}")
        with np.load(prior_arrays, allow_pickle=False) as stored:
            for key, value in arrays.items():
                np.testing.assert_array_equal(stored[key], value)
        rows.append({"root": root, "selected_chain": rebuilt["selected_chain"],
                     "base_intersection": rebuilt["base_intersection"],
                     "all_intersection": rebuilt["all_intersection"],
                     "reference_metadata_sha256": original.sha(prior_meta),
                     "reference_arrays_sha256": original.sha(prior_arrays)})
        print(f"root={root} recomputed", flush=True)
    seal.verify(deps)
    plans.source.write(OUTPUT, {"dependencies": deps, "v7_plans_sha256": plans.source.sha(plans.OUT),
        "rows": rows, "scope": "same-code recalculation on exposed development roots, not independent replication"})


if __name__ == "__main__":
    main()
