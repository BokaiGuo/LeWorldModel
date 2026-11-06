"""Compare paired saved calls to locate and verify random-refinement divergence."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v25_random_refinement_e0_dev_20261104_codex as panel

OUT = panel.ROOT / "results/pc_v25_random_refinement_divergence_posthoc_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    audit = json.loads((panel.OUT / "audit.json").read_text())
    panel.seal.verify(manifest["dependencies"])
    if audit["status"] != "passed" or len(audit["rows"]) != 16:
        raise ValueError("panel audit missing")
    deps = panel.seal.capture([Path(__file__), panel.OUT / "manifest.json", panel.OUT / "audit.json"])
    rows = []
    for root, task in enumerate(panel.tasks()):
        records = {}
        for arm in panel.ARMS:
            row = json.loads((panel.OUT / f"root{root:02d}_{arm}.json").read_text())
            if row["status"] != "completed" or row["task"] != task:
                raise ValueError("incomplete paired result")
            calls = list(panel.prior.prior.read_calls(panel.OUT / f"root{root:02d}_{arm}_calls"))
            records[arm] = (row, calls)
        base, base_calls = records["v25"]
        random, random_calls = records["random_refinement"]
        if len(base_calls) != len(random_calls):
            raise ValueError("paired call count differs")
        divergence = None
        for index, (left, right) in enumerate(zip(base_calls, random_calls, strict=True)):
            if not np.array_equal(left[0], right[0]):
                divergence = index
                break
            np.testing.assert_array_equal(left[1], right[1])
            np.testing.assert_array_equal(left[2], right[2])
        if divergence is None:
            raise ValueError("seed replacement did not change any prediction")
        original, replaced = base_calls[divergence][0], random_calls[divergence][0]
        if original.shape != replaced.shape or original.shape != (4, 15, 2):
            raise ValueError("unexpected first divergent batch")
        np.testing.assert_array_equal(original[0], replaced[0])
        seed = random["result"]["config"]["seed"]
        rng = np.random.default_rng(np.random.SeedSequence([20310000, seed]))
        expected = rng.uniform(-1, 1, replaced[1:].shape).astype(np.float32)
        np.testing.assert_array_equal(replaced[1:], expected)
        rows.append({"root": root, "first_divergent_call": divergence,
                     "identical_preceding_calls": divergence, "seed_batch_size": len(replaced),
                     "incumbent_preserved": True, "random_slots_verified": 3,
                     "v25_score": base["result"]["selected_score"],
                     "random_score": random["result"]["selected_score"]})
    panel.seal.verify(manifest["dependencies"])
    panel.seal.verify(deps)
    panel.journal.write_json(OUT, {"dependencies": deps, "rows": rows,
        "scope": "post-hoc paired journal consistency, not causal or independent replication evidence"})
    print(f"Verified first divergence and incumbent on {len(rows)} paired roots.")


if __name__ == "__main__":
    main()
