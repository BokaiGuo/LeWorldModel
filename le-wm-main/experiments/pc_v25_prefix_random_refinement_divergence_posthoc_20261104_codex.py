"""Verify paired prefix-preserving random refinement directly from journals."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v25_prefix_random_refinement_e0_dev_20261104_codex as panel

base = panel.previous
OUT = base.ROOT / "results/pc_v25_prefix_random_refinement_divergence_posthoc_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    audit = json.loads((panel.OUT / "audit.json").read_text())
    base.seal.verify(manifest["dependencies"])
    if audit["status"] != "passed" or len(audit["rows"]) != 16:
        raise ValueError("panel audit missing")
    deps = base.seal.capture([Path(__file__), panel.OUT / "manifest.json", panel.OUT / "audit.json"])
    rows = []
    for root, task in enumerate(panel.tasks()):
        results, calls = {}, {}
        for arm in base.ARMS:
            row = json.loads((panel.OUT / f"root{root:02d}_{arm}.json").read_text())
            if row["status"] != "completed" or row["task"] != task:
                raise ValueError("incomplete paired result")
            results[arm] = row
            calls[arm] = list(base.prior.prior.read_calls(panel.OUT / f"root{root:02d}_{arm}_calls"))
        if len(calls["v25"]) != len(calls["random_refinement"]):
            raise ValueError("paired call count differs")
        divergence = None
        for index, (left, right) in enumerate(zip(calls["v25"], calls["random_refinement"], strict=True)):
            if not np.array_equal(left[0], right[0]):
                divergence = index
                break
            np.testing.assert_array_equal(left[1], right[1])
            np.testing.assert_array_equal(left[2], right[2])
        if divergence is None:
            raise ValueError("no changed seed call")
        original, changed = calls["v25"][divergence][0], calls["random_refinement"][divergence][0]
        if original.shape != changed.shape or original.shape != (4, 15, 2):
            raise ValueError("unexpected first divergent batch")
        np.testing.assert_array_equal(original[0], changed[0])
        np.testing.assert_array_equal(original[:, :5], changed[:, :5])
        seed = results["random_refinement"]["result"]["config"]["seed"]
        rng = np.random.default_rng(np.random.SeedSequence([20312000, seed]))
        expected = rng.uniform(-1, 1, changed[1:, 5:, :].shape).astype(np.float32)
        np.testing.assert_array_equal(changed[1:, 5:, :], expected)
        rows.append({"root": root, "first_divergent_call": divergence,
                     "incumbent_preserved": True, "all_seed_prefixes_preserved": True,
                     "random_tail_slots_verified": 3})
    base.seal.verify(manifest["dependencies"])
    base.seal.verify(deps)
    base.journal.write_json(OUT, {"dependencies": deps, "rows": rows,
        "scope": "paired journal consistency; not independent replication"})
    print(f"Verified incumbent and prefix-preserving divergence on {len(rows)} paired roots.")


if __name__ == "__main__":
    main()
