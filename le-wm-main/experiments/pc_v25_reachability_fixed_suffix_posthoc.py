"""Join audited same-bank response retention and fixed-opportunity suffix results."""
import itertools
import json
from pathlib import Path

import numpy as np

from experiments import pc_v25_response_random_fixed_opportunity_posthoc_20261104_codex as fixed

panel = fixed.panel
OUT = panel.ROOT / "results/pc_v25_reachability_fixed_suffix_posthoc_20261104_codex.json"
PROTOCOL = panel.ROOT.parent / "docs/PC_V25_REACHABILITY_FIXED_SUFFIX_POSTHOC_PROTOCOL_20261104_CODEX.md"


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    audit = json.loads((fixed.OUT / "audit.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("fixed-opportunity audit not passed")
    fixed_manifest = json.loads((fixed.OUT / "manifest.json").read_text())
    panel.seal.verify(fixed_manifest["dependencies"])
    deps = panel.seal.capture([Path(__file__), PROTOCOL, fixed.OUT / "audit.json",
        fixed.OUT / "manifest.json", *[fixed.OUT / f"root{i:02d}.json" for i in range(16)],
        *[fixed.OUT / f"root{i:02d}_response_selection.json" for i in range(16)],
        *[panel.OUT / f"root{i:02d}_1920_narrow.json" for i in range(16)]])
    rows = []
    for root, task in enumerate(panel.tasks()):
        source = json.loads((panel.OUT / f"root{root:02d}_1920_narrow.json").read_text())
        data = json.loads((fixed.OUT / f"root{root:02d}.json").read_text())
        if source["status"] != "completed" or data["status"] != "completed":
            rows.append({"root": root, "status": "source_failed"})
            continue
        if source["task"] != task or data["task"] != task:
            raise ValueError("task mismatch")
        selection = json.loads((fixed.OUT / f"root{root:02d}_response_selection.json").read_text())
        if data["arms"]["response"]["selection_sha256"] != panel.seal.digest(
                fixed.OUT / f"root{root:02d}_response_selection.json"):
            raise ValueError("fixed bank digest mismatch")
        layer = source["result"]["child"]["layers"][0]
        bank = np.asarray([x["chain"] for x in layer["candidates"]], np.float32)
        np.testing.assert_array_equal(bank, np.asarray(selection["bank"], np.float32))
        scores = data["arms"]["response"]["diagnostic"]["scores"]
        if len(scores) != len(bank):
            raise ValueError("suffix score mismatch")
        alternatives = dict(layer["retention_alternatives"])
        alternatives["lookahead"] = selection["selections"]["lookahead"]
        result = {name: {"indices": indices, "best": min(scores[i] for i in indices)}
                  for name, indices in alternatives.items()}
        pair_counts = {"concordant": 0, "discordant": 0, "ties": 0}
        for left, right in itertools.combinations(layer["reachability"], 2):
            product = ((left["value"] - right["value"]) *
                       (scores[left["candidate"]] - scores[right["candidate"]]))
            key = "concordant" if product > 0 else "discordant" if product < 0 else "ties"
            pair_counts[key] += 1
        added, removed = layer["response_added"], layer["response_removed"]
        rows.append({"root": root, "status": "completed", "rules": result,
                     "added": [{"index": i, "suffix_score": scores[i]} for i in added],
                     "removed": [{"index": i, "suffix_score": scores[i]} for i in removed],
                     "probe_pairs": pair_counts})
    panel.seal.verify(deps)
    panel.journal.write_json(OUT, {"scope": "posthoc fixed-opportunity; not gate evidence",
                                   "dependencies": deps, "rows": rows})
    print(f"Saved {len(rows)} root-level fixed-opportunity comparisons.", flush=True)


if __name__ == "__main__":
    run()
