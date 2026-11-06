"""Inspect saved v18/v19 refinement seeds without issuing model calls."""
import json

import numpy as np

from experiments import pc_v19_retention_fix_e0_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    out = source.OUT
    manifest = json.loads((out / "manifest.json").read_text())
    audit = json.loads((out / "audit.json").read_text())
    plans = json.loads((out / "plans.json").read_text())
    if audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(out / "plans.json"):
        raise ValueError("source audit mismatch")
    seal.verify(manifest["dependencies"])
    rows = []
    for root in range(len(manifest["tasks"])):
        values = {}
        for arm in ("v18", "v19"):
            row = next(row for row in plans["rows"] if row["root"] == root and row["arm"] == arm)
            if row["status"] != "completed" or seal.digest(out / row["trace"]) != row["trace_sha256"]:
                raise ValueError("incomplete or altered trace")
            result = row["result"]["result"]
            child = result["child"]
            final_candidates = np.asarray([c["chain"] for c in child["layers"][-1]["candidates"]], np.float32)
            seed = np.asarray(child["refinement"][0]["mean_before"], np.float32)
            with np.load(out / row["trace"], allow_pickle=False) as trace:
                count = len(row["result"]["calls"])
                matching = [i for i in range(count) if trace[f"actions_{i}"].shape == final_candidates.shape
                            and np.array_equal(trace[f"actions_{i}"], final_candidates)]
                if len(matching) != 1:
                    raise ValueError("final candidate batch is not uniquely identified")
                boundary = matching[0]
                best = float("inf")
                seed_occurrences, bank_match = [], False
                for i in range(boundary + 1):
                    actions, scores = trace[f"actions_{i}"], trace[f"scores_{i}"]
                    if actions.shape[1:] != seed.shape:
                        continue
                    best = min(best, float(scores.min()))
                    indices = np.flatnonzero(np.all(actions == seed, axis=(1, 2)))
                    if len(indices):
                        seed_occurrences.extend((i, float(scores[j])) for j in indices)
                        bank_match |= i == 0
                if not seed_occurrences:
                    raise ValueError("refinement seed was not previously scored")
                seed_score = min(score for _, score in seed_occurrences)
                if not np.isclose(seed_score, best, rtol=0, atol=1e-12):
                    raise ValueError("refinement seed was not the best pre-refinement chain")
                values[arm] = {"seed": seed, "best_before_refinement": best,
                               "seed_matches_global_bank": bank_match,
                               "seed_batches": [item[0] for item in seed_occurrences],
                               "refinement_calls": len(child["refinement"]),
                               "selected_score": result["selected_score"]}
        rows.append({"root": root, "same_refinement_seed": bool(np.array_equal(values["v18"]["seed"], values["v19"]["seed"])),
                     "same_refinement_call_count": values["v18"]["refinement_calls"] == values["v19"]["refinement_calls"],
                     "arms": {arm: {key: value for key, value in data.items() if key != "seed"}
                              for arm, data in values.items()}})
    seal.verify(manifest["dependencies"])
    result = {"source_plans_sha256": seal.digest(out / "plans.json"),
              "source_audit_sha256": seal.digest(out / "audit.json"), "rows": rows,
              "scope": "saved trace provenance, not a causal ablation"}
    with (out / "refinement_provenance_posthoc.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
