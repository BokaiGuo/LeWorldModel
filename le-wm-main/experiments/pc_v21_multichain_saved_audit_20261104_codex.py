"""Audit v21 restart allocation and local incumbents from saved calls."""
import json

import numpy as np

from experiments import pc_v21_multichain_e0_dev_20261104_codex as source
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
    for row in plans["rows"]:
        if row["arm"] != "v21":
            continue
        if row["status"] != "completed" or seal.digest(out / row["trace"]) != row["trace_sha256"]:
            raise ValueError("invalid source run")
        result = row["result"]
        child = result["result"]["child"]
        starts, generations = child["refinement_starts"], child["refinement"]
        counts = [item["samples"] for item in starts]
        if not counts or max(counts) - min(counts) > 1:
            raise ValueError("uneven restart allocation")
        offset = len(result["calls"]) - len(generations) - 1
        with np.load(out / row["trace"], allow_pickle=False) as trace:
            for restart, start in enumerate(starts):
                indices = [i for i, generation in enumerate(generations) if generation["restart"] == restart]
                if not indices or indices != list(range(indices[0], indices[-1] + 1)):
                    raise ValueError("non-contiguous restart")
                seed = np.asarray(generations[indices[0]]["mean_before"], np.float32)
                found = False
                for i in range(offset):
                    actions, scores = trace[f"actions_{i}"], trace[f"scores_{i}"]
                    if actions.shape[1:] == seed.shape:
                        found |= bool(np.any(np.all(actions == seed, axis=(1, 2))
                                             & (np.abs(scores-start["initial_score"]) < 1e-12)))
                if not found:
                    raise ValueError("restart seed was not scored before refinement")
                best, samples = start["initial_score"], 0
                for i in indices:
                    actions, scores = trace[f"actions_{offset+i}"], trace[f"scores_{offset+i}"]
                    if len(actions) != generations[i]["batch_size"]:
                        raise ValueError("generation batch mismatch")
                    samples += len(actions)
                    best = min(best, float(scores.min()))
                    np.testing.assert_allclose(generations[i]["local_best_score"], best, rtol=0, atol=1e-12)
                if samples != start["samples"]:
                    raise ValueError("restart sample mismatch")
                np.testing.assert_allclose(start["best_score"], best, rtol=0, atol=1e-12)
        rows.append({"root": row["root"], "starts": len(starts), "samples": counts,
                     "best_by_start": [item["best_score"] for item in starts]})
    seal.verify(manifest["dependencies"])
    record = {"status": "passed", "plans_sha256": seal.digest(out / "plans.json"), "rows": rows,
              "scope": "saved restart and incumbent reconstruction, not causal attribution"}
    with (out / "multichain_audit.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
