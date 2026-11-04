"""Verify the reserved multi-generation refinement on saved v22 traces."""
import json

import numpy as np

from experiments import pc_v22_e0_same_chain_mechanism_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    out = source.OUT
    manifest = json.loads((out / "manifest.json").read_text())
    plans = json.loads((out / "plans.json").read_text())
    audit = json.loads((out / "plans_audit.json").read_text())
    if audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(out / "plans.json"):
        raise ValueError("plan audit mismatch")
    seal.verify(manifest["dependencies"])
    rows = []
    for row in plans["rows"]:
        if row["status"] != "completed" or seal.digest(out / row["trace"]) != row["trace_sha256"]:
            raise ValueError("incomplete or altered plan")
        child, config = row["result"]["child"], row["result"]["config"]
        generations, starts = child["refinement"], child["refinement_starts"]
        offset = row["batches"] - len(generations) - 1
        with np.load(out / row["trace"], allow_pickle=False) as trace:
            for restart, start in enumerate(starts):
                indices = [i for i, generation in enumerate(generations) if generation["restart"] == restart]
                if len(indices) < 2 or sum(generations[i]["batch_size"] for i in indices) != start["samples"]:
                    raise ValueError("refinement was not iterative")
                if start["samples"] < config["min_refine_samples"]:
                    raise ValueError("refinement reserve violated")
                for first, second in zip(indices, indices[1:]):
                    actions, scores = trace[f"actions_{offset+first}"], trace[f"scores_{offset+first}"]
                    elite = actions[np.argsort(scores, kind="stable")[:min(len(actions), config["refine_elites"])]]
                    expected_mean = elite.astype(np.float64).mean(axis=0)
                    expected_scale = np.maximum(elite.astype(np.float64).std(axis=0), .07)
                    np.testing.assert_allclose(generations[second]["mean_before"], expected_mean, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(generations[second]["scale_before"], expected_scale, rtol=0, atol=1e-12)
        rows.append({"root": row["root"], "starts": len(starts),
                     "samples": [item["samples"] for item in starts],
                     "generations": [sum(g["restart"] == i for g in generations) for i in range(len(starts))]})
    seal.verify(manifest["dependencies"])
    result = {"status": "passed", "plans_sha256": seal.digest(out / "plans.json"), "rows": rows,
              "scope": "saved distribution-update reconstruction; not an optimization sufficiency guarantee"}
    with (out / "refinement_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
