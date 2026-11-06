"""Evaluate stored local response against stored actual candidate outcomes."""
import json

import numpy as np

from experiments import pc_v19_e0_same_chain_mechanism_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    out = source.OUT
    manifest = json.loads((out / "manifest.json").read_text())
    audit = json.loads((out / "plans_audit.json").read_text())
    plans = json.loads((out / "plans.json").read_text())
    if audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(out / "plans.json"):
        raise ValueError("unverified plans")
    seal.verify(manifest["dependencies"])
    rows = []
    for row in plans["rows"]:
        if row["status"] != "completed" or seal.digest(out / row["trace"]) != row["trace_sha256"]:
            raise ValueError("missing or changed trace")
        result = row["result"]
        config = result["config"]
        layers = result["child"]["layers"]
        nominals = np.asarray(config["nominals"], np.float32)
        with np.load(out / row["trace"], allow_pickle=False) as trace:
            batches = [(trace[f"actions_{i}"], trace[f"positions_{i}"], trace[f"scores_{i}"])
                       for i in range(row["batches"])]
            for depth, layer in enumerate(layers):
                parents = ([np.empty((0, config["action_dim"]), np.float32)] if depth == 0 else
                           [np.asarray(layers[depth-1]["candidates"][i]["chain"], np.float32)
                            for i in layers[depth-1]["retained"]])
                for i, candidate in enumerate(layer["candidates"]):
                    chain = np.asarray(candidate["chain"], np.float32)
                    nominal_index = candidate["path"][-1]["nominal"]
                    parent_matches = [j for j, parent in enumerate(parents)
                                      if np.array_equal(chain[:len(parent)], parent)]
                    if not parent_matches:
                        raise ValueError("candidate has no retained parent")
                    # Duplicate parents yield the same nominal actions and
                    # response in a deterministic model; choose the first.
                    parent = parent_matches[0]
                    response = next(r for r in layer["responses"]
                                    if r["parent"] == parent and r["nominal"] == nominal_index)
                    nominal = np.concatenate((parents[parent], nominals[nominal_index]))
                    base = []
                    for actions, positions, scores in batches:
                        if actions.shape[1:] == nominal.shape:
                            hits = np.flatnonzero(np.all(actions == nominal, axis=(1, 2)))
                            base.extend(np.r_[positions[j], scores[j]] for j in hits)
                    if not base:
                        raise ValueError("nominal not present in scored actions")
                    for item in base[1:]:
                        np.testing.assert_allclose(item, base[0], rtol=0, atol=1e-12)
                    matrix = np.asarray(response["matrix"], np.float64)
                    delta = (chain[len(parents[parent]):] - nominals[nominal_index]).astype(np.float64).ravel()
                    actual = np.asarray(candidate["feature"], np.float64) - base[0]
                    linear = matrix @ delta
                    error, magnitude, predicted = np.linalg.norm(linear-actual), np.linalg.norm(actual), np.linalg.norm(linear)
                    rows.append({"root": row["root"], "depth": depth, "candidate": i,
                                 "action_max": float(np.max(np.abs(delta))),
                                 "absolute_error": float(error),
                                 "relative_error": float(error/magnitude) if magnitude > 1e-8 else None,
                                 "cosine": float(linear @ actual / (predicted*magnitude))
                                 if min(predicted, magnitude) > 1e-8 else None})
    summary = []
    for depth in range(3):
        subset = [x for x in rows if x["depth"] == depth and x["action_max"] > 0]
        rel = [x["relative_error"] for x in subset if x["relative_error"] is not None]
        summary.append({"depth": depth, "nonzero_candidates": len(subset),
                        "median_absolute_error": float(np.median([x["absolute_error"] for x in subset])),
                        "median_relative_error": float(np.median(rel)) if rel else None,
                        "relative_error_count": len(rel)})
    seal.verify(manifest["dependencies"])
    output = {"plans_sha256": seal.digest(out / "plans.json"), "rows": rows, "summary": summary,
              "scope": "saved-data model-internal linearization; not causal or independent"}
    with (out / "response_nonlinearity_posthoc.json").open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
