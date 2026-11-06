"""Recompute candidate trust decisions from saved v20 layer records."""
import json

import numpy as np

from experiments import pc_v20_trust_e0_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    out = source.OUT
    manifest = json.loads((out / "manifest.json").read_text())
    audit = json.loads((out / "audit.json").read_text())
    plans = json.loads((out / "plans.json").read_text())
    if audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(out / "plans.json"):
        raise ValueError("source audit mismatch")
    seal.verify(manifest["dependencies"])
    totals = [{"depth": i, "candidates": 0, "admitted": 0} for i in range(3)]
    for row in plans["rows"]:
        if row["arm"] != "v20":
            continue
        if row["status"] != "completed":
            raise ValueError("incomplete v20 source")
        result = row["result"]["result"]
        config = result["config"]
        layers = result["child"]["layers"]
        nominals = np.asarray(config["nominals"], np.float64)
        for depth, layer in enumerate(layers):
            parents = ([np.empty((0, config["action_dim"]), np.float32)] if depth == 0 else
                       [np.asarray(layers[depth-1]["candidates"][i]["chain"], np.float32)
                        for i in layers[depth-1]["retained"]])
            for i, candidate in enumerate(layer["candidates"]):
                chain = np.asarray(candidate["chain"], np.float32)
                parent = next(j for j, p in enumerate(parents) if np.array_equal(chain[:len(p)], p))
                nominal = candidate["path"][-1]["nominal"]
                response = next(r for r in layer["responses"] if r["parent"] == parent and r["nominal"] == nominal)
                delta = (chain[-config["segment_length"]:].astype(np.float64) - nominals[nominal]).ravel()
                actual = np.asarray(candidate["feature"]) - np.asarray(response["nominal_feature"])
                error = float(np.linalg.norm(actual - np.asarray(response["matrix"]) @ delta))
                bound = config["trust_absolute"] + config["trust_relative"] * float(np.linalg.norm(actual))
                recorded = layer["trust"][i]
                np.testing.assert_allclose([error, bound], [recorded["absolute_error"], recorded["bound"]],
                                           rtol=0, atol=1e-12)
                if recorded["candidate"] != i or recorded["admitted"] != (error <= bound):
                    raise ValueError("trust decision mismatch")
                totals[depth]["candidates"] += 1
                totals[depth]["admitted"] += int(recorded["admitted"])
            if any(not layer["trust"][index]["admitted"] for index in layer["retained"]):
                raise ValueError("untrusted candidate retained")
    seal.verify(manifest["dependencies"])
    result = {"status": "passed", "plans_sha256": seal.digest(out / "plans.json"), "depths": totals,
              "scope": "saved trust-rule reconstruction; no causal claim"}
    with (out / "trust_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
