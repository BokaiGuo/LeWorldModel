"""Analyze saved v22 RGB planning responses without new model calls."""
import json

import numpy as np

from experiments import pc_v22_timed_feedback_reacher_wiring_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    out = source.OUT
    manifest = json.loads((out / "manifest.json").read_text())
    audit = json.loads((out / "context_audit.json").read_text())
    if audit["status"] != "passed" or audit["first_audit_sha256"] != seal.digest(out / "audit.json"):
        raise ValueError("saved-data audit mismatch")
    seal.verify(manifest["dependencies"])
    rows, spread = [], 0.
    for step in range(3):
        decision = json.loads((out / f"v22_decision{step}.json").read_text())
        if seal.digest(out / decision["trace"]) != decision["trace_sha256"]:
            raise ValueError("trace hash mismatch")
        result = decision["planner_result"]
        config, layers = result["config"], result["child"]["layers"]
        nominals = np.asarray(config["nominals"], np.float32)
        with np.load(out / decision["trace"], allow_pickle=False) as trace:
            batches = [(trace[f"actions_{i}"], trace[f"features_{i}"], trace[f"scores_{i}"])
                       for i in range(len(decision["calls"]))]
            for depth, layer in enumerate(layers):
                parents = ([np.empty((0, config["action_dim"]), np.float32)] if depth == 0 else
                           [np.asarray(layers[depth-1]["candidates"][i]["chain"], np.float32)
                            for i in layers[depth-1]["retained"]])
                for index, candidate in enumerate(layer["candidates"]):
                    chain = np.asarray(candidate["chain"], np.float32)
                    nominal_index = candidate["path"][-1]["nominal"]
                    matches = [j for j, parent in enumerate(parents)
                               if np.array_equal(chain[:len(parent)], parent)]
                    if not matches:
                        raise ValueError("candidate parent missing")
                    parent = matches[0]
                    response = next(item for item in layer["responses"]
                                    if item["parent"] == parent and item["nominal"] == nominal_index)
                    nominal = np.concatenate((parents[parent], nominals[nominal_index]))
                    bases = []
                    for actions, features, scores in batches:
                        if actions.shape[1:] == nominal.shape:
                            hits = np.flatnonzero(np.all(actions == nominal, axis=(1, 2)))
                            bases.extend(np.r_[features[j], scores[j]] for j in hits)
                    if not bases:
                        raise ValueError("nominal probe missing")
                    spread = max(spread, *(float(np.max(np.abs(item - bases[0]))) for item in bases))
                    actual = np.asarray(candidate["feature"], np.float64) - bases[0]
                    delta = (chain[len(parents[parent]):] - nominals[nominal_index]).astype(np.float64).ravel()
                    linear = np.asarray(response["matrix"], np.float64) @ delta
                    if len(actual) != 3 or len(linear) != 3:
                        raise ValueError("unexpected chart and score dimensions")
                    def measure(a, p):
                        actual_norm, predicted_norm = float(np.linalg.norm(a)), float(np.linalg.norm(p))
                        error = float(np.linalg.norm(p - a))
                        return {"absolute_error": error, "actual_norm": actual_norm, "predicted_norm": predicted_norm,
                                "relative_error": error / actual_norm if actual_norm > 1e-8 else None,
                                "cosine": float(p @ a / (actual_norm * predicted_norm))
                                if min(actual_norm, predicted_norm) > 1e-8 else None}
                    rows.append({"step": step, "depth": depth, "candidate": index,
                                 "action_max": float(np.max(np.abs(delta))),
                                 "chart": measure(actual[:2], linear[:2]),
                                 "score": measure(actual[2:], linear[2:])})
    seal.verify(manifest["dependencies"])
    output = {"scope": "model-internal saved-data diagnostic; not PC1 certification",
              "context_audit_sha256": seal.digest(out / "context_audit.json"),
              "max_nominal_batch_spread": spread, "rows": rows}
    with (out / "linearization_posthoc.json").open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"rows": len(rows), "max_nominal_batch_spread": spread}, indent=2))


if __name__ == "__main__":
    main()
