"""Audit v22 refinement generations on the sealed real-RGB decisions."""
import json

import numpy as np

from experiments import pc_v22_timed_feedback_reacher_wiring_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    out = source.OUT
    manifest = json.loads((out / "manifest.json").read_text())
    audit = json.loads((out / "context_audit.json").read_text())
    if audit["status"] != "passed" or audit["first_audit_sha256"] != seal.digest(out / "audit.json"):
        raise ValueError("context audit mismatch")
    seal.verify(manifest["dependencies"])
    rows = []
    for step in range(3):
        decision = json.loads((out / f"v22_decision{step}.json").read_text())
        if seal.digest(out / decision["trace"]) != decision["trace_sha256"]:
            raise ValueError("trace mismatch")
        result = decision["planner_result"]
        config, child = result["config"], result["child"]
        generations, starts = child["refinement"], child["refinement_starts"]
        offset = len(decision["calls"]) - len(generations) - 1
        if offset < 0 or not starts:
            raise ValueError("invalid refinement trace")
        with np.load(out / decision["trace"], allow_pickle=False) as trace:
            for restart, start in enumerate(starts):
                indices = [i for i, generation in enumerate(generations) if generation["restart"] == restart]
                if (len(indices) < 2 or start["samples"] < config["min_refine_samples"]
                        or sum(generations[i]["batch_size"] for i in indices) != start["samples"]):
                    raise ValueError("refinement reserve or generations invalid")
                for first, second in zip(indices, indices[1:]):
                    actions = trace[f"actions_{offset + first}"]
                    scores = trace[f"scores_{offset + first}"]
                    elite = actions[np.argsort(scores, kind="stable")[:min(len(actions), config["refine_elites"])]]
                    np.testing.assert_allclose(generations[second]["mean_before"],
                                               elite.astype(np.float64).mean(axis=0), rtol=0, atol=1e-12)
                    np.testing.assert_allclose(generations[second]["scale_before"],
                                               np.maximum(elite.astype(np.float64).std(axis=0), .07),
                                               rtol=0, atol=1e-12)
        rows.append({"step": step, "remaining_steps": decision["remaining_steps"],
                     "samples": [start["samples"] for start in starts],
                     "generations": [sum(g["restart"] == i for g in generations) for i in range(len(starts))],
                     "tokens": decision["resource"]["prediction_tokens"]})
    seal.verify(manifest["dependencies"])
    result = {"status": "passed", "context_audit_sha256": seal.digest(out / "context_audit.json"),
              "rows": rows, "scope": "saved CEM distribution reconstruction; not search sufficiency"}
    with (out / "refinement_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
