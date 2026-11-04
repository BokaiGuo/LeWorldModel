"""Verify common initial context and CEM own-carry policy for PushT pair."""
import json

import numpy as np

from experiments import pc_v7_pusht_matched_warm_cem_dev_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.PREFIX.with_name(run.PREFIX.name + "_pair_audit.json")


def file(prefix, arm, name):
    return prefix.with_name(prefix.name + f"_{arm}_{name}")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    audit = json.loads(run.PREFIX.with_name(run.PREFIX.name + "_audit.json").read_text())
    v7_audit = json.loads(run.V7_AUDIT.read_text())
    if (audit["manifest_sha256"] != seal.digest(run.MANIFEST) or audit["result_sha256"] != seal.digest(run.RESULT)
            or v7_audit["manifest_sha256"] != seal.digest(run.V7_MANIFEST)
            or v7_audit["result_sha256"] != seal.digest(run.V7_RESULT)):
        raise ValueError("audit binding mismatch")
    v7_context = file(run.v7.PREFIX, "response", "context0.npz")
    cem_context = file(run.PREFIX, "smooth_cem", "context0.npz")
    with np.load(v7_context, allow_pickle=False) as first, np.load(cem_context, allow_pickle=False) as second:
        for key in ("frames", "history", "goal"):
            np.testing.assert_array_equal(first[key], second[key])
    v7_execution = json.loads(file(run.v7.PREFIX, "response", "execution.json").read_text())
    cem_execution = json.loads(file(run.PREFIX, "smooth_cem", "execution.json").read_text())
    if v7_execution["status"] != "completed" or cem_execution["status"] != "completed":
        raise ValueError("incomplete pair")
    np.testing.assert_array_equal(v7_execution["outcomes"][0]["state"], cem_execution["outcomes"][0]["state"])
    previous = None
    for step in range(3):
        decision = json.loads(file(run.PREFIX, "smooth_cem", f"decision{step}.json").read_text())
        selected = np.asarray(decision["result"]["selected"]["chain"], np.float32)
        if decision["step"] != step or decision["counts"]["prediction_token_samples"] != 1920:
            raise ValueError("decision or budget mismatch")
        if step == 0:
            if decision["carry_in"] is not None:
                raise ValueError("unexpected first carry")
        else:
            np.testing.assert_array_equal(np.asarray(decision["carry_in"], np.float32), previous[5:])
            np.testing.assert_array_equal(np.asarray(decision["result"]["carry"], np.float32), previous[5:])
            initial = decision["trace"][0]
            np.testing.assert_array_equal(np.asarray(initial["raw"][0], np.float32), np.zeros_like(previous[5:]))
            np.testing.assert_array_equal(np.asarray(initial["raw"][1], np.float32), previous[5:])
            if initial["scores"] != [decision["result"]["zero_score"], decision["result"]["carry_score"]]:
                raise ValueError("seed score mismatch")
            restarts = decision["result"]["restarts"]
            counts = [x["samples"] for x in restarts]
            if ([x["name"] for x in restarts] != ["zero", "carry"] or abs(counts[0] - counts[1]) > 1
                    or sum(counts) + 2 != 1920 // (3-step)):
                raise ValueError("restart allocation mismatch")
        previous = selected
    run.live.HOST.write_new(OUT, {"v7_audit_sha256": seal.digest(run.V7_AUDIT),
        "cem_audit_sha256": seal.digest(run.PREFIX.with_name(run.PREFIX.name + "_audit.json")),
        "initial_context_matches": ["frames", "history", "goal", "physical_state"],
        "own_carry_and_restarts_verified": True,
        "scope": "same root; subsequent physical states differ; no independent provenance claim"})
    print("Initial contexts match; CEM carry and equal-budget restarts verified.", flush=True)


if __name__ == "__main__":
    main()
