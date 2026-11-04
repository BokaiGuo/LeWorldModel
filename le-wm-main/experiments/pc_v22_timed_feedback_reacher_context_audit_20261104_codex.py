"""Audit RGB context bindings and paired final scores for saved decisions."""
import json

import numpy as np

from experiments import pc_v22_timed_feedback_reacher_wiring_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    out = run.OUT
    manifest = json.loads((out / "manifest.json").read_text())
    first = json.loads((out / "audit.json").read_text())
    if first["status"] != "passed" or first["summary_sha256"] != seal.digest(out / "summary.json"):
        raise ValueError("first audit missing or inconsistent")
    seal.verify(manifest["dependencies"])
    initial, rows = None, []
    for arm in run.ARMS:
        execution = json.loads((out / f"{arm}_execution.json").read_text())
        if execution["status"] != "completed":
            raise ValueError("incomplete execution")
        with np.load(out / f"{arm}_context0.npz", allow_pickle=False) as context:
            current = {key: context[key].copy() for key in ("frames", "root", "goal", "history")}
        if initial is None:
            initial = current
        else:
            for key in initial:
                np.testing.assert_array_equal(initial[key], current[key])
        for step, event in enumerate(execution["events"]):
            decision = json.loads((out / f"{arm}_decision{step}.json").read_text())
            selected = np.asarray(decision["selected"], np.float32)
            np.testing.assert_array_equal(selected[:5], event["dispatch"])
            with np.load(out / f"{arm}_context{step + 1}.npz", allow_pickle=False) as successor:
                if run.old.context_lib.rgb_digest(successor["root"]) != event["successor_root_sha256"]:
                    raise ValueError("successor RGB mismatch")
            with np.load(out / decision["trace"], allow_pickle=False) as trace:
                last = len(decision["calls"]) - 1
                actions, scores = trace[f"actions_{last}"], trace[f"scores_{last}"]
                matched = np.flatnonzero(np.all(actions == selected, axis=(1, 2)))
                if not len(matched):
                    raise ValueError("final pair does not contain selected chain")
                if float(decision["planner_result"]["selected_score"]) != float(scores[matched].min()):
                    raise ValueError("selected score does not match final pair")
                if float(scores[matched].min()) != float(scores.min()):
                    raise ValueError("selected chain is not the final-pair minimum")
            rows.append({"arm": arm, "step": step, "context_binding": "passed", "pair": "passed"})
    seal.verify(manifest["dependencies"])
    result = {"status": "passed", "first_audit_sha256": seal.digest(out / "audit.json"),
              "rows": rows, "scope": "saved context consistency, not independent physical replay"}
    with (out / "context_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
