"""Saved-data audit for the timed Reacher feedback wiring diagnostic."""
import json

import numpy as np

from experiments import pc_timed_feedback_reacher_wiring_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def audit():
    out = run.OUT
    manifest = json.loads((out / "manifest.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    if summary["manifest_sha256"] != seal.digest(out / "manifest.json"):
        raise ValueError("manifest hash mismatch")
    seal.verify(manifest["dependencies"])
    rows = []
    for arm in run.ARMS:
        execution = json.loads((out / f"{arm}_execution.json").read_text())
        if execution["status"] != "completed" or len(execution["events"]) != 3:
            rows.append({"arm": arm, "status": "failed", "error": execution["error"]})
            continue
        previous, gaps = None, []
        for step, event in enumerate(execution["events"]):
            decision_path = out / f"{arm}_decision{step}.json"
            decision = json.loads(decision_path.read_text())
            if (seal.digest(decision_path) != event["receipt"]["decision_sha256"]
                    or event["receipt"]["sealed_ns"] >= event["execution_return_ns"]
                    or decision["manifest_sha256"] != seal.digest(out / "manifest.json")
                    or decision["context_binding"]["context_sha256"] != seal.digest(out / f"{arm}_context{step}.npz")
                    or decision["remaining_steps"] != 15 - 5 * step
                    or decision["dispatch"] != event["dispatch"]):
                raise ValueError("decision, clock, or execution mismatch")
            selected = np.asarray(decision["selected"], np.float32)
            if previous is not None:
                np.testing.assert_array_equal(decision["carry_in"], previous[5:])
            elif decision["carry_in"] is not None or decision["carry_score"] is not None:
                raise ValueError("unexpected initial carry")
            trace_path = out / decision["trace"]
            if seal.digest(trace_path) != decision["trace_sha256"]:
                raise ValueError("trace hash mismatch")
            tokens, best, found = 0, float("inf"), False
            with np.load(trace_path, allow_pickle=False) as data:
                for i, call in enumerate(decision["calls"]):
                    actions, scores = data[f"actions_{i}"], data[f"scores_{i}"]
                    if (actions.dtype != np.float32 or not np.isfinite(actions).all()
                            or np.any(np.abs(actions) > 1) or actions.shape[1] % 5
                            or scores.shape != (len(actions),) or not np.isfinite(scores).all()):
                        raise ValueError("invalid actions or scores")
                    cost = len(actions) * actions.shape[1] // 5
                    if call["prediction_tokens"] != cost or call["status"] != "completed":
                        raise ValueError("invalid call ledger")
                    tokens += cost
                    if i == 0 and previous is not None:
                        np.testing.assert_array_equal(actions[1], previous[5:])
                        np.testing.assert_allclose(scores[1], decision["carry_score"], rtol=0, atol=1e-12)
                    if actions.shape[1:] == selected.shape:
                        best = min(best, float(scores.min()))
                        found |= bool(np.any(np.all(actions == selected, axis=(1, 2))))
            if tokens != decision["resource"]["prediction_tokens"] or tokens > 1920 or not found:
                raise ValueError("budget or scored-chain mismatch")
            gaps.append(float(decision["planner_result"]["selected_score"] - best))
            previous = selected
        rows.append({"arm": arm, "status": "passed", "decisions": 3,
                     "selected_minus_best_scored": gaps})
    seal.verify(manifest["dependencies"])
    output = {"status": "passed" if all(row["status"] == "passed" for row in rows) else "incomplete",
              "summary_sha256": seal.digest(out / "summary.json"), "rows": rows,
              "scope": "saved-data consistency; no independent replay or isolation certification"}
    with (out / "audit.json").open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    audit()
