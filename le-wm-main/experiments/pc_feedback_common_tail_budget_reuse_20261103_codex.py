"""Retrospective 119 recorded CEM candidates plus one shared feedback tail."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_response_feedback_executed_tail_intervention_20261103_codex as tail

PROTOCOL = tail.feedback.prior.prior.ROOT.parent / "docs/PC_FEEDBACK_COMMON_TAIL_BUDGET_REUSE_DIAGNOSTIC_PROTOCOL_20261103_CODEX.md"
OUTPUT = tail.OUTPUT.with_name("pc_feedback_common_tail_budget_reuse_20261103_codex.json")


def dependencies():
    paths = (Path(__file__), PROTOCOL, tail.OUTPUT, tail.diag.OUTPUT)
    return {str(path): tail.feedback.prior.prior.sha(path) for path in paths}


def analyze(row, old):
    if row["task"] != old["task"]:
        raise ValueError("task mismatch")
    chain, task = np.asarray(row["chain"]), row["task"]
    goal, controls = np.asarray(task["goal"]), {}
    for name, c in row["controls"].items():
        previous = old["controls"][name]
        np.testing.assert_array_equal(c["prefix"], previous["prefix"])
        requested = np.concatenate([np.asarray(batch["requested"]) for batch in previous["own_trace"]])
        recorded = np.concatenate([np.asarray(batch["scores"]) for batch in previous["own_trace"]])
        if requested.shape != (120, 15, 2) or recorded.shape != (120,):
            raise ValueError("invalid source trace")
        prefix = np.asarray(c["prefix"])
        np.testing.assert_array_equal(requested[:, :5], np.broadcast_to(prefix, (120, 5, 2)))
        actual = np.linalg.norm(tail.diag.exp.prior.exact.endpoints(task, requested.astype(np.float32)) - goal, axis=1)
        np.testing.assert_allclose(actual, recorded, atol=1e-12, rtol=0)
        shared = np.concatenate((prefix, chain[5:]))
        shared_score = np.linalg.norm(tail.diag.exp.prior.exact.endpoints(task, shared[None].astype(np.float32))[0] - goal)
        np.testing.assert_allclose(shared_score, c["executed_tail_score"], atol=1e-12, rtol=0)
        candidates = np.concatenate((actual[:119], [shared_score]))
        best = int(np.argmin(candidates))
        controls[name] = {"old_score": float(np.min(actual)), "shared_score": float(shared_score),
                          "replacement_score": float(candidates[best]),
                          "replacement_source": "shared_tail" if best == 119 else "recorded_cem",
                          "replacement_index": best}
    own = {name: bool(controls["selected"]["replacement_score"] < controls[name]["replacement_score"] - 1e-10)
           for name in tail.diag.CONTROL_NAMES}
    return {"root": row["root"], "nonprogress": row["nonprogress"], "controls": controls,
            "active_better": row["active_better"], "replacement_better": own,
            "base_intersection": bool(row["nonprogress"] and all(row["active_better"][n] and own[n] for n in ("zero", "greedy")))}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    hashes = dependencies()
    data, old = json.loads(tail.OUTPUT.read_text()), json.loads(tail.diag.OUTPUT.read_text())
    if data["dependencies"] != tail.dependencies() or old["dependencies"] != tail.diag.dependencies():
        raise ValueError("source seal mismatch")
    prior = {r["root"]: r for r in old["rows"]}
    print("Replaying 119 recorded CEM samples plus one shared feedback tail; post-hoc only.", flush=True)
    rows = [analyze(row, prior[row["root"]]) for row in data["rows"]]
    if len(rows) != 24 or len({r["root"] for r in rows}) != 24 or dependencies() != hashes:
        raise ValueError("panel or source drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} post-hoc sample-replacement comparisons.", flush=True)


if __name__ == "__main__":
    main()
