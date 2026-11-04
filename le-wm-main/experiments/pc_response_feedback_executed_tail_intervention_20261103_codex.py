"""Post-hoc replay of the actually dispatched feedback tail, not a PC2 claim."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_response_diverse_beam_same_chain_diag_20261103_codex as diag
from experiments import pc_response_diverse_fixed_terminal_exact_dev_20261103_codex as feedback

PROTOCOL = feedback.prior.prior.ROOT.parent / "docs/PC_RESPONSE_FEEDBACK_EXECUTED_TAIL_INTERVENTION_PROTOCOL_20261103_CODEX.md"
OUTPUT = feedback.PREFIX.with_name("pc_response_feedback_executed_tail_intervention_20261103_codex.json")


def dependencies():
    paths = (Path(__file__), PROTOCOL, feedback.PLANS, feedback.AUDIT, diag.OUTPUT)
    return {str(p): feedback.prior.prior.sha(p) for p in paths}


def analyze(row, old):
    if row["arm"] != "response_diverse" or row["task"] != old["task"]:
        raise ValueError("task or arm mismatch")
    chain = np.concatenate([np.asarray(s["dispatched"]) for s in row["stages"]])
    if chain.shape != (15, 2) or not np.isfinite(chain).all() or np.any(np.abs(chain) > 1):
        raise ValueError("invalid executed chain")
    np.testing.assert_array_equal(chain[:5], old["controls"]["selected"]["prefix"])
    task, goal = row["task"], np.asarray(row["task"]["goal"])
    controls = {}
    for name, previous in old["controls"].items():
        prefix = np.asarray(previous["prefix"])
        actions = np.asarray([np.concatenate((prefix, np.zeros((10, 2)))),
                              np.concatenate((prefix, chain[5:]))])
        points = diag.exp.prior.exact.endpoints(task, actions.astype(np.float32))
        scores = np.linalg.norm(points - goal, axis=1)
        np.testing.assert_allclose(scores[0], previous["common_scores"][0], atol=1e-12, rtol=0)
        controls[name] = {"prefix": prefix.tolist(), "zero_score": float(scores[0]),
                          "executed_tail_score": float(scores[1]),
                          "active_change": float(scores[1] - scores[0]),
                          "own_score": previous["own_score"]}
    np.testing.assert_allclose(controls["selected"]["executed_tail_score"],
                               row["final_distance"], atol=1e-12, rtol=0)
    selected = controls["selected"]
    active = {name: bool(selected["active_change"] < controls[name]["active_change"] - 1e-10)
              for name in diag.CONTROL_NAMES}
    own = {name: bool(selected["own_score"] < controls[name]["own_score"] - 1e-10)
           for name in diag.CONTROL_NAMES}
    return {"root": row["root"], "task": task, "chain": chain.tolist(),
            "nonprogress": old["nonprogress"], "controls": controls, "active_better": active,
            "own_better": own,
            "base_intersection": bool(old["nonprogress"] and all(active[n] and own[n] for n in ("zero", "greedy"))),
            "all_controls_intersection": bool(old["nonprogress"] and all(active[n] and own[n] for n in diag.CONTROL_NAMES))}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    hashes = dependencies()
    plans = json.loads(feedback.PLANS.read_text())
    audit = json.loads(feedback.AUDIT.read_text())
    old = json.loads(diag.OUTPUT.read_text())
    if plans["dependencies"] != feedback.dependencies() or audit["plans_sha256"] != feedback.prior.prior.sha(feedback.PLANS):
        raise ValueError("feedback seal mismatch")
    if old["dependencies"] != diag.dependencies():
        raise ValueError("intervention seal mismatch")
    old_rows = {r["root"]: r for r in old["rows"]}
    rows = [analyze(row, old_rows[row["root"]]) for row in plans["rows"] if row["arm"] == "response_diverse"]
    if len(rows) != 24 or len({r["root"] for r in rows}) != 24 or hashes != dependencies():
        raise ValueError("panel or source drift")
    print("Replaying executed feedback tails under sealed prefixes; retrospective diagnostic only.", flush=True)
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} retrospective intervention rows.", flush=True)


if __name__ == "__main__":
    main()
