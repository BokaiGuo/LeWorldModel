"""Audit conditional tail intervention without treating oracle as planner."""
import json

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as prior
from experiments import pc_wall_crossing_tail_intervention_20261107_codex as run

OUTPUT = run.trial.ROOT / "results/pc_wall_crossing_tail_intervention_audit_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(run.OUTPUT.read_text())
    if data["dependencies"] != {p: prior.sha(p) for p in data["dependencies"]}:
        raise ValueError("run dependencies changed")
    baseline = json.loads(run.trial.SOURCE.read_text())
    temporal = json.loads(run.trial.OUTPUT.read_text())
    rows = []
    for row, b, t in zip(data["rows"], baseline["rows"], temporal["rows"], strict=True):
        expected = run.choose(t["task"], (("per_step", b["pool_longer"]), ("temporal", t["temporal"])))
        if expected is None:
            if row["eligible"]:
                raise ValueError("spurious selection")
            rows.append({"root": row["root"], "eligible": False})
            continue
        step, initial_cost, arm, prefix_index, candidate_index, chain, successor = expected
        selection = row["selection"]
        if (not row["eligible"] or selection["crossing_step"] != step or selection["arm"] != arm
                or selection["prefix_index"] != prefix_index or selection["candidate_index"] != candidate_index):
            raise ValueError("selection mismatch")
        np.testing.assert_array_equal(selection["initial_chain"], chain)
        np.testing.assert_allclose(selection["prefix_successor"], successor, rtol=0, atol=1e-12)
        scores = {}
        for name in ("crossed", "zero"):
            search = row[name]
            actions = np.asarray(search["actions"])
            costs = np.asarray(search["costs"])
            fixed = chain[:step] if name == "crossed" else np.zeros((step, 2))
            if actions.shape != (384, 15, 2) or np.abs(actions).max() > 1:
                raise ValueError("invalid action bank")
            np.testing.assert_array_equal(actions[:, :step], np.broadcast_to(fixed, (384, step, 2)))
            np.testing.assert_array_equal(actions[0, step:], chain[step:])
            actual = np.linalg.norm(prior.exact.endpoints(row["task"], actions) - row["task"]["goal"], axis=1)
            np.testing.assert_allclose(actual, costs, rtol=0, atol=1e-12)
            if int(np.argmin(costs)) != search["best"]:
                raise ValueError("best mismatch")
            scores[name] = float(costs[search["best"]])
        init_distance = float(np.linalg.norm(np.asarray(row["task"]["start"])[:2] - row["task"]["goal"]))
        prefix_distance = float(np.linalg.norm(successor[:2] - row["task"]["goal"]))
        rows.append({"root": row["root"], "eligible": True, "crossing_step": step,
                     "initial_chain_cost": initial_cost, "initial_distance": init_distance,
                     "prefix_distance": prefix_distance, **scores})
    eligible = [r for r in rows if r["eligible"]]
    summary = {"roots_total": len(rows), "eligible_roots": len(eligible),
               "crossed_success": sum(r["crossed"] < .05 for r in eligible),
               "zero_success": sum(r["zero"] < .05 for r in eligible),
               "crossed_only_success": sum(r["crossed"] < .05 <= r["zero"] for r in eligible),
               "zero_only_success": sum(r["zero"] < .05 <= r["crossed"] for r in eligible),
               "crossed_mean_distance_eligible": float(np.mean([r["crossed"] for r in eligible])),
               "zero_mean_distance_eligible": float(np.mean([r["zero"] for r in eligible])),
               "nonprogress_prefixes": sum(r["prefix_distance"] >= r["initial_distance"] - 1e-12 for r in eligible)}
    summary["diagnostic_continuation_condition"] = summary["crossed_only_success"] >= 3
    output = {"rows": rows, "summary": summary, "input_sha256": prior.sha(run.OUTPUT),
              "audit_sha256": prior.sha(__file__),
              "scope": "post-hoc selected exact prefix; not PC2 or deployable planning"}
    with OUTPUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
