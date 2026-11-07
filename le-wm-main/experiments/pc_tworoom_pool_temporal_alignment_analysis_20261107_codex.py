"""Audit aligned vectors and compare temporal candidate selections with physical outcomes."""
import json
from collections import defaultdict

import numpy as np

from experiments import pc_tworoom_pool_temporal_alignment_20261107_codex as run
from experiments.pc_input_permissions import file_hash


def pool_outcome(row):
    pred = np.asarray(row["predicted_cost"])
    real = np.asarray(row["actual_cost"])
    if pred.shape != real.shape or pred.ndim != 2 or pred.shape[0] != 6:
        raise ValueError("bad cost shape")
    any_step = np.asarray([any(t["success"] for t in trace) for trace in row["physical"]])
    packet_step = np.asarray([any(trace[i]["success"] for i in range(4, len(trace), 5))
                              for trace in row["physical"]])
    p, a = int(np.argmin(pred.min(axis=1))), int(np.argmin(real.min(axis=1)))
    return {"pool": bool(any_step.any()), "predicted": bool(any_step[p]), "encoded": bool(any_step[a]),
            "predicted_sampled": bool(packet_step[p]), "encoded_sampled": bool(packet_step[a]),
            "pool_sampled": bool(packet_step.any()), "different_choice": p != a,
            "encoded_only": bool(any_step[a] and not any_step[p]),
            "predicted_only": bool(any_step[p] and not any_step[a])}


def main():
    data = json.loads(run.OUT.read_text())
    if (data["source_sha256"] != file_hash(run.source.OUTPUT) or data["protocol_sha256"] != file_hash(run.PROTOCOL)
            or data["runner_sha256"] != file_hash(run.__file__) or data["feature_sha256"] != file_hash(run.FEATURES)):
        raise ValueError("artifact hash drift")
    errors = defaultdict(list)
    with np.load(run.FEATURES, allow_pickle=False) as features:
        for i, row in enumerate(data["rows"]):
            pred, actual, goal = features[f"p{i}"], features[f"a{i}"], features[f"g{i}"]
            assert pred.shape == actual.shape and pred.shape[0] == 6
            np.testing.assert_allclose(np.square(pred - goal).sum(-1), row["predicted_cost"], atol=1e-3, rtol=0)
            np.testing.assert_allclose(np.square(actual - goal).sum(-1), row["actual_cost"], atol=1e-3, rtol=0)
            for t in range(pred.shape[1]):
                errors[5 * (t + 1)].extend(np.square(pred[:, t] - actual[:, t]).mean(-1).tolist())
    print("feature MSE by physical step", {t: round(float(np.mean(x)), 6) for t, x in errors.items()})
    first = {}
    for row in data["rows"]:
        first.setdefault(row["episode"], row)
    for name, rows in (("all_decisions", data["rows"]),
                       ("non_initial_decisions", [r for r in data["rows"] if not r["initial_success"]]),
                       ("first_decision_per_episode", list(first.values())),
                       ("first_non_initial_episode", [r for r in first.values() if not r["initial_success"]])):
        outcomes = [pool_outcome(row) for row in rows]
        print(name, "n", len(rows), {field: sum(x[field] for x in outcomes) for field in outcomes[0]})


if __name__ == "__main__":
    main()
