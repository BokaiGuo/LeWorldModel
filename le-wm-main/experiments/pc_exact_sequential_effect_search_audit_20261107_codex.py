"""Replay each sequential query and verify that only its prior history is used."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_sequential_effect_search_20261107_codex as run

OUTPUT = run.geom.ROOT / "results/pc_exact_sequential_effect_search_audit_20261107_codex.json"


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(run.OUTPUT.read_text())
    if data["dependencies"] != {p: run.base.sha(p) for p in data["dependencies"]}:
        raise ValueError("dependency drift")
    rows = []
    for root, (row, task) in enumerate(zip(data["rows"], run.tasks(), strict=True)):
        if row["root"] != root or row["task"] != task or row["per_arm_model_steps"] != 1200:
            raise ValueError("root or budget mismatch")
        offsets = np.random.default_rng(2740000 + root).uniform(-2, 2, 80)
        np.testing.assert_array_equal(offsets, row["offsets"])
        chains = np.stack([run.geom.make_chain(task["start"], task["goal"], x)[0] for x in offsets])
        pool, a, b = chains[:8], chains[8:16, 5:], chains[16:, 5:]
        def score(x):
            return np.linalg.norm(run.base.exact.endpoints(task, x) - np.asarray(task["goal"]), axis=1)
        parent = score(pool)
        a_chains = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, 8, 5, 2)),
                                   np.broadcast_to(a[None], (8, 8, 10, 2))), axis=2).reshape(64, 15, 2)
        ac = score(a_chains).reshape(8, 8)
        np.testing.assert_allclose(parent, row["parent_cost"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(ac, row["a_cost"], atol=1e-12, rtol=0)
        common, common_actions = np.r_[parent, ac.ravel()], np.concatenate((pool, a_chains))
        fixed = {"static": run.prior.choose(run.prior.proposal_predictions(a, b, ac, parent,
                                                                            2741000 + root)["descriptor"]),
                 "random": np.random.default_rng(2742000 + root).choice(512, 8, replace=False).tolist()}
        rng = np.random.default_rng(2741000 + root)
        scrambled = np.stack([x[rng.permutation(8)] for x in ac])
        for arm in run.ARMS:
            recorded = row["outcomes"][arm]
            previous = []
            for step, (idx, val) in enumerate(recorded["history"]):
                expected = fixed[arm][step] if arm in fixed else run.next_pair(
                    a, b, scrambled if arm == "shuffled" else ac, previous, step,
                    "row_mean" if arm == "row_mean" else "descriptor")
                if idx != expected:
                    raise ValueError(f"history mismatch {root} {arm} {step}")
                action = np.concatenate((pool[idx // 64, :5], b[idx % 64]))
                np.testing.assert_allclose(score(action[None]), [val], atol=1e-12, rtol=0)
                previous.append((idx, val))
            if len(previous) != 8 or len({idx for idx, _ in previous}) != 8:
                raise ValueError("invalid history")
            q, c = int(np.argmin([v for _, v in previous])), int(np.argmin(common))
            idx, val = previous[q]
            use_q = val < common[c]
            chosen = np.concatenate((pool[idx // 64, :5], b[idx % 64])) if use_q else common_actions[c]
            np.testing.assert_array_equal(recorded["chain"], chosen)
            np.testing.assert_allclose(recorded["score"], val if use_q else common[c], atol=1e-12, rtol=0)
            if recorded["origin"] != ("queried_B" if use_q else "common"):
                raise ValueError("origin mismatch")
        rows.append({"root": root, "queries_per_arm": 8})
    with OUTPUT.open("x") as stream:
        json.dump({"source": run.base.sha(run.OUTPUT), "runner": run.base.sha(Path(__file__)), "rows": rows,
                   "scope": "same predictor and physical backend; not independent engine"},
                  stream, allow_nan=False)
    print(f"audited {len(rows)} roots, five arms, eight sequential queries each")


if __name__ == "__main__":
    audit()
