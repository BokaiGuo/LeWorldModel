"""Post-hoc replay of the 290-query panel with scored parents eligible."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_vector_budget290_dev_20261107_codex as old
from experiments.pc_input_permissions import file_hash
from experiments.pc_scored_candidate_selection_20261107_codex import ScoredCandidate, select_scored

OUT = old.ROOT / "results/pc_tworoom_parent_competition_replay_20261107_codex.json"


def select_all(a, parents_incumbents, ids, b, retained):
    a, scores, ids, b = map(np.asarray, (a, parents_incumbents, ids, b))
    if a.shape != (8, 8) or scores.shape != (10,) or ids.ndim != 1 or ids.shape != b.shape:
        raise ValueError("invalid shapes")
    if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= 64) or np.any(np.diff(ids) <= 0):
        raise ValueError("invalid B identities")
    if len(retained) != 2 or len(set(retained)) != 2 or any(type(i) is not int or i < 0 or i >= 8 for i in retained):
        raise ValueError("invalid retained prefixes")
    required = set(i * 8 + j for i in retained for j in range(8))
    available = set(ids.tolist())
    if not required <= available:
        raise ValueError("missing selected B queries")
    # Preserve the historical A-before-incumbent-before-B tie order.
    candidates = [ScoredCandidate("a", i, float(v)) for i, v in enumerate(a.flat)]
    candidates += [ScoredCandidate("inc", i, float(v)) for i, v in enumerate(scores[8:])]
    candidates += [ScoredCandidate("b", int(i), float(v)) for i, v in zip(ids, b, strict=True)
                   if int(i) in required]
    candidates += [ScoredCandidate("parent", i, float(v)) for i, v in enumerate(scores[:8])]
    return select_scored(candidates)


def main():
    plan = json.loads(old.PLAN.read_text())
    for key, path in (("pool_sha256", old.POOL), ("protocol_sha256", old.PROTOCOL),
                      ("runner_sha256", old.__file__), ("rule_sha256", old.prev.rule.__file__),
                      ("source_plan_sha256", old.prev.base.source.PLAN),
                      ("source_pool_sha256", old.prev.base.source.POOL)):
        if plan[key] != file_hash(path):
            raise ValueError(f"source binding changed: {key}")
    rows = []
    with np.load(old.POOL, allow_pickle=False) as pool:
        for item in plan["rows"]:
            root = item["root"]
            parent_inc = pool[f"r{root}_parent_inc_cost"]
            decisions = {}
            for method, original in item["decisions"].items():
                result = select_all(pool[f"r{root}_a_cost"], parent_inc,
                                    pool[f"r{root}_b_indices"], pool[f"r{root}_b_cost"],
                                    original["retained"])
                decisions[method] = {"before": [original["bank"], original["index"]],
                                     "after": [result.source, result.index],
                                     "changed": [result.source, result.index] != [original["bank"], original["index"]]}
            rows.append({"root": root, "row": item["row"],
                         "best_parent_minus_best_incumbent": float(np.min(parent_inc[:8]) - np.min(parent_inc[8:])),
                         "decisions": decisions})
    if [row["root"] for row in rows] != list(range(10, 50)):
        raise ValueError("incomplete panel")
    payload = {"scope": "posthoc selection replay, exposed development panel",
               "plan_sha256": file_hash(old.PLAN), "pool_sha256": file_hash(old.POOL),
               "runner_sha256": file_hash(__file__),
               "selector_sha256": file_hash(Path(__file__).with_name("pc_scored_candidate_selection_20261107_codex.py")),
               "roots": rows,
               "changed_decisions": sum(d["changed"] for row in rows for d in row["decisions"].values()),
               "parents_strictly_better_than_incumbent": sum(row["best_parent_minus_best_incumbent"] < 0 for row in rows)}
    with OUT.open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in payload.items() if k != "roots"}, indent=2))


if __name__ == "__main__":
    main()
