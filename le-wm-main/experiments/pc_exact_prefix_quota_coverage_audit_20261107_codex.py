"""Post-hoc physical-action identity and full candidate ceiling on sealed quota run."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_combination_full_pool_coverage_20261107_codex as ceiling
from experiments import pc_exact_prefix_quota_proposal_20261107_codex as source

PROTOCOL = source.geom.ROOT.parent / "docs/PC_EXACT_PREFIX_QUOTA_COVERAGE_AUDIT_PROTOCOL_20261107_CODEX.md"
OUTPUT = source.geom.ROOT / "results/pc_exact_prefix_quota_coverage_audit_20261107_codex.json"


def key(value):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite action")
    # Normalize signed zero because it denotes the same numerical action.
    return np.ascontiguousarray(value + 0.0).tobytes()


def identities(pool, a, b, selections):
    if pool.shape != (8, 15, 2) or a.shape != (8, 10, 2) or b.shape != (64, 10, 2):
        raise ValueError("invalid action pool")
    common = {key(x) for x in pool}
    common.update(key(np.concatenate((prefix[:5], tail))) for prefix in pool for tail in a)
    arms = {}
    for arm, ids in selections.items():
        if len(ids) != 8 or len(set(ids)) != 8 or any(i < 0 or i >= 512 for i in ids):
            raise ValueError("invalid pair indices")
        values = [key(np.concatenate((pool[i // 64, :5], b[i % 64]))) for i in ids]
        arms[arm] = {"repeats_within": len(values) - len(set(values)),
                     "already_common": sum(x in common for x in values),
                     "unique_novel": len(set(values) - common)}
    return {"unique_prefixes": len({key(x[:5]) for x in pool}),
            "unique_b_tails": len({key(x) for x in b}),
            "unique_common_chains": len(common), "arms": arms}


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(source.OUTPUT.read_text())
    if data["dependencies"] != {p: source.base.sha(p) for p in data["dependencies"]}:
        raise ValueError("historical dependency drift")
    rows = []
    for root, (row, task) in enumerate(zip(data["rows"], source.tasks(), strict=True)):
        if row["root"] != root or row["task"] != task or row["per_arm_model_steps"] != 1200:
            raise ValueError("task or budget mismatch")
        offsets = np.random.default_rng(2730000 + root).uniform(-2, 2, 80)
        np.testing.assert_array_equal(offsets, row["offsets"])
        chains = np.stack([source.geom.make_chain(task["start"], task["goal"], x)[0] for x in offsets])
        pool, a, b = chains[:8], chains[8:16, 5:], chains[16:, 5:]
        def score(x):
            return np.linalg.norm(source.base.exact.endpoints(task, x) - np.asarray(task["goal"]), axis=1)
        parents = score(pool)
        aa = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, 8, 5, 2)),
                             np.broadcast_to(a[None], (8, 8, 10, 2))), axis=2).reshape(-1, 15, 2)
        ac = score(aa).reshape(8, 8)
        np.testing.assert_allclose(parents, row["parent_cost"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(ac, row["a_cost"], atol=1e-12, rtol=0)
        combos = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, 64, 5, 2)),
                                 np.broadcast_to(b[None], (8, 64, 10, 2))), axis=2)
        bc = score(combos.reshape(-1, 15, 2)).reshape(8, 64)
        selections = {arm: data["pairs"] for arm, data in row["outcomes"].items()}
        coverage = ceiling.summarize(parents, ac, bc, selections)
        for arm, outcome in row["outcomes"].items():
            np.testing.assert_allclose(bc.ravel()[outcome["pairs"]], outcome["queried_costs"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(coverage["arms"][arm]["selected_score"], outcome["score"], atol=1e-12, rtol=0)
        rows.append({"root": root, "identity": identities(pool, a, b, selections),
                     "coverage": coverage, "b_costs": bc.tolist()})
    if len(rows) != 24:
        raise ValueError("root count mismatch")
    summary = {"roots": len(rows), "pool_improves": sum(r["coverage"]["improving_combinations"] > 0 for r in rows),
               "pool_new_successes": sum(r["coverage"]["pool_has_new_success"] for r in rows),
               "minimum_unique_prefixes": min(r["identity"]["unique_prefixes"] for r in rows),
               "minimum_unique_b_tails": min(r["identity"]["unique_b_tails"] for r in rows),
               "arms": {arm: {"repeats_within": sum(r["identity"]["arms"][arm]["repeats_within"] for r in rows),
                              "already_common": sum(r["identity"]["arms"][arm]["already_common"] for r in rows),
                              "finds_new_success": sum(r["coverage"]["arms"][arm]["found_new_success"] for r in rows),
                              "mean_regret": float(np.mean([r["coverage"]["arms"][arm]["regret_to_pool_best"]
                                                            for r in rows]))} for arm in source.ARMS}}
    with OUTPUT.open("x") as stream:
        json.dump({"protocol": source.base.sha(PROTOCOL), "runner": source.base.sha(Path(__file__)),
                   "source": source.base.sha(source.OUTPUT), "ceiling_helper": source.base.sha(ceiling.__file__),
                   "rows": rows, "summary": summary, "evaluation_only_b_queries": 24 * 512,
                   "scope": "post-hoc exact dynamics; not independent or RGB-only"}, stream, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    audit()
