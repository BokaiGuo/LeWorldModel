"""Retrospective equal-query retention on the previously exposed A/B pool."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_block_heldout_effect_20261107_codex as source
from experiments.pc_input_permissions import file_hash

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_VECTOR_RETENTION_POSTHOC_PROTOCOL_20261107_CODEX.md"
AB = Path(str(source.PREFIX) + "_ab_direct_physical.json")
OUTPUT = ROOT / "results/pc_tworoom_vector_retention_posthoc_20261107_codex.json"
METHODS = ("parent", "lookahead", "vector", "shuffled")


def retention(a_cost, parent, seed):
    a_cost = np.asarray(a_cost, dtype=float)
    parent = np.asarray(parent, dtype=float)
    if a_cost.shape != (8, 8) or parent.shape != (8,) or not np.isfinite(a_cost).all() or not np.isfinite(parent).all():
        raise ValueError("invalid cost input")
    rank_parent = np.argsort(parent, kind="stable")
    rank_look = np.argsort(a_cost.min(axis=1), kind="stable")
    first = int(rank_look[0])
    centered = a_cost - a_cost.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    shuffled = np.stack([row[rng.permutation(8)] for row in centered])

    def distant(values):
        dist = np.square(values - values[first]).sum(axis=1)
        dist[first] = -np.inf
        return [first, int(np.argmax(dist))]

    return {"parent": rank_parent[:2].tolist(), "lookahead": rank_look[:2].tolist(),
            "vector": distant(centered), "shuffled": distant(shuffled)}


def select_chain(a_cost, b_cost, retained):
    a_cost, b_cost = np.asarray(a_cost), np.asarray(b_cost)
    retained = np.asarray(retained, dtype=int)
    if (a_cost.shape != (8, 8) or b_cost.shape != (8, 8)
            or retained.shape != (2,) or len(set(retained.tolist())) != 2
            or np.any(retained < 0) or np.any(retained >= 8)
            or not np.isfinite(a_cost).all() or not np.isfinite(b_cost).all()):
        raise ValueError("invalid candidates or scores")
    ai = int(np.argmin(a_cost))
    # Sort flattened original indices so equal B scores resolve by original identity.
    indices = np.sort((retained[:, None] * 8 + np.arange(8)).reshape(-1))
    bi = int(indices[np.argmin(b_cost.reshape(-1)[indices])])
    return ("a", ai) if a_cost.flat[ai] <= b_cost.flat[bi] else ("b", bi)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(source.PLAN.read_text())
    physical = json.loads(source.PHYSICAL.read_text())
    ab = json.loads(AB.read_text())
    if (plan["pool_sha256"] != file_hash(source.POOL)
            or physical["pool_sha256"] != file_hash(source.POOL)
            or physical["plan_sha256"] != file_hash(source.PLAN)
            or ab["pool_sha256"] != file_hash(source.POOL)
            or ab["plan_sha256"] != file_hash(source.PLAN)
            or ab["b_physical_sha256"] != file_hash(source.PHYSICAL)):
        raise ValueError("source binding mismatch")
    rows = []
    with np.load(source.POOL, allow_pickle=False) as pool:
        for root in range(10):
            a_cost = pool[f"r{root}_a_cost"]
            b_cost = pool[f"r{root}_b_cost"]
            retained = retention(a_cost, pool[f"r{root}_parent"], 29114000 + root)
            decisions = {m: select_chain(a_cost, b_cost, retained[m]) for m in METHODS}
            raw = physical["rows"][root]
            direct = ab["rows"][root]
            if raw["root"] != root or direct["root"] != root or len(raw["outcomes"]) != 64:
                raise ValueError("root or outcome count mismatch")
            if direct["choices"]["a"] != int(np.argmin(a_cost)):
                raise ValueError("A replay does not bind model best")
            outcomes = {}
            for method, (bank, index) in decisions.items():
                outcome = direct["a"] if bank == "a" else raw["outcomes"][index]
                trace = outcome["trace"]
                if len(trace) != 25 or outcome["ever"] != any(x["terminated"] for x in trace):
                    raise ValueError("invalid outcome trace")
                if outcome["prefix_success"] != any(x["terminated"] for x in trace[:10]):
                    raise ValueError("invalid prefix success")
                covered = [i for i in retained[method] if any(
                    x["ever"] and not x["prefix_success"] for x in raw["outcomes"][i*8:(i+1)*8])]
                outcomes[method] = {"retained": retained[method], "bank": bank, "index": index,
                                    "ever": outcome["ever"], "prefix_success": outcome["prefix_success"],
                                    "final_distance": outcome["final_distance"], "useful_prefixes": covered}
            rows.append({"root": root, "row": raw["row"], "methods": outcomes})
    summary = {}
    for method in METHODS:
        items = [row["methods"][method] for row in rows]
        summary[method] = {"ever": sum(x["ever"] for x in items),
                           "mean_final_distance": float(np.mean([x["final_distance"] for x in items])),
                           "roots_with_useful_prefix": sum(bool(x["useful_prefixes"]) for x in items),
                           "useful_prefixes": sum(len(x["useful_prefixes"]) for x in items),
                           "a_selected": sum(x["bank"] == "a" for x in items),
                           "wins_vs_lookahead": sum(row["methods"][method]["ever"] and
                                                    not row["methods"]["lookahead"]["ever"] for row in rows),
                           "losses_vs_lookahead": sum(not row["methods"][method]["ever"] and
                                                      row["methods"]["lookahead"]["ever"] for row in rows)}
    summary["development_exit"] = (summary["vector"]["ever"] > summary["lookahead"]["ever"]
                                   and summary["vector"]["ever"] > summary["shuffled"]["ever"])
    with OUTPUT.open("x") as f:
        json.dump({"rows": rows, "summary": summary, "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "plan_sha256": file_hash(source.PLAN),
                   "pool_sha256": file_hash(source.POOL), "physical_sha256": file_hash(source.PHYSICAL),
                   "a_replay_sha256": file_hash(AB),
                   "scope": "posthoc exposed task and candidates; not independent or PC gate"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
