"""Independently rebuild frozen-bank retention, choice and physical summaries."""
import hashlib
import json

import numpy as np

from experiments import pc_tworoom_quality_gated_retention_20261107_codex as run
from experiments.pc_input_permissions import file_hash

OUT = run.ROOT / "results/pc_tworoom_quality_gated_retention_audit_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan, phys, result = (json.loads(path.read_text()) for path in (run.PLAN, run.PHYSICAL, run.RESULT))
    if (file_hash(run.PLAN) != phys["plan_sha256"] or file_hash(run.POOL) != phys["pool_sha256"]
            or file_hash(run.PLAN) != result["plan_sha256"]
            or file_hash(run.PHYSICAL) != result["physical_sha256"]):
        raise ValueError("artifact binding")
    for key, path in (("old_pool_sha256", run.old.POOL), ("old_plan_sha256", run.old.PLAN),
                      ("source_pool_sha256", run.old.base.source.POOL),
                      ("protocol_sha256", run.PROTOCOL), ("runner_sha256", run.__file__)):
        if plan[key] != file_hash(path):
            raise ValueError(f"source binding {key}")
    counts = {method: 0 for method in run.METHODS}
    with (np.load(run.POOL, allow_pickle=False) as pool,
          np.load(run.old.POOL, allow_pickle=False) as old,
          np.load(run.old.base.source.POOL, allow_pickle=False) as initial):
        for task, actual in zip(plan["rows"], phys["rows"], strict=True):
            root = task["root"]
            if root != actual["root"]:
                raise ValueError("root mismatch")
            a, b = old[f"r{root}_a"].reshape(64, 25, 2), old[f"r{root}_b"].reshape(64, 25, 2)
            ac, ids, bc = pool[f"r{root}_a_cost"], pool[f"r{root}_b_ids"], pool[f"r{root}_b_cost"]
            ic, actions = pool[f"r{root}_inc_cost"], pool[f"r{root}_actions"]
            order = np.argsort(ac.min(axis=1), kind="stable")
            center = ac - ac.mean(axis=1, keepdims=True)
            rng = np.random.default_rng(29114000 + root)
            shuffled = np.stack([row[rng.permutation(8)] for row in center])

            def furthest(v, allowed):
                d = np.square(v - v[order[0]]).sum(axis=1)
                choices = sorted(int(i) for i in allowed if i != order[0])
                return [int(order[0]), max(choices, key=lambda i: (float(d[i]), -i))]

            expected = {"lookahead": order[:2].tolist(), "vector": furthest(center, order),
                        "gated": furthest(center, order[:4]), "shuffled": furthest(shuffled, order[:4])}
            union = np.asarray(sorted({p * 8 + j for method in run.METHODS
                                       for p in expected[method] for j in range(8)}))
            if not np.array_equal(union, ids) or len(ids) != task["union_b_queries"]:
                raise ValueError("union query mismatch")
            inc = []
            for name in ("block_cem", "block_random"):
                idx = int(np.argmin(initial[f"r{root}_{name}_scores"]))
                inc.append(initial[f"r{root}_{name}_actions"][idx])
            for method_idx, method in enumerate(run.METHODS):
                decision = task["decisions"][method]
                if decision["retained"] != expected[method]:
                    raise ValueError("retention mismatch")
                wanted = np.sort(np.asarray([p * 8 + j for p in expected[method] for j in range(8)]))
                positions = np.searchsorted(ids, wanted)
                if not np.array_equal(ids[positions], wanted):
                    raise ValueError("unqueried B")
                ai = int(np.argmin(ac))
                j = int(np.argmin(bc[positions]))
                ii = int(np.argmin(ic))
                choices = [(float(ic[ii]), "incumbent", ii), (float(ac.flat[ai]), "a", ai),
                           (float(bc[positions[j]]), "b", int(wanted[j]))]
                cost, bank, index = min(choices, key=lambda item: item[0])
                expected_action = inc[index] if bank == "incumbent" else (a if bank == "a" else b)[index]
                if (decision["bank"] != bank or decision["index"] != index or decision["cost"] != cost
                        or not np.array_equal(actions[method_idx], expected_action)
                        or hashlib.sha256(np.ascontiguousarray(expected_action).tobytes()).hexdigest()
                        != decision["action_sha256"]):
                    raise ValueError("selection or action mismatch")
                outcome = actual["outcomes"][method]
                if len(outcome["trace"]) != 25 or outcome["ever"] != any(t["terminated"] for t in outcome["trace"]):
                    raise ValueError("physical trace mismatch")
                counts[method] += outcome["ever"]
    if any(counts[m] != result["summary"][m]["ever"] for m in run.METHODS):
        raise ValueError("summary mismatch")
    with OUT.open("x") as stream:
        json.dump({"plan_sha256": file_hash(run.PLAN), "physical_sha256": file_hash(run.PHYSICAL),
                   "auditor_sha256": file_hash(__file__), "roots": len(plan["rows"]),
                   "summary": counts, "scope": "selection reconstruction, not independent model replay"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
