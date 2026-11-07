"""Post-hoc feasibility audit of causal-prefix score canonicalization."""
import json

import numpy as np

from experiments import pc_temporal_effect_retention_20261107_codex as src
from experiments.pc_causal_prefix_scores_20261107_codex import select
from experiments.pc_input_permissions import file_hash

PROTOCOL = src.ROOT.parent / "docs/PC_CAUSAL_PREFIX_SAVED_AUDIT_PROTOCOL_20261107_CODEX.md"
OUTPUT = src.ROOT / "results/pc_causal_prefix_saved_audit_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(src.PLAN.read_text())
    if any(plan["sha256"][key] != file_hash(path) for key, path in
           (("pool", src.POOL), ("old_pool", src.old.POOL), ("initial_pool", src.old.base.source.POOL),
            ("temporal_scores", src.temporal.SCORES))):
        raise ValueError("source drift")
    rows = []
    with (np.load(src.old.POOL, allow_pickle=False) as old,
          np.load(src.old.base.source.POOL, allow_pickle=False) as initial,
          np.load(src.temporal.SCORES, allow_pickle=False) as temporal,
          np.load(src.POOL, allow_pickle=False) as saved):
        for task in plan["rows"]:
            root = task["root"]
            a = old[f"r{root}_a"].reshape(64, 25, 2)
            b = old[f"r{root}_b"].reshape(64, 25, 2)
            inc = np.asarray([initial[f"r{root}_{arm}_actions"][
                int(np.argmin(initial[f"r{root}_{arm}_scores"]))] for arm in ("block_cem", "block_random")])
            a_scores = temporal[f"r{root}_scores"][:, 1:].reshape(64, 5)
            ids, b_scores = saved[f"r{root}_b_ids"], saved[f"r{root}_b_times"]
            inc_scores = saved[f"r{root}_inc_times"]
            methods = {}
            for method in src.METHODS:
                decision = task["decisions"][method]
                allowed = np.sort((np.asarray(decision["retained"])[:, None] * 8 + np.arange(8)).ravel())
                loc = np.searchsorted(ids, allowed)
                if np.any(loc >= len(ids)) or not np.array_equal(ids[loc], allowed):
                    raise ValueError("missing permitted B")
                actions = np.concatenate((inc, a, b[allowed]))
                scores = np.concatenate((inc_scores, a_scores, b_scores[loc]))
                chosen, value, details = select(actions, scores, packet_steps=5, max_disagreement=0.005)
                reversed_order = np.arange(len(actions))[::-1]
                reverse_choice, _, _ = select(actions[reversed_order], scores[reversed_order],
                                               packet_steps=5, max_disagreement=0.005)
                if not np.array_equal(actions[chosen], actions[reversed_order][reverse_choice]):
                    raise ValueError("order-dependent action")
                methods[method] = {"choice_changed": src.digest(actions[chosen]) != decision["action_sha256"],
                                   "corrected_sha256": src.digest(actions[chosen]), "cost": value, **details}
            rows.append({"root": root, "methods": methods})
            print(f"root={root} checked", flush=True)
    summary = {m: {"changed": sum(r["methods"][m]["choice_changed"] for r in rows),
                   "max_spread": max(r["methods"][m]["max_spread"] for r in rows)} for m in src.METHODS}
    with OUTPUT.open("x") as f:
        json.dump({"rows": rows, "summary": summary, "plan_sha256": file_hash(src.PLAN),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "utility_sha256": file_hash(src.ROOT / "experiments/pc_causal_prefix_scores_20261107_codex.py"),
                   "scope": "post-hoc exposed saved candidate audit"}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
