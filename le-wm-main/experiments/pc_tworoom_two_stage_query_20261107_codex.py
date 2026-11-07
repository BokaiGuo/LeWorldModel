"""Two-stage RGB query reproduction; physical truth is never used to plan."""
import json
import sys
import time

import numpy as np
import torch

from experiments import pc_tworoom_common_tail_retention_20261107_codex as base
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT, PANEL = base.ROOT, base.PANEL
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_TWO_STAGE_QUERY_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_two_stage_query_plan_20261107_codex.json"
COMPARISON = ROOT / "results/pc_tworoom_two_stage_query_comparison_20261107_codex.json"


def evaluate_two_stage(pool, permutation, score):
    """Score A once, then only the two retained B rows for each arm."""
    pool = np.asarray(pool, dtype=np.float32)
    if pool.shape != (13, 9, 25, 2) or not np.isfinite(pool).all() or np.abs(pool).max() > 1:
        raise ValueError("invalid pool")
    a = np.asarray(score(pool[:, :5].reshape(65, 25, 2), "A")).reshape(13, 5)
    if not np.isfinite(a).all():
        raise ValueError("invalid A scores")
    retained = base.retain(a, permutation)
    arms = {}
    for name, indices in retained.items():
        b = np.asarray(score(pool[indices, 5:].reshape(8, 25, 2), name)).reshape(2, 4)
        if not np.isfinite(b).all():
            raise ValueError("invalid B scores")
        local = np.unravel_index(np.argmin(b), b.shape)
        arms[name] = {"retained": indices, "b_costs": b.tolist(),
                      "selected": [int(indices[local[0]]), int(local[1])]}
    return {"a_costs": a.tolist(), "arms": arms}


def plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    ref = json.loads(PANEL.REFERENCE.read_text())
    if ref["public_sha256"] != file_hash(PANEL.PUBLIC):
        raise ValueError("public input drift")
    model, scaler, transform, normalizer, boundary = base.wide.setup()
    fingerprint = model_fingerprint(model)
    rows = []
    print("Two-stage fixed-candidate RGB reproduction: score A then retained B only.", flush=True)
    with np.load(PANEL.PUBLIC, allow_pickle=False) as public, torch.inference_mode():
        for i, seed in enumerate(public["seeds"]):
            prefixes, tails = base.libraries(public["current"][i], public["goal"][i])
            pool = base.chains(prefixes, tails).astype(np.float32)
            context = {"pixels": transform(public["current"][i])[None, None, None].cuda(),
                       "goal": transform(public["goal"][i])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            calls = []

            def score(actions, stage):
                torch.cuda.synchronize()
                start = time.perf_counter()
                values, sizes = [], []
                for begin in range(0, len(actions), 32):
                    part = actions[begin:begin + 32]
                    normal = scaler.transform(part.reshape(-1, 2)).reshape(1, len(part), 5, 10)
                    cost, _ = boundary.score(context, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                    values.extend(cost[0].cpu().tolist())
                    sizes.append(len(part))
                torch.cuda.synchronize()
                calls.append({"stage": stage, "batches": sizes, "chains": len(actions),
                              "packets": 5 * len(actions), "seconds": time.perf_counter() - start})
                return values

            result = evaluate_two_stage(pool, np.random.default_rng(20261107 + i).permutation(13), score)
            rows.append({"seed": int(seed), "prefixes": prefixes.tolist(), "tails": tails.tolist(),
                         "calls": calls, **result})
            print(f"root={i} selected={ {k: v['selected'] for k, v in result['arms'].items()} }", flush=True)
    if len(rows) != 40 or model_fingerprint(model) != fingerprint:
        raise ValueError("panel/model drift")
    with PLAN.open("x") as f:
        json.dump({"track": "primary_rgb", "rows": rows, "public_sha256": file_hash(PANEL.PUBLIC),
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer,
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "base_sha256": file_hash(base.__file__), "wide_sha256": file_hash(base.wide.__file__),
                   "repair_sha256": file_hash(base.wide.repair.__file__)}, f, allow_nan=False)


def compare():
    if COMPARISON.exists():
        raise FileExistsError(COMPARISON)
    new, old, replay = (json.loads(p.read_text()) for p in (PLAN, base.PLAN, base.RESULT))
    if replay["plan_sha256"] != file_hash(base.PLAN) or replay["truth_sha256"] != file_hash(PANEL.TRUTH):
        raise ValueError("prior replay provenance mismatch")
    for key in ("public_sha256", "model_sha256", "normalizer_sha256", "wide_sha256", "repair_sha256"):
        if old[key] != new[key]:
            raise ValueError(f"dependency drift: {key}")
    for key, path in (("protocol_sha256", PROTOCOL), ("runner_sha256", __file__),
                      ("base_sha256", base.__file__)):
        if new[key] != file_hash(path):
            raise ValueError(f"new plan drift: {key}")
    for key, path in (("protocol_sha256", base.PROTOCOL), ("runner_sha256", base.__file__)):
        if old[key] != file_hash(path):
            raise ValueError(f"old plan drift: {key}")
    if not (len(new["rows"]) == len(old["rows"]) == len(replay["rows"]) == 40):
        raise ValueError("root count mismatch")
    rows = []
    for n, o, r in zip(new["rows"], old["rows"], replay["rows"]):
        if n["seed"] != o["seed"] or n["seed"] != r["seed"]:
            raise ValueError("root mismatch")
        for key in ("prefixes", "tails"):
            np.testing.assert_array_equal(np.asarray(n[key], np.float32), np.asarray(o[key], np.float32))
        old_cost = np.asarray(o["costs"])
        arms = {}
        for name, arm in n["arms"].items():
            indices = arm["retained"]
            p, t = arm["selected"]
            old_b = old_cost[indices, 5:]
            trace = r["physical"][p][t + 5]
            arms[name] = {"retained_equal": indices == o["retained"][name],
                          "selected_equal": [p, t] == o["selected"][name],
                          "b_max_abs_diff": float(np.max(np.abs(np.asarray(arm["b_costs"]) - old_b))),
                          "success_from_verified_prior_replay": any(step[1] for step in trace)}
        rows.append({"seed": n["seed"],
                     "a_max_abs_diff": float(np.max(np.abs(np.asarray(n["a_costs"]) - old_cost[:, :5]))),
                     "arms": arms})
    with COMPARISON.open("x") as f:
        json.dump({"new_plan_sha256": file_hash(PLAN), "old_plan_sha256": file_hash(base.PLAN),
                   "old_replay_sha256": file_hash(base.RESULT), "rows": rows}, f, allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "compare"):
        raise SystemExit("expected plan|compare")
    {"plan": plan, "compare": compare}[sys.argv[1]]()
