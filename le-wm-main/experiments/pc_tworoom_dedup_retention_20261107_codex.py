"""Distinct-prefix two-stage development experiment, preserving prior artifacts."""
import json
import sys
import time

import numpy as np
import torch

from experiments import pc_tworoom_common_tail_retention_20261107_codex as base
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT, PANEL = base.ROOT, base.PANEL
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_DEDUP_RETENTION_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_dedup_retention_plan_20261107_codex.json"
RESULT = ROOT / "results/pc_tworoom_dedup_retention_result_20261107_codex.json"


def unique_prefixes(prefixes):
    a = np.asarray(prefixes, dtype=np.float32)
    if a.ndim != 3 or a.shape[1:] != (10, 2) or not np.isfinite(a).all() or np.abs(a).max() > 1:
        raise ValueError("invalid prefixes")
    representatives, inverse = [], []
    for i, prefix in enumerate(a):
        existing = next((j for j, k in enumerate(representatives) if np.array_equal(prefix, a[k])), None)
        if existing is None:
            existing = len(representatives)
            representatives.append(i)
        inverse.append(existing)
    if len(representatives) < 2:
        raise ValueError("at least two distinct prefixes required")
    return np.asarray(representatives), np.asarray(inverse)


def retain(a_cost, order):
    a = np.asarray(a_cost, dtype=float)
    order = np.asarray(order)
    if (a.ndim != 2 or a.shape[1] != 5 or len(a) < 2 or not np.isfinite(a).all()
            or order.shape != (len(a),) or not np.array_equal(np.sort(order), np.arange(len(a)))):
        raise ValueError("invalid costs or permutation")
    rank = np.argsort(a[:, :4].min(axis=1), kind="stable")
    anchor = int(rank[0])
    vectors = a[:, 4, None] - a[:, :4]

    def diversity(v):
        distances = ((v - v[anchor]) ** 2).sum(axis=1)
        distances[anchor] = -1
        return [anchor, int(np.argmax(distances))]

    return {"lookahead": rank[:2].tolist(), "descriptor": diversity(vectors),
            "shuffled": diversity(vectors[order])}


def evaluate(prefixes, tails, order_seed, score):
    representatives, inverse = unique_prefixes(prefixes)
    tails = np.asarray(tails, np.float32)
    if tails.shape != (9, 15, 2) or not np.isfinite(tails).all() or np.abs(tails).max() > 1:
        raise ValueError("invalid tails")
    prefix = np.asarray(prefixes, np.float32)[representatives]
    n = len(prefix)
    pool = np.concatenate((np.broadcast_to(prefix[:, None], (n, 9, 10, 2)),
                           np.broadcast_to(tails[None], (n, 9, 15, 2))), axis=2)
    a = np.asarray(score(pool[:, :5].reshape(n * 5, 25, 2), "A"), dtype=float).reshape(n, 5)
    order = np.random.default_rng(order_seed).permutation(n)
    kept = retain(a, order)
    arms = {}
    for name, indices in kept.items():
        b = np.asarray(score(pool[indices, 5:].reshape(8, 25, 2), name), dtype=float).reshape(2, 4)
        if not np.isfinite(b).all():
            raise ValueError("invalid B scores")
        local = np.unravel_index(np.argmin(b), b.shape)
        arms[name] = {"retained": representatives[indices].tolist(),
                      "selected": [int(representatives[indices[local[0]]]), int(local[1])],
                      "b_costs": b.tolist()}
    return {"representatives": representatives.tolist(), "inverse": inverse.tolist(),
            "permutation": order.tolist(), "a_costs": a.tolist(), "arms": arms}


def plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    reference = json.loads(PANEL.REFERENCE.read_text())
    if reference["public_sha256"] != file_hash(PANEL.PUBLIC):
        raise ValueError("public drift")
    model, scaler, transform, normalizer, boundary = base.wide.setup()
    fingerprint = model_fingerprint(model)
    rows = []
    print("Deduplicate exact action prefixes before shared A and retained-only B model queries.", flush=True)
    with np.load(PANEL.PUBLIC, allow_pickle=False) as public, torch.inference_mode():
        for i, seed in enumerate(public["seeds"]):
            prefixes, tails = base.libraries(public["current"][i], public["goal"][i])
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

            outcome = evaluate(prefixes, tails, 20261107 + i, score)
            rows.append({"seed": int(seed), "prefixes": prefixes.tolist(), "tails": tails.tolist(),
                         "calls": calls, **outcome})
            print(f"root={i} distinct={len(outcome['representatives'])}", flush=True)
    if len(rows) != 40 or model_fingerprint(model) != fingerprint:
        raise ValueError("panel/model drift")
    with PLAN.open("x") as f:
        json.dump({"track": "primary_rgb", "rows": rows, "public_sha256": file_hash(PANEL.PUBLIC),
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer,
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "base_sha256": file_hash(base.__file__), "wide_sha256": file_hash(base.wide.__file__),
                   "repair_sha256": file_hash(base.wide.repair.__file__)}, f, allow_nan=False)


def replay_comparison():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    new, old, physical = (json.loads(p.read_text()) for p in (PLAN, base.PLAN, base.RESULT))
    if physical["plan_sha256"] != file_hash(base.PLAN) or physical["truth_sha256"] != file_hash(PANEL.TRUTH):
        raise ValueError("old replay drift")
    for key in ("public_sha256", "model_sha256", "normalizer_sha256", "wide_sha256", "repair_sha256"):
        if new[key] != old[key]:
            raise ValueError(f"dependency drift: {key}")
    for key, path in (("public_sha256", PANEL.PUBLIC), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__), ("base_sha256", base.__file__),
                      ("wide_sha256", base.wide.__file__), ("repair_sha256", base.wide.repair.__file__)):
        if new[key] != file_hash(path):
            raise ValueError(f"new plan drift: {key}")
    for key, path in (("protocol_sha256", base.PROTOCOL), ("runner_sha256", base.__file__)):
        if old[key] != file_hash(path):
            raise ValueError(f"old plan drift: {key}")
    if not (len(new["rows"]) == len(old["rows"]) == len(physical["rows"]) == 40):
        raise ValueError("root count drift")
    rows = []
    for n, o, p in zip(new["rows"], old["rows"], physical["rows"]):
        if n["seed"] != o["seed"] or n["seed"] != p["seed"]:
            raise ValueError("root mismatch")
        for key in ("prefixes", "tails"):
            np.testing.assert_array_equal(np.asarray(n[key], np.float32), np.asarray(o[key], np.float32))
        arms = {}
        for name, arm in n["arms"].items():
            prefix, tail = arm["selected"]
            def success(index, suffix):
                return any(step[1] for step in p["physical"][index][suffix + 5])
            arms[name] = {"success": success(prefix, tail),
                          "retained_useful": sum(any(success(idx, t) for t in range(4)) for idx in arm["retained"]),
                          "selected_index_equal_old": arm["selected"] == o["selected"][name]}
        rows.append({"seed": n["seed"], "distinct": len(n["representatives"]), "arms": arms})
    with RESULT.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "old_plan_sha256": file_hash(base.PLAN),
                   "replay_sha256": file_hash(base.RESULT), "rows": rows}, f, allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "replay_comparison"):
        raise SystemExit("expected plan|replay_comparison")
    {"plan": plan, "replay_comparison": replay_comparison}[sys.argv[1]]()
