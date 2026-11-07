"""Equal-query fixed-prefix tail reoptimization, with separated physical replay."""
import json
import sys
import time

import numpy as np
import torch

from experiments import pc_tworoom_common_tail_retention_20261107_codex as base
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT, PANEL = base.ROOT, base.PANEL
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_SELECTED_PREFIX_EQUAL_CEM_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_selected_prefix_equal_cem_plan_20261107_codex.json"
RESULT = ROOT / "results/pc_tworoom_selected_prefix_equal_cem_result_20261107_codex.json"


def tail_cem(prefix, initial_tails, seed, score):
    prefix = np.asarray(prefix, np.float32)
    initial_tails = np.asarray(initial_tails, np.float32)
    if (prefix.shape != (10, 2) or initial_tails.shape != (4, 15, 2)
            or not np.isfinite(prefix).all() or not np.isfinite(initial_tails).all()
            or np.max(np.abs(prefix)) > 1 or np.max(np.abs(initial_tails)) > 1):
        raise ValueError("invalid actions")
    rng = np.random.default_rng(seed)
    mean, std = np.zeros((15, 2)), np.full((15, 2), .6)
    best_cost, best_chain, generations = np.inf, None, []
    for iteration in range(4):
        tails = np.clip(mean + std * rng.standard_normal((64, 15, 2)), -1, 1).astype(np.float32)
        if iteration == 0:
            tails[:4] = initial_tails
            tails[4] = 0
        chains = np.concatenate((np.broadcast_to(prefix, (64, 10, 2)), tails), axis=1)
        costs = np.asarray(score(chains), dtype=float)
        if costs.shape != (64,) or not np.isfinite(costs).all():
            raise ValueError("invalid scores")
        order = np.argsort(costs, kind="stable")
        if costs[order[0]] < best_cost:
            best_cost, best_chain = float(costs[order[0]]), chains[order[0]].copy()
        elite = tails[order[:8]].astype(float)
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .05)
        generations.append({"tails": tails.tolist(), "costs": costs.tolist(),
                            "elite": order[:8].tolist()})
    return {"actions": best_chain.tolist(), "cost": best_cost, "generations": generations}


def plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    prior = json.loads(base.PLAN.read_text())
    for key, path in (("public_sha256", PANEL.PUBLIC), ("protocol_sha256", base.PROTOCOL),
                      ("runner_sha256", base.__file__), ("wide_sha256", base.wide.__file__),
                      ("repair_sha256", base.wide.repair.__file__)):
        if prior[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    model, scaler, transform, normalizer, boundary = base.wide.setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != prior["model_sha256"] or normalizer != prior["normalizer_sha256"]:
        raise ValueError("model/normalizer drift")
    rows = []
    print("Optimize each sealed prefix with identical 4x64 tail CEM; no physical outcomes are read.", flush=True)
    with np.load(PANEL.PUBLIC, allow_pickle=False) as public, torch.inference_mode():
        for i, old in enumerate(prior["rows"]):
            if old["seed"] != int(public["seeds"][i]):
                raise ValueError("root mismatch")
            prefix_id, tail_id = old["selected"]["descriptor"]
            prefix = np.asarray(old["prefixes"][prefix_id], np.float32)
            tails = np.asarray(old["tails"][5:], np.float32)
            ctx = {"pixels": transform(public["current"][i])[None, None, None].cuda(),
                   "goal": transform(public["goal"][i])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
            counts = []

            def score(chains):
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = []
                for begin in range(0, len(chains), 32):
                    part = chains[begin:begin + 32]
                    normal = scaler.transform(part.reshape(-1, 2)).reshape(1, len(part), 5, 10)
                    cost, _ = boundary.score(ctx, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                    out.extend(cost[0].cpu().tolist())
                torch.cuda.synchronize()
                counts.append({"chains": len(chains), "packets": len(chains) * 5,
                               "seconds": time.perf_counter() - start})
                return out

            arms = {}
            for name, p in (("selected", prefix), ("zero", np.zeros_like(prefix)),
                            ("reversed", prefix[::-1])):
                start = len(counts)
                arms[name] = tail_cem(p, tails, 2026110700 + i, score)
                arms[name]["calls"] = counts[start:]
            rows.append({"seed": old["seed"], "prefix_index": prefix_id, "old_tail_index": tail_id, "arms": arms})
            print(f"root={i} costs={ {k: round(v['cost'], 3) for k,v in arms.items()} }", flush=True)
    if len(rows) != 40 or model_fingerprint(model) != fingerprint:
        raise ValueError("panel/model drift")
    with PLAN.open("x") as f:
        json.dump({"track": "primary_rgb", "rows": rows, "prior_plan_sha256": file_hash(base.PLAN),
                   "public_sha256": file_hash(PANEL.PUBLIC), "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer, "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__)}, f, allow_nan=False, separators=(",", ":"))


def replay():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if RESULT.exists():
        raise FileExistsError(RESULT)
    sealed = json.loads(PLAN.read_text())
    for key, path in (("prior_plan_sha256", base.PLAN), ("public_sha256", PANEL.PUBLIC),
                      ("protocol_sha256", PROTOCOL), ("runner_sha256", __file__)):
        if sealed[key] != file_hash(path):
            raise ValueError(f"plan drift: {key}")
    ref = json.loads(PANEL.REFERENCE.read_text())
    if ref["truth_sha256"] != file_hash(PANEL.TRUTH):
        raise ValueError("truth drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(PANEL.PUBLIC, allow_pickle=False) as public, np.load(PANEL.TRUTH, allow_pickle=False) as truth:
            for i, row in enumerate(sealed["rows"]):
                if row["seed"] != int(public["seeds"][i]):
                    raise ValueError("root mismatch")
                traces = {}
                for name, arm in row["arms"].items():
                    chain = np.asarray(arm["actions"], np.float32)
                    if chain.shape != (25, 2) or not np.isfinite(chain).all() or np.abs(chain).max() > 1:
                        raise ValueError("invalid chain")
                    env.reset(seed=row["seed"], options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.render(), public["current"][i])
                    trace = []
                    for action in chain:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append([float(info["distance_to_target"]), bool(term), bool(trunc)])
                    traces[name] = trace
                rows.append({"seed": row["seed"], "traces": traces})
                print(f"root={i} success={ {k: any(x[1] for x in v) for k,v in traces.items()} }", flush=True)
    finally:
        env.close()
    if len(rows) != 40:
        raise ValueError("panel drift")
    with RESULT.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "truth_sha256": file_hash(PANEL.TRUTH),
                   "rows": rows}, f, allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "replay"):
        raise SystemExit("expected plan|replay")
    {"plan": plan, "replay": replay}[sys.argv[1]]()
