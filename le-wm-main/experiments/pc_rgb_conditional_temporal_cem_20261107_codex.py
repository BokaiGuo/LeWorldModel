"""Fixed-prefix conditional tail CEM with common noise and sealed RGB scores."""
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_vector_retention_40dev_20261107_codex as old
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

BASE = old.ROOT / "results/pc_rgb_conditional_temporal_cem_20261107_codex"
POOL, PLAN, PHYS, RESULT = [Path(str(BASE) + suffix) for suffix in
                            ("_pool.npz", "_plan.json", "_physical.json", "_result.json")]
PROTOCOL = old.ROOT.parent / "docs/PC_RGB_CONDITIONAL_TEMPORAL_CEM_PROTOCOL_20261107_CODEX.md"
METHODS = ("endpoint", "temporal")
N, K, ROUNDS = 64, 8, 3


def digest(action):
    return hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest()


def sample_tails(mean, std, noise):
    mean, std, noise = map(np.asarray, (mean, std, noise))
    if (mean.shape != (3, 2) or std.shape != (3, 2) or noise.shape != (N, 3, 2)
            or not all(np.isfinite(x).all() for x in (mean, std, noise)) or np.any(std <= 0)):
        raise ValueError("invalid tail distribution")
    return np.clip(mean[None] + std[None] * noise, -1, 1).astype(np.float32)


def optimize(prefix, initial_tail, score, root, mode):
    prefix, initial_tail = np.asarray(prefix), np.asarray(initial_tail)
    if (prefix.shape != (10, 2) or initial_tail.shape != (3, 2) or mode not in METHODS
            or not np.isfinite(prefix).all() or np.any(np.abs(prefix) > 1)):
        raise ValueError("invalid prefix or mode")
    mean = initial_tail.astype(np.float64).copy()
    std = np.full((3, 2), 0.5)
    candidates, scores, records = [], [], []
    for iteration in range(ROUNDS):
        noise = np.random.default_rng(7391700 + root * 10 + iteration).standard_normal((N, 3, 2))
        tails = sample_tails(mean, std, noise)
        actions = np.concatenate((np.broadcast_to(prefix[None], (N, 10, 2)),
                                  np.repeat(tails, 5, axis=1)), axis=1)
        values = np.asarray(score(actions), dtype=np.float64)
        if values.shape != (N, 5) or not np.isfinite(values).all():
            raise ValueError("invalid model temporal scores")
        objective = values[:, -1] if mode == "endpoint" else values[:, 2:].min(axis=-1)
        elite = np.argsort(objective, kind="stable")[:K]
        candidates.append(actions)
        scores.append(values)
        records.append({"iteration": iteration, "mean_before": mean.tolist(), "std_before": std.tolist(),
                        "elite": elite.tolist()})
        mean = tails[elite].mean(axis=0, dtype=np.float64)
        std = np.maximum(tails[elite].std(axis=0, dtype=np.float64), 0.05)
    return np.concatenate(candidates), np.concatenate(scores), records


def choose(incumbents, incumbent_scores, candidates, candidate_scores):
    actions = np.concatenate((incumbents, candidates))
    costs = np.concatenate((incumbent_scores[:, -1], candidate_scores[:, -1]))
    if actions.shape != (2 + N * ROUNDS, 25, 2) or costs.shape != (2 + N * ROUNDS,):
        raise ValueError("invalid candidate population")
    index = int(np.argmin(costs))
    return actions[index], index, float(costs[index])


def seal():
    if POOL.exists() or PLAN.exists():
        raise FileExistsError("sealed output exists")
    prior = json.loads(old.PLAN.read_text())
    source = old.base.source
    common, inputs_path = source.block.prior.common, source.block.prior.inputs.INPUT
    if (prior["source_pool_sha256"] != file_hash(source.POOL)
            or prior["input_sha256"] != file_hash(inputs_path)):
        raise ValueError("source drift")
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != prior["model_sha256"] or normalizer != prior["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, old.PLAN, source.POOL, inputs_path,
                                      common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224 and action scaler",
                         clock="five packets of five environment steps")
    arrays, rows = {}, []
    rows_by_root = {r["root"]: r["row"] for r in prior["rows"]}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (np.load(source.POOL, allow_pickle=False) as initial,
          np.load(inputs_path, allow_pickle=False) as data, torch.inference_mode()):
        wrapper = RGBLeWM(model, audit)
        for root in range(20, 50):
            inc = np.asarray([initial[f"r{root}_{arm}_actions"][
                int(np.argmin(initial[f"r{root}_{arm}_scores"]))] for arm in ("block_cem", "block_random")],
                dtype=np.float32)
            old_cost = np.asarray([np.min(initial[f"r{root}_{arm}_scores"])
                                   for arm in ("block_cem", "block_random")])
            context = {"pixels": transform(data["current"][root])[None, None, None].cuda(),
                       "goal": transform(data["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}

            def score(chains):
                chains = np.asarray(chains, dtype=np.float32)
                if chains.ndim != 3 or chains.shape[1:] != (25, 2) or np.any(np.abs(chains) > 1):
                    raise ValueError("illegal action chain")
                norm = scaler.transform(chains.reshape(-1, 2)).reshape(-1, 5, 10)
                out = []
                for start in range(0, len(norm), 32):
                    batch = torch.as_tensor(norm[None, start:start+32], dtype=torch.float32, device="cuda")
                    endpoint, result = wrapper.score(context, batch)
                    predicted, goal = result["predicted_emb"], result["goal_emb"]
                    if predicted.shape[2] != 6:
                        raise ValueError("model clock mismatch")
                    costs = (predicted[:, :, 1:] - goal[:, -1, None, None]).square().sum(-1)
                    if not torch.allclose(endpoint, costs[:, :, -1], rtol=0, atol=0.001):
                        raise ValueError("model endpoint mismatch")
                    out.append(costs[0].cpu().numpy())
                return np.concatenate(out)

            inc_scores = score(inc)
            err = float(np.max(np.abs(inc_scores[:, -1] - old_cost)))
            if err > 0.005:
                raise ValueError(f"incumbent score mismatch root={root}: {err}")
            parent_id = int(np.argmin(inc_scores[:, -1]))
            parent = inc[parent_id]
            blocks = parent.reshape(5, 5, 2)
            if not np.array_equal(blocks, np.broadcast_to(blocks[:, :1], blocks.shape)):
                raise ValueError("incumbent not block constant")
            prefix, tail = parent[:10], blocks[2:, 0]
            decisions, all_actions, all_scores, selected = {}, [], [], []
            for method in METHODS:
                actions, scores, iterations = optimize(prefix, tail, score, root, method)
                if not np.array_equal(actions[:, :10], np.broadcast_to(prefix, (N * ROUNDS, 10, 2))):
                    raise ValueError("prefix changed")
                selected_action, index, value = choose(inc, inc_scores, actions, scores)
                all_actions.append(actions)
                all_scores.append(scores)
                selected.append(selected_action)
                decisions[method] = {"index": index, "cost": value,
                                     "action_sha256": digest(selected_action),
                                     "iterations": iterations}
            if not np.array_equal(all_actions[0][:N], all_actions[1][:N]):
                raise ValueError("common first-round noise failed")
            arrays[f"r{root}_inc"] = inc
            arrays[f"r{root}_inc_scores"] = inc_scores
            arrays[f"r{root}_actions"] = np.asarray(all_actions)
            arrays[f"r{root}_scores"] = np.asarray(all_scores)
            arrays[f"r{root}_selected"] = np.asarray(selected)
            rows.append({"root": root, "row": rows_by_root[root], "parent": parent_id,
                         "parent_sha256": digest(parent), "decisions": decisions,
                         "incumbent_recompute_error": err, "logical_queries_per_arm": 600 + 2 + N * ROUNDS})
            print(f"root={root} two arms scored {2*N*ROUNDS} tails", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    paths = {"pool": POOL, "old_plan": old.PLAN, "source_pool": source.POOL, "inputs": inputs_path,
             "protocol": PROTOCOL, "runner": __file__}
    with PLAN.open("x") as f:
        json.dump({"rows": rows, "sha256": {k: file_hash(v) for k, v in paths.items()},
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer, "track": RGB,
                   "shared_gpu_seconds_synchronized": elapsed, "scope": "exposed development"}, f, indent=2)


def replay():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYS.exists():
        raise FileExistsError(PHYS)
    plan = json.loads(PLAN.read_text())
    paths = {"pool": POOL, "old_plan": old.PLAN, "source_pool": old.base.source.POOL,
             "inputs": old.base.source.block.prior.inputs.INPUT, "protocol": PROTOCOL, "runner": __file__}
    if any(plan["sha256"][k] != file_hash(v) for k, v in paths.items()):
        raise ValueError("sealed data drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(POOL, allow_pickle=False) as pool, h5py.File(old.base.source.block.prior.common.DATA, "r") as data:
            for task in plan["rows"]:
                root, outcomes = task["root"], {}
                candidates = list(pool[f"r{root}_selected"]) + [pool[f"r{root}_inc"][task["parent"]]]
                for name, action in zip((*METHODS, "incumbent"), candidates, strict=True):
                    expected = (task["parent_sha256"] if name == "incumbent"
                                else task["decisions"][name]["action_sha256"])
                    if digest(action) != expected:
                        raise ValueError("selected action drift")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task["row"]])
                    actor._set_goal_state(data["proprio"][task["row"] + 25])
                    trace = []
                    for item in action:
                        _, _, term, trunc, info = actor.step(item.copy())
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                    outcomes[name] = {"ever": any(x["terminated"] for x in trace),
                                      "final": trace[-1]["terminated"], "trace": trace}
                rows.append({"root": root, "outcomes": outcomes})
                print(f"root={root} physical replay complete", flush=True)
    finally:
        env.close()
    with PHYS.open("x") as f:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN), "pool_sha256": file_hash(POOL)}, f)


def summarize():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, truth = json.loads(PLAN.read_text()), json.loads(PHYS.read_text())
    if truth["plan_sha256"] != file_hash(PLAN) or truth["pool_sha256"] != file_hash(POOL):
        raise ValueError("replay binding drift")
    if [r["root"] for r in truth["rows"]] != list(range(20, 50)):
        raise ValueError("incomplete panel")
    summary = {}
    for m in (*METHODS, "incumbent"):
        data = [r["outcomes"][m] for r in truth["rows"]]
        summary[m] = {"ever": sum(x["ever"] for x in data), "final": sum(x["final"] for x in data),
                      "mean_final_distance": float(np.mean([x["trace"][-1]["distance"] for x in data])),
                      "wins_vs_endpoint": sum(r["outcomes"][m]["ever"] and
                                              not r["outcomes"]["endpoint"]["ever"] for r in truth["rows"]),
                      "losses_vs_endpoint": sum(not r["outcomes"][m]["ever"] and
                                                r["outcomes"]["endpoint"]["ever"] for r in truth["rows"])}
        if m in METHODS:
            summary[m]["incumbent_selected"] = sum(r["decisions"][m]["index"] < 2 for r in plan["rows"])
    a, b = summary["temporal"], summary["endpoint"]
    passed = a["wins_vs_endpoint"] >= 2 and a["losses_vs_endpoint"] <= 1 and a["final"] >= b["final"] - 1
    with RESULT.open("x") as f:
        json.dump({"summary": summary, "development_exit": bool(passed),
                   "plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(PHYS),
                   "scope": "exposed fixed-prefix development"}, f, indent=2, allow_nan=False)
    print(json.dumps({"summary": summary, "development_exit": bool(passed)}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "replay", "summarize"):
        raise SystemExit("expected seal|replay|summarize")
    {"seal": seal, "replay": replay, "summarize": summarize}[sys.argv[1]]()
