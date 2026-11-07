"""Equal-opportunity model tail search for sealed actual-chain interventions."""
import json
import sys

import h5py
import numpy as np
import torch

from experiments import pc_shared_rgb_executed_chain_intervention_20261107_codex as base
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = base.ROOT
PROTOCOL = ROOT.parent / "docs/PC_SHARED_RGB_EQUAL_TAIL_REPLAN_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_shared_rgb_equal_tail_replan_20261107_codex"
PLAN = BASE.with_name(BASE.name + "_plan.json")
PHYSICAL = BASE.with_name(BASE.name + "_physical.json")
POOL = BASE.with_name(BASE.name + "_pool.npz")


def make_batch(prefix, mean, std, best, noise):
    prefix, mean, std, best, noise = map(np.asarray, (prefix, mean, std, best, noise))
    if (prefix.shape != (5, 2) or mean.ndim != 2 or mean.shape[1] != 2 or not 0 <= len(mean) <= 20
            or std.shape != mean.shape or best.shape != mean.shape or noise.shape != (16, *mean.shape)):
        raise ValueError("invalid prefix or optimizer shape")
    if any(not np.isfinite(a).all() for a in (prefix, mean, std, best, noise)) or np.any(std < 0):
        raise ValueError("invalid optimizer values")
    tails = np.clip(mean[None] + std[None] * noise, -1, 1).astype(np.float32)
    tails[0] = best
    chains = np.zeros((16, 25, 2), dtype=np.float32)
    chains[:, :5] = prefix
    chains[:, 5:5 + len(mean)] = tails
    if np.abs(chains).max() > 1:
        raise ValueError("illegal actions")
    return chains


def plan():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("plan exists")
    previous = json.loads(base.PLAN.read_text())
    for key, path in (("protocol_sha256", base.PROTOCOL), ("runner_sha256", base.__file__),
                      ("ledger_sha256", base.ledger.OUT)):
        if previous[key] != file_hash(path):
            raise ValueError("source drift")
    model, scaler, transform, normalizer = base.ledger.prior.source.block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    oldplan = json.loads(base.ledger.prior.source.PLAN.read_text())
    if fingerprint != previous["model_sha256"] or normalizer != oldplan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, base.PLAN,
                         base.ledger.prior.source.block.prior.inputs.INPUT,
                         base.ledger.prior.source.block.prior.common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224 and TwoRoom action scaler", clock="5 packets x 5 steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, arrays = [], {}
    try:
        with (np.load(base.ledger.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(base.ledger.prior.source.block.prior.common.DATA, "r") as data, torch.inference_mode()):
            for item in previous["rows"]:
                root, task = item["root"], item["row"]
                tail = np.asarray(item["common_tail"], dtype=np.float32).reshape(-1, 2)
                prefixes = np.asarray(item["prefix_candidates"], dtype=np.float32)
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][task])
                actor._set_goal_state(data["proprio"][task + 25])
                ctx = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
                outcomes = {}
                for arm in base.ARMS:
                    wrapper = RGBLeWM(model, audit)
                    prefix = prefixes[item["choices"][arm]]
                    mean, std, incumbent, bestcost = tail.copy(), np.full_like(tail, .4), tail.copy(), np.inf
                    rng = np.random.default_rng(27124000 + root)
                    candidates, costs = [], []
                    for _ in range(3):
                        noise = rng.normal(size=(16, *tail.shape))
                        chains = make_batch(prefix, mean, std, incumbent, noise)
                        norm = scaler.transform(chains.reshape(-1, 2)).reshape(1, 16, 5, 10)
                        scores, _ = wrapper.score(ctx, torch.as_tensor(norm, device="cuda", dtype=torch.float32))
                        values = scores[0].cpu().numpy()
                        if values.shape != (16,) or not np.isfinite(values).all():
                            raise ValueError("invalid scores")
                        candidates.append(chains)
                        costs.append(values)
                        index = int(np.argmin(values))
                        if values[index] < bestcost:
                            bestcost = float(values[index])
                            incumbent = chains[index, 5:5 + len(tail)].copy()
                        elite = chains[np.argsort(values, kind="stable")[:4], 5:5 + len(tail)]
                        mean = elite.mean(axis=0)
                        std = np.maximum(elite.std(axis=0), .05)
                    all_candidates = np.concatenate(candidates)
                    all_costs = np.concatenate(costs)
                    choice = int(np.argmin(all_costs))
                    arrays[f"r{root}_{arm}_chains"] = all_candidates
                    arrays[f"r{root}_{arm}_scores"] = all_costs
                    outcomes[arm] = {"choice": choice, "score": float(all_costs[choice]), "queries": 48,
                                     "receipts": wrapper.receipts}
                rows.append({"root": root, "row": task, "actual_horizon": item["original_length"], "arms": outcomes})
                print(f"root={root} queries=240", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weights changed")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"track": RGB, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "source_plan_sha256": file_hash(base.PLAN), "pool_sha256": file_hash(POOL),
                   "model_sha256": fingerprint, "rows": rows,
                   "scope": "posthoc fixed prefixes and evaluator-provided common seed, not deployable"},
                  f, indent=2, allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan_record = json.loads(PLAN.read_text())
    for key, path in (("protocol_sha256", PROTOCOL), ("runner_sha256", __file__),
                      ("source_plan_sha256", base.PLAN), ("pool_sha256", POOL)):
        if plan_record[key] != file_hash(path):
            raise ValueError("plan drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(POOL, allow_pickle=False) as pool,
              h5py.File(base.ledger.prior.source.block.prior.common.DATA, "r") as data):
            for item in plan_record["rows"]:
                root, task, outcomes = item["root"], item["row"], {}
                for arm in base.ARMS:
                    chains, scores = pool[f"r{root}_{arm}_chains"], pool[f"r{root}_{arm}_scores"]
                    choice = item["arms"][arm]["choice"]
                    if chains.shape != (48, 25, 2) or choice != int(np.argmin(scores)):
                        raise ValueError("candidate or selection mismatch")
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task + 25])
                    trace = []
                    for action in chains[choice, :item["actual_horizon"]]:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                        if term or trunc:
                            break
                    outcomes[arm] = {"ever": any(t["terminated"] for t in trace),
                                     "prefix_success": any(t["terminated"] for t in trace[:5]), "trace": trace}
                rows.append({"root": root, "row": task, "outcomes": outcomes})
    finally:
        env.close()
    summary = {arm: {"ever": sum(r["outcomes"][arm]["ever"] for r in rows),
                     "prefix_success": sum(r["outcomes"][arm]["prefix_success"] for r in rows)}
               for arm in base.ARMS}
    with PHYSICAL.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "rows": rows, "summary": summary,
                   "scope": "posthoc equal-opportunity reoptimized chains"}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
