"""Fixed-opportunity RGB-only retention development test; truth is evaluation-only."""
import json
import sys
import time

import h5py
import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_tworoom_reference_vs_hold_20261107_codex as inputs
from experiments import pc_tworoom_rgb_action_effect_alignment_20261107_codex as common
from experiments.pc_exact_action_identity_retention_dev_20261106_codex import select

ROOT = common.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_RGB_RETENTION_FIXED_POOL_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_rgb_retention_fixed_pool_plan_20261107_codex.json"
PHYSICAL = ROOT / "results/pc_tworoom_rgb_retention_fixed_pool_physical_20261107_codex.json"
ARMS = ("parent", "lookahead", "descriptor", "shuffled")


def to_physical(normalized, scaler):
    x = np.asarray(normalized)
    raw = scaler.inverse_transform(x.reshape(-1, 2)).reshape(x.shape)
    if not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1 + 1e-6:
        raise ValueError("illegal action")
    return np.clip(raw, -1, 1)


def run_plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    model, scaler, transform, action_hash = common.model_setup()
    fingerprint = model_fingerprint(model)
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, inputs.INPUT,
                                      common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and TwoRoom action scaler", clock="5 packets of 5 steps")
    low = np.broadcast_to((-1 - scaler.mean_) / scaler.scale_, (25, 2))
    high = np.broadcast_to((1 - scaler.mean_) / scaler.scale_, (25, 2))
    zero = np.broadcast_to(scaler.transform(np.zeros((1, 2)))[0], (25, 2)).copy()
    rows, started = [], time.perf_counter()
    with np.load(inputs.INPUT, allow_pickle=False) as data, torch.inference_mode():
        for root in range(10):
            row = int(data["rows"][root])
            wrapper = RGBLeWM(model, audit)
            pixels = transform(data["current"][root])[None, None, None].cuda()
            goal = transform(data["goal"][root])[None, None, None].cuda()
            context = {"pixels": pixels, "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
            calls = 0

            def score(candidate):
                nonlocal calls
                candidate = np.asarray(candidate, dtype=np.float32)
                if candidate.ndim != 3 or candidate.shape[1:] != (25, 2):
                    raise ValueError("invalid chain shape")
                to_physical(candidate, scaler)
                values = []
                for offset in range(0, len(candidate), 32):
                    chunk = torch.as_tensor(candidate[offset:offset + 32].reshape(1, -1, 5, 10),
                                            dtype=torch.float32, device="cuda")
                    cost, _ = wrapper.score(context, chunk)
                    values.extend(cost[0].cpu().numpy().tolist())
                calls += len(candidate)
                return np.asarray(values)

            rng = np.random.default_rng(2718000 + root)
            mean, std = zero.copy(), np.ones((25, 2))
            gen_actions, gen_costs = [], []
            for iteration in range(3):
                candidate = np.clip(mean + rng.normal(size=(100, 25, 2)) * std, low, high)
                candidate[0] = mean
                cost = score(candidate)
                elite = candidate[np.argsort(cost, kind="stable")[:10]]
                mean, std = elite.mean(0), np.maximum(elite.std(0), 1e-5)
                gen_actions.extend(candidate.tolist())
                gen_costs.extend(cost.tolist())
            gen_actions, gen_costs = np.asarray(gen_actions), np.asarray(gen_costs)
            chosen = []
            for idx in np.argsort(gen_costs, kind="stable"):
                if all(not np.array_equal(gen_actions[idx, :10], gen_actions[j, :10]) for j in chosen):
                    chosen.append(int(idx))
                if len(chosen) == 4:
                    break
            if len(chosen) != 4:
                raise ValueError("insufficient unique prefixes")
            rng_global = np.random.default_rng(2718100 + root)
            global_chains = rng_global.uniform(low, high, size=(4, 25, 2))
            global_cost = score(global_chains)
            pool = np.concatenate((gen_actions[chosen], global_chains))
            parent = np.r_[gen_costs[chosen], global_cost]
            rng_tail = np.random.default_rng(2718200 + root)
            tails = rng_tail.uniform(low[10:], high[10:], size=(8, 15, 2))
            cross = np.concatenate((np.broadcast_to(pool[:, None, :10], (8, 8, 10, 2)),
                                    np.broadcast_to(tails[None], (8, 8, 15, 2))), axis=2)
            cross_cost = score(cross.reshape(64, 25, 2)).reshape(8, 8)
            selected = select(cross_cost, parent, parent, 2718300 + root)
            incumbent_chains = np.concatenate((gen_actions, global_chains, cross.reshape(64, 25, 2)))
            incumbent_costs = np.r_[gen_costs, global_cost, cross_cost.ravel()]
            incumbent_idx = int(np.argmin(incumbent_costs))
            refinements = []
            for prefix_id, chain in enumerate(pool):
                rng_refine = np.random.default_rng(2718400 + root * 8 + prefix_id)
                tail_mean, tail_std = chain[10:].copy(), np.ones((15, 2))
                ref_actions, ref_cost = [], []
                for iteration in range(3):
                    tail = np.clip(tail_mean + rng_refine.normal(size=(12, 15, 2)) * tail_std,
                                   low[10:], high[10:])
                    tail[0] = tail_mean
                    candidates = np.concatenate((np.broadcast_to(chain[:10], (12, 10, 2)), tail), axis=1)
                    value = score(candidates)
                    elite = tail[np.argsort(value, kind="stable")[:4]]
                    tail_mean, tail_std = elite.mean(0), np.maximum(elite.std(0), 1e-5)
                    ref_actions.extend(candidates.tolist())
                    ref_cost.extend(value.tolist())
                refinements.append({"actions": ref_actions, "costs": ref_cost})
            outcomes = {}
            for arm in ARMS:
                choices = [(incumbent_costs[incumbent_idx], "incumbent", incumbent_chains[incumbent_idx])]
                for idx in selected[arm]:
                    j = int(np.argmin(refinements[idx]["costs"]))
                    choices.append((refinements[idx]["costs"][j], f"prefix_{idx}",
                                    np.asarray(refinements[idx]["actions"][j])))
                value, origin, chain = min(choices, key=lambda x: x[0])
                outcomes[arm] = {"cost": float(value), "origin": origin,
                                 "actions": to_physical(chain, scaler).tolist()}
            if calls != 300 + 4 + 64 + 8 * 36:
                raise ValueError("scoring counter mismatch")
            rows.append({"root": root, "row": row, "generation_actions": gen_actions.tolist(),
                         "generation_costs": gen_costs.tolist(), "pool": pool.tolist(),
                         "parent_cost": parent.tolist(), "a_tails": tails.tolist(),
                         "a_cost": cross_cost.tolist(), "selected": {arm: selected[arm] for arm in ARMS},
                         "incumbent_index": incumbent_idx, "incumbent_cost": float(incumbent_costs[incumbent_idx]),
                         "refinements": refinements, "outcomes": outcomes, "physical_scores": calls,
                         "counterfactual_deployment_scores": 440})
            print(f"{root+1}/10 row={row} completed", flush=True)
    torch.cuda.synchronize()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weights changed")
    with PLAN.open("x") as stream:
        json.dump({"rows": rows, "model_sha256": fingerprint, "normalizer_action_sha256": action_hash,
                   "input_sha256": file_hash(inputs.INPUT), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "elapsed_seconds": time.perf_counter() - started,
                   "scope": "RGB-only exposed development; offline optimization opportunities, not closed-loop"},
                  stream, separators=(",", ":"), allow_nan=False)


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan = json.loads(PLAN.read_text())
    if (plan["input_sha256"] != file_hash(inputs.INPUT) or plan["protocol_sha256"] != file_hash(PROTOCOL)
            or plan["runner_sha256"] != file_hash(__file__)):
        raise ValueError("plan drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with h5py.File(common.DATA, "r") as data:
            for row in plan["rows"]:
                outcomes = {}
                for arm in ARMS:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    actions = np.asarray(row["outcomes"][arm]["actions"], dtype=np.float32)
                    if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                        raise ValueError("illegal action")
                    trace = []
                    for action in actions:
                        _, _, terminated, truncated, info = actor.step(action)
                        trace.append({"distance": float(info["distance_to_target"]),
                                      "terminated": bool(terminated), "truncated": bool(truncated)})
                    outcomes[arm] = {"trace": trace, "ever_success": any(x["terminated"] for x in trace),
                                     "final_distance": trace[-1]["distance"]}
                rows.append({"root": row["root"], "row": row["row"], "outcomes": outcomes})
    finally:
        env.close()
    with PHYSICAL.open("x") as stream:
        json.dump({"rows": rows, "plan_sha256": file_hash(PLAN), "runner_sha256": file_hash(__file__)},
                  stream, indent=2, allow_nan=False)
    for arm in ARMS:
        print(f"{arm}: {sum(row['outcomes'][arm]['ever_success'] for row in rows)}/{len(rows)}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": run_plan, "physical": physical}[sys.argv[1]]()
