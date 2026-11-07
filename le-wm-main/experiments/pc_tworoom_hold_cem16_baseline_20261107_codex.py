"""Endpoint-scored CEM16 closed loop on the exposed hold-retention panel."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_rgb_hold_retention_feedback_20261107_codex as hold
from experiments import pc_tworoom_rgb_deadline_cem_20261107_codex as cem
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = hold.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_HOLD_CEM16_BASELINE_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_tworoom_hold_cem16_baseline_20261107_codex"
OUTPUT, POOL = Path(str(PREFIX) + ".json"), Path(str(PREFIX) + "_pool.npz")


def solve(boundary, ctx, scaler, old_tail, remaining, seed):
    if remaining not in (5, 10, 15, 20, 25):
        raise ValueError("unaligned deadline")
    rng = np.random.default_rng(seed)
    zero = scaler.transform(np.zeros((1, 2)))[0]
    low, high = (-1 - scaler.mean_) / scaler.scale_, (1 - scaler.mean_) / scaler.scale_
    mean, std = np.broadcast_to(zero, (5, 2)).copy(), np.ones((5, 2))
    old = np.asarray(scaler.transform(cem.old_or_hold(old_tail)), dtype=np.float32)[None]
    chains, costs, rounds = [old], [], []

    def score(normal):
        batch = torch.as_tensor(normal[:, :remaining].reshape(1, -1, remaining // 5, 10),
                                dtype=torch.float32, device=ctx["pixels"].device)
        result, _ = boundary.score(ctx, batch)
        values = result[0].cpu().numpy()
        if not np.isfinite(values).all() or len(values) != len(normal):
            raise ValueError("invalid scores")
        return values

    costs.append(score(old))
    for _ in range(3):
        sample = cem.candidates(rng, mean, std, low, high, 5)
        normal = np.repeat(sample, 5, axis=1).astype(np.float32)
        value = score(normal)
        rounds.append({"mean": mean.tolist(), "std": std.tolist()})
        elite = sample[np.argsort(value, kind="stable")[:2]]
        mean, std = elite.mean(0), np.maximum(elite.std(0), 1e-5)
        chains.append(normal)
        costs.append(value)
    chains, costs = np.concatenate(chains), np.concatenate(costs)
    if chains.shape != (16, 25, 2) or costs.shape != (16,):
        raise ValueError("query budget mismatch")
    selected = int(np.argmin(costs))
    actions = scaler.inverse_transform(chains[selected]).astype(np.float32)[:remaining]
    if not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6:
        raise ValueError("illegal selected action")
    return np.clip(actions, -1, 1), chains, costs, selected, rounds


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists() or POOL.exists():
        raise FileExistsError("output exists")
    old = json.loads(hold.OUT.read_text())
    for key, path in (("runner_sha256", hold.__file__), ("protocol_sha256", hold.PROTOCOL),
                      ("search_sha256", hold.search.__file__)):
        if old[key] != file_hash(path):
            raise ValueError("comparison source drift")
    source = hold.prior.source.block.prior
    if old["input_sha256"] != file_hash(source.inputs.INPUT):
        raise ValueError("input drift")
    model, scaler, transform, normalizer = source.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != old["model_sha256"] or normalizer != old["normalizer_sha256"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, hold.OUT, source.inputs.INPUT,
                         source.common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224 TwoRoom action scaler",
                         clock="five environment steps per packet; remaining deadline")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, arrays = [], {}
    try:
        with np.load(source.inputs.INPUT, allow_pickle=False) as inputs, \
                h5py.File(source.common.DATA, "r") as data, torch.inference_mode():
            for task in old["rows"]:
                root, row = task["root"], task["row"]
                if row != int(inputs["rows"][root]):
                    raise ValueError("task identity drift")
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][row])
                actor._set_goal_state(data["proprio"][row + 25])
                old_tail, decisions, trace = None, [], []
                start = time.perf_counter()
                for iteration in range(5):
                    context = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                               "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    boundary = RGBLeWM(model, audit)
                    remaining = 25 - len(trace)
                    seed = 8140000 + root * 10 + iteration
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    actions, pool, costs, selected, rounds = solve(boundary, context, scaler, old_tail, remaining, seed)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - before
                    key = f"r{root}_i{iteration}"
                    arrays[key + "_normal"], arrays[key + "_cost"] = pool, costs
                    dispatched = []
                    for action in actions[:5]:
                        _, _, term, trunc, info = actor.step(action.copy())
                        dispatched.append(action.tolist())
                        trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                        if term or trunc:
                            break
                    decisions.append({"seed": seed, "remaining": remaining, "pool_key": key, "chosen": selected,
                                      "actions": actions.tolist(), "dispatched": dispatched, "rounds": rounds,
                                      "planning_seconds": elapsed, "queries": 16,
                                      "packet_steps": 16 * (remaining // 5), "receipts": boundary.receipts})
                    old_tail = actions[len(dispatched):].copy()
                    if term or trunc:
                        break
                rows.append({"root": root, "row": row, "trace": trace, "decisions": decisions,
                             "ever_success": any(x["terminated"] for x in trace),
                             "task_seconds": time.perf_counter() - start})
                print(f"root={root} success={rows[-1]['ever_success']}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model mutated")
    with POOL.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with OUTPUT.open("x") as f:
        json.dump({"track": RGB, "scope": "exposed development only", "rows": rows,
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer,
                   "pool_sha256": file_hash(POOL), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "comparison_sha256": file_hash(hold.OUT)},
                  f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
