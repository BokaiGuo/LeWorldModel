"""Budgeted RGB-only feedback block-CEM baselines on exposed TwoRoom tasks."""
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_shared_rgb_feedback_retention_20261107_codex as prior
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_RGB_DEADLINE_CEM_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_tworoom_rgb_deadline_cem_20261107_codex"
POOL = Path(str(BASE) + "_pool.npz")
OUT = Path(str(BASE) + "_run.json")
ARMS = ("cem13", "cem300")


def candidates(rng, mean, std, low, high, count):
    mean, std = np.asarray(mean), np.asarray(std)
    if mean.shape != (5, 2) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("invalid CEM distribution")
    out = np.clip(mean + rng.normal(size=(count, 5, 2)) * std, low, high)
    if not np.isfinite(out).all():
        raise ValueError("invalid candidates")
    return out


def old_or_hold(old_tail):
    if old_tail is None:
        return np.zeros((25, 2), dtype=np.float32)
    old_tail = np.asarray(old_tail, dtype=np.float32)
    if old_tail.ndim != 2 or old_tail.shape[1] != 2 or len(old_tail) > 25 or not np.isfinite(old_tail).all() or np.abs(old_tail).max() > 1:
        raise ValueError("invalid old tail")
    return np.pad(old_tail, ((0, 25 - len(old_tail)), (0, 0)))


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists() or POOL.exists():
        raise FileExistsError("sealed result exists")
    earlier = json.loads(prior.OUT.read_text())
    if earlier["runner_sha256"] != file_hash(prior.__file__) or earlier["protocol_sha256"] != file_hash(prior.PROTOCOL):
        raise ValueError("baseline comparison source drift")
    old_plan = json.loads(prior.prior.source.PLAN.read_text())
    model, scaler, transform, normalizer = prior.prior.source.block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != old_plan["model_sha256"] or normalizer != old_plan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, prior.OUT, prior.prior.source.block.prior.inputs.INPUT,
                                      prior.prior.source.block.prior.common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="single observed frame; five packets of five environment steps")
    low, high = (-1-scaler.mean_) / scaler.scale_, (1-scaler.mean_) / scaler.scale_
    zero = scaler.transform(np.zeros((1, 2)))[0]
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, arrays = [], {}
    try:
        with (np.load(prior.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(prior.prior.source.block.prior.common.DATA, "r") as data, torch.inference_mode()):
            for root in range(10, 50):
                row = int(inputs["rows"][root])
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                outcomes = {}
                for arm in ARMS:
                    task_started = time.perf_counter()
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row + 25])
                    old_tail, decisions, trace = None, [], []
                    for iteration in range(5):
                        image = actor.render().copy()
                        ctx = {"pixels": transform(image)[None, None, None].cuda(), "goal": goal,
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        boundary = RGBLeWM(model, audit)
                        rng = np.random.default_rng(8140000 + root * 10 + iteration)
                        mean = np.broadcast_to(zero, (5, 2)).copy()
                        std = np.ones((5, 2))
                        old = old_or_hold(old_tail)
                        old_normal = np.asarray(scaler.transform(old), dtype=np.float32)
                        collected, values, rounds = [], [], []
                        remaining = 25 - len(trace)
                        if remaining not in (5, 10, 15, 20, 25):
                            raise ValueError("deadline not aligned to action packets")

                        def score(normal):
                            costs = []
                            for start in range(0, len(normal), 32):
                                chunk = torch.as_tensor(normal[start:start+32, :remaining].reshape(1, -1, remaining // 5, 10),
                                                        dtype=torch.float32, device="cuda")
                                cost, _ = boundary.score(ctx, chunk)
                                costs.extend(cost[0].cpu().numpy().tolist())
                            costs = np.asarray(costs)
                            if len(costs) != len(normal) or not np.isfinite(costs).all():
                                raise ValueError("invalid CEM scores")
                            return costs

                        torch.cuda.synchronize()
                        planning_start = time.perf_counter()
                        if arm == "cem13":
                            collected.append(old_normal[None])
                            values.append(score(old_normal[None]))
                        count, elite_count = (4, 2) if arm == "cem13" else (100, 10)
                        for round_index in range(3):
                            sample = candidates(rng, mean, std, low, high, count)
                            if arm == "cem300" and round_index == 0:
                                sample[0] = old_normal.reshape(5, 5, 2)[:, 0]
                                np.testing.assert_allclose(np.repeat(sample[0], 5, axis=0), old_normal, atol=1e-6, rtol=0)
                            normal = np.repeat(sample, 5, axis=1).astype(np.float32)
                            cost = score(normal)
                            elite = sample[np.argsort(cost, kind="stable")[:elite_count]]
                            rounds.append({"mean": mean.tolist(), "std": std.tolist()})
                            mean, std = elite.mean(0), np.maximum(elite.std(0), 1e-5)
                            collected.append(normal)
                            values.append(cost)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter()-planning_start
                        chain_norm, costs = np.concatenate(collected), np.concatenate(values)
                        expected = 13 if arm == "cem13" else 300
                        if chain_norm.shape != (expected, 25, 2) or costs.shape != (expected,):
                            raise ValueError("budget mismatch")
                        selected = int(np.argmin(costs))
                        raw = scaler.inverse_transform(chain_norm[selected]).astype(np.float32)
                        if not np.isfinite(raw).all() or np.abs(raw).max() > 1 + 1e-6:
                            raise ValueError("illegal action")
                        raw = np.clip(raw, -1, 1)
                        key = f"r{root}_{arm}_i{iteration}"
                        arrays[key + "_normal"] = chain_norm
                        arrays[key + "_costs"] = costs
                        dispatched = []
                        for action in raw[:5]:
                            _, _, term, trunc, info = actor.step(action.copy())
                            dispatched.append(action.tolist())
                            trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                            if term or trunc:
                                break
                        decisions.append({"iteration": iteration, "seed": 8140000 + root * 10 + iteration,
                                          "old_tail_sha256": None if old_tail is None else prior.prior.action_hash(old_tail),
                                          "pool_key": key, "chosen": selected, "chosen_cost": float(costs[selected]),
                                          "actions": raw.tolist(), "dispatched": dispatched, "queries": expected,
                                          "remaining_steps": remaining, "model_packet_steps": expected * (remaining // 5),
                                          "rounds": rounds, "planning_seconds": elapsed, "receipts": boundary.receipts})
                        old_tail = raw[len(dispatched):].copy()
                        if term or trunc:
                            break
                    outcomes[arm] = {"decisions": decisions, "trace": trace,
                                     "ever_success": any(x["terminated"] for x in trace),
                                     "truncated": any(x["truncated"] for x in trace),
                                     "terminal_distance": trace[-1]["distance"],
                                     "queries": sum(x["queries"] for x in decisions),
                                     "wall_before_serialization_seconds": time.perf_counter() - task_started}
                rows.append({"root": root, "row": row, "arms": outcomes})
                print(f"root={root} success={[outcomes[a]['ever_success'] for a in ARMS]}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with POOL.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with OUT.open("x") as stream:
        json.dump({"track": RGB, "protocol": file_hash(PROTOCOL), "runner": file_hash(__file__),
                   "source": file_hash(prior.OUT), "pool": file_hash(POOL), "model": fingerprint,
                   "audit_hash": audit.fingerprint, "rows": rows,
                   "scope": "exposed RGB-only development; budgeted CEM not independently tuned"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
