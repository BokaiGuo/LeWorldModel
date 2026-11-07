"""Paired RGB-only CEM16 variance-floor development experiment."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as base
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

PROTOCOL = base.ROOT.parent / "docs/PC_RGB_CEM16_VARIANCE_FLOOR_PROTOCOL_20261107_CODEX.md"
OUTPUT = base.ROOT / "results/pc_tworoom_cem16_variance_floor_20261107_codex.json"


def update_elites(samples, costs, floor):
    samples, costs = np.asarray(samples), np.asarray(costs)
    if (samples.shape != (5, 5, 2) or costs.shape != (5,)
            or not np.isfinite(samples).all() or not np.isfinite(costs).all()
            or not np.isfinite(floor) or floor <= 0):
        raise ValueError("invalid elite inputs")
    elite = samples[np.argsort(costs, kind="stable")[:2]]
    return elite.mean(0), np.maximum(elite.std(0), floor)


def solve(boundary, current, scaler, old_tail, remaining, seed, floor):
    if remaining not in (5, 10, 15, 20, 25):
        raise ValueError("invalid time")
    rng = np.random.default_rng(seed)
    zero = scaler.transform(np.zeros((1, 2)))[0]
    low, high = (-1 - scaler.mean_) / scaler.scale_, (1 - scaler.mean_) / scaler.scale_
    mean, std = np.broadcast_to(zero, (5, 2)).copy(), np.ones((5, 2))
    incumbent = np.asarray(scaler.transform(base.cem.old_or_hold(old_tail)), dtype=np.float32)[None]
    calls = 0

    def score(normal):
        nonlocal calls
        values = []
        for start in range(0, len(normal), 32):
            chunk = torch.as_tensor(normal[start:start + 32, :remaining].reshape(1, -1, remaining // 5, 10),
                                    dtype=torch.float32, device=current["pixels"].device)
            endpoint, context = boundary.score(current, chunk)
            pred, goal = context["predicted_emb"], context["goal_emb"][:, -1]
            if pred.shape[2] != remaining // 5 + 1:
                raise ValueError("time mismatch")
            costs = (pred[:, :, 1:] - goal[:, None, None]).square().sum(-1)
            if not torch.isfinite(costs).all() or not torch.allclose(endpoint, costs[:, :, -1], atol=1e-3, rtol=0):
                raise ValueError("invalid model score")
            values.extend(costs.min(-1).values[0].cpu().tolist())
            calls += 1
        return np.asarray(values)

    chains, costs = [incumbent], [score(incumbent)]
    rounds = []
    for _ in range(3):
        sample = base.cem.candidates(rng, mean, std, low, high, 5)
        normal = np.repeat(sample, 5, axis=1).astype(np.float32)
        value = score(normal)
        rounds.append({"mean": mean.tolist(), "std": std.tolist(), "costs": value.tolist()})
        mean, std = update_elites(sample, value, floor)
        chains.append(normal)
        costs.append(value)
    chains, costs = np.concatenate(chains), np.concatenate(costs)
    index = int(np.argmin(costs))
    actions = scaler.inverse_transform(chains[index]).astype(np.float32)[:remaining]
    if len(costs) != 16 or not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6:
        raise ValueError("invalid budget or selected action")
    return {"actions": np.clip(actions, -1, 1), "chosen": index, "chosen_cost": float(costs[index]),
            "rounds": rounds, "queries": 16, "packet_steps": 16 * remaining // 5, "forward_calls": calls}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old = json.loads(base.OUTPUT.read_text())
    if (old["runner_sha256"] != file_hash(base.__file__) or old["protocol_sha256"] != file_hash(base.PROTOCOL)
            or old["input_sha256"] != file_hash(base.INPUT)):
        raise ValueError("baseline provenance drift")
    model, scaler, transform, normalizer = base.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != old["model_sha256"] or normalizer != old["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, base.OUTPUT, base.INPUT,
                                      base.common.CHECKPOINT / "weights.pt", base.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 TwoRoom dataset action scaler",
                         clock="5-step packets and remaining 25-step deadline")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(base.INPUT, allow_pickle=False) as inputs, h5py.File(base.common.DATA, "r") as data,
              torch.inference_mode()):
            for i, original in enumerate(old["rows"]):
                episode, row = int(inputs["episodes"][i]), int(inputs["rows"][i])
                if (episode, row) != (original["episode"], original["row"]):
                    raise ValueError("task identity drift")
                goal = transform(inputs["goal"][i])[None, None, None].cuda()
                arms = {}
                for label, floor in (("baseline", 1e-5), ("floor", 0.2)):
                    started = time.perf_counter()
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row + 25])
                    tail, decisions, trace = None, [], []
                    for iteration in range(5):
                        current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                   "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        boundary = RGBLeWM(model, audit)
                        remaining = 25 - len(trace)
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        result = solve(boundary, current, scaler, tail, remaining,
                                       8140000 + episode * 10 + iteration, floor)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - before
                        actions, dispatched = result.pop("actions"), []
                        if label == "baseline":
                            expected = original["arms"]["cem16"]["decisions"][iteration]
                            np.testing.assert_allclose(actions, expected["actions"], rtol=0, atol=1e-6)
                            if result["chosen"] != expected["chosen"]:
                                raise ValueError("baseline choice mismatch")
                            np.testing.assert_allclose(result["chosen_cost"], expected["chosen_cost"], rtol=0, atol=1e-3)
                        for action in actions[:5]:
                            _, _, term, trunc, info = actor.step(action.copy())
                            dispatched.append(action.tolist())
                            trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                            if term or trunc:
                                break
                        decisions.append({**result, "remaining": remaining, "actions": actions.tolist(),
                                          "dispatched": dispatched, "planning_seconds": elapsed})
                        tail = actions[len(dispatched):].copy()
                        if term or trunc:
                            break
                    if label == "baseline":
                        expected_trace = original["arms"]["cem16"]["trace"]
                        if len(trace) != len(expected_trace):
                            raise ValueError("baseline trace length mismatch")
                        for actual, expected in zip(trace, expected_trace, strict=True):
                            if (actual["terminated"] != expected["terminated"] or actual["truncated"] != expected["truncated"]
                                    or abs(actual["distance"] - expected["distance"]) > 1e-5):
                                raise ValueError("baseline physical replay mismatch")
                    arms[label] = {"ever_success": any(t["terminated"] for t in trace), "trace": trace,
                                   "decisions": decisions, "wall_seconds": time.perf_counter() - started}
                rows.append({"episode": episode, "row": row, "initial_success": original["initial_success"], "arms": arms})
                print(f"episode={episode} baseline={arms['baseline']['ever_success']} floor={arms['floor']['ever_success']}",
                      flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with OUTPUT.open("x") as stream:
        json.dump({"scope": "exposed RGB-only development baseline tuning; not gate evidence",
                   "track": RGB, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "baseline_sha256": file_hash(base.OUTPUT), "input_sha256": file_hash(base.INPUT),
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer, "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)
    print(f"sealed {len(rows)} tasks", flush=True)


if __name__ == "__main__":
    main()
