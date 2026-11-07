"""Preselected TwoRoom episodes, reference replay, and temporal-score planners."""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_rgb_action_effect_alignment_20261107_codex as common
from experiments import pc_tworoom_reference_vs_hold_20261107_codex as prior
from experiments import pc_rgb_temporal_final_search_20261107_codex as program
from experiments import pc_tworoom_rgb_deadline_cem_20261107_codex as cem
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = common.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_TEMPORAL_NEW_EPISODES_PROTOCOL_20261107_CODEX.md"
INPUT = ROOT / "results/pc_tworoom_temporal_new_episodes_input_20261107_codex.npz"
REFERENCE = ROOT / "results/pc_tworoom_temporal_new_episodes_reference_20261107_codex.json"
OUTPUT = ROOT / "results/pc_tworoom_temporal_new_episodes_feedback_20261107_codex.json"
ARMS = ("lookahead", "effect", "cem16", "cem300")


def prepare():
    if INPUT.exists():
        raise FileExistsError(INPUT)
    with h5py.File(common.DATA, "r") as data, np.load(prior.INPUT, allow_pickle=False) as old:
        episodes = np.arange(9000, 9040)
        existing = set(np.asarray(data["ep_idx"])[old["rows"]].tolist())
        if existing.intersection(episodes.tolist()):
            raise ValueError("overlap with prior 50 root panel")
        rows = np.asarray(data["ep_offset"])[episodes]
        lengths = np.asarray(data["ep_len"])[episodes]
        if np.any(lengths <= 25):
            raise ValueError("episode shorter than target horizon")
        current, goal, actions = [], [], []
        for ep, row in zip(episodes, rows, strict=True):
            row = int(row)
            if int(data["ep_idx"][row]) != ep or int(data["ep_idx"][row + 25]) != ep or int(data["step_idx"][row]) != 0:
                raise ValueError("episode/time binding mismatch")
            current.append(data["pixels"][row])
            goal.append(data["pixels"][row + 25])
            actions.append(data["action"][row:row + 25])
        actions = np.asarray(actions, np.float32)
        if actions.shape != (40, 25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
            raise ValueError("reference actions invalid")
    with INPUT.open("xb") as stream:
        np.savez_compressed(stream, episodes=episodes, rows=rows, current=np.asarray(current),
                            goal=np.asarray(goal), actions=actions)
    print("Sealed 40 predetermined episode rows and reference actions.", flush=True)


def reference():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if REFERENCE.exists():
        raise FileExistsError(REFERENCE)
    rows = []
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    try:
        with np.load(INPUT, allow_pickle=False) as inputs, h5py.File(common.DATA, "r") as data:
            for episode, row, actions in zip(inputs["episodes"], inputs["rows"], inputs["actions"], strict=True):
                row, episode = int(row), int(episode)
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][row])
                actor._set_goal_state(data["proprio"][row + 25])
                initial = bool(float((actor.agent_position - actor.target_position).norm()) < 16.)
                trace = []
                for action in actions:
                    _, _, term, trunc, info = actor.step(action.copy())
                    if trunc:
                        raise ValueError("unexpected reference truncation")
                    trace.append({"distance": float(info["distance_to_target"]), "success": bool(term)})
                rows.append({"episode": episode, "row": row, "initial_success": initial, "trace": trace})
    finally:
        env.close()
    with REFERENCE.open("x") as stream:
        json.dump({"scope": "reference physical replay; no filtering",
                   "input_sha256": file_hash(INPUT), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)), "rows": rows}, stream, indent=2)
    print(f"reference episodes={len(rows)} ever_success={sum(any(t['success'] for t in r['trace']) for r in rows)}")


def cem_temporal(boundary, current, scaler, old_tail, remaining, seed, budget):
    if budget not in (16, 300) or remaining not in (5, 10, 15, 20, 25):
        raise ValueError("invalid budget or time")
    rng = np.random.default_rng(seed)
    zero = scaler.transform(np.zeros((1, 2)))[0]
    low, high = (-1 - scaler.mean_) / scaler.scale_, (1 - scaler.mean_) / scaler.scale_
    mean, std = np.broadcast_to(zero, (5, 2)).copy(), np.ones((5, 2))
    old = cem.old_or_hold(old_tail)
    incumbent = np.asarray(scaler.transform(old), dtype=np.float32)[None]
    calls = 0
    ledger = []

    def score(normal):
        nonlocal calls
        values = []
        for start in range(0, len(normal), 32):
            chunk = torch.as_tensor(normal[start:start + 32, :remaining].reshape(1, -1, remaining // 5, 10),
                                    dtype=torch.float32, device=current["pixels"].device)
            endpoint, context = boundary.score(current, chunk)
            pred, goal = context["predicted_emb"], context["goal_emb"][:, -1]
            if pred.shape[2] != remaining // 5 + 1:
                raise ValueError("temporal packet mismatch")
            costs = (pred[:, :, 1:] - goal[:, None, None]).square().sum(-1)
            if not torch.isfinite(costs).all() or not torch.allclose(endpoint, costs[:, :, -1], atol=1e-3, rtol=0):
                raise ValueError("invalid temporal cost")
            values.extend(costs.min(-1).values[0].cpu().tolist())
            calls += 1
        return np.asarray(values)

    all_chains, all_costs = [incumbent], [score(incumbent)]
    for count in ((5, 5, 5) if budget == 16 else (100, 100, 99)):
        samples = cem.candidates(rng, mean, std, low, high, count)
        normal = np.repeat(samples, 5, axis=1).astype(np.float32)
        costs = score(normal)
        elite = samples[np.argsort(costs, kind="stable")[:2 if budget == 16 else 10]]
        ledger.append({"count": count, "mean": mean.tolist(), "std": std.tolist(), "costs": costs.tolist()})
        mean, std = elite.mean(0), np.maximum(elite.std(0), 1e-5)
        all_chains.append(normal)
        all_costs.append(costs)
    chains, costs = np.concatenate(all_chains), np.concatenate(all_costs)
    if len(costs) != budget:
        raise ValueError("CEM budget mismatch")
    index = int(np.argmin(costs))
    actions = scaler.inverse_transform(chains[index]).astype(np.float32)[:remaining]
    if not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6:
        raise ValueError("CEM illegal actions")
    return {"actions": np.clip(actions, -1, 1), "chosen": index, "chosen_cost": float(costs[index]),
            "incumbent_cost": float(costs[0]), "rounds": ledger, "queries": budget,
            "packet_steps": budget * (remaining // 5), "forward_calls": calls}


def feedback():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    reference_data = json.loads(REFERENCE.read_text())
    if (reference_data["input_sha256"] != file_hash(INPUT)
            or reference_data["protocol_sha256"] != file_hash(PROTOCOL)
            or reference_data["runner_sha256"] != file_hash(Path(__file__))):
        raise ValueError("reference provenance drift")
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, INPUT, REFERENCE, Path(program.__file__),
                                      common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 TwoRoom dataset action scaler",
                         clock="5-step packets and remaining 25-step deadline")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("RGB-only new-episode temporal program vs low/high-budget temporal CEM.", flush=True)
    try:
        with np.load(INPUT, allow_pickle=False) as inputs, h5py.File(common.DATA, "r") as data:
            with torch.inference_mode():
                for index, (episode, row) in enumerate(zip(inputs["episodes"], inputs["rows"], strict=True)):
                    episode, row = int(episode), int(row)
                    if reference_data["rows"][index]["episode"] != episode:
                        raise ValueError("reference task mismatch")
                    goal = transform(inputs["goal"][index])[None, None, None].cuda()
                    arms = {}
                    for arm in ARMS:
                        started = time.perf_counter()
                        env.reset(seed=42)
                        actor = env.unwrapped
                        actor._set_state(data["proprio"][row])
                        actor._set_goal_state(data["proprio"][row + 25])
                        old_tail, decisions, trace = None, [], []
                        for iteration in range(5):
                            current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                       "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                            boundary = RGBLeWM(model, audit)
                            remaining = 25 - len(trace)
                            torch.cuda.synchronize()
                            before = time.perf_counter()
                            if arm in ("lookahead", "effect"):
                                result = program.compile_rgb(
                                    boundary, current, scaler=scaler, old_tail=old_tail,
                                    candidate_seed=27121000 + episode * 10 + iteration,
                                    strategy_seed=27122000 + episode * 10 + iteration,
                                    strategy=arm, remaining_steps=remaining, final_rule="temporal")
                                extra = {"retained": result["retained_prefixes"],
                                         "choice": {"source": result["choice"].source, "index": result["choice"].index,
                                                    "cost": result["choice"].cost},
                                         "temporal_scores": result["temporal_scores"],
                                         "parent_cost": result["parent_cost"].tolist(),
                                         "probe_cost": result["probe_cost"].tolist(),
                                         "tail_cost": result["tail_cost"].tolist(),
                                         "candidate_seed": 27121000 + episode * 10 + iteration,
                                         "strategy_seed": 27122000 + episode * 10 + iteration}
                                queries, packets = result["model_queries"], result["model_packet_steps"]
                            else:
                                result = cem_temporal(boundary, current, scaler, old_tail, remaining,
                                                      8140000 + episode * 10 + iteration,
                                                      16 if arm == "cem16" else 300)
                                extra = {k: result[k] for k in ("chosen", "chosen_cost", "incumbent_cost", "rounds")}
                                extra["candidate_seed"] = 8140000 + episode * 10 + iteration
                                queries, packets = result["queries"], result["packet_steps"]
                            torch.cuda.synchronize()
                            elapsed = time.perf_counter() - before
                            actions, dispatched = result["actions"], []
                            for action in actions[:5]:
                                _, _, term, trunc, info = actor.step(action.copy())
                                dispatched.append(action.tolist())
                                trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                              "terminated": bool(term), "truncated": bool(trunc)})
                                if term or trunc:
                                    break
                            decisions.append({"iteration": iteration, "remaining": remaining,
                                              "actions": actions.tolist(), "dispatched": dispatched,
                                              "queries": queries, "packet_steps": packets,
                                              "forward_calls": result["forward_calls"], "planning_seconds": elapsed,
                                              "receipts": boundary.receipts, **extra})
                            old_tail = actions[len(dispatched):].copy()
                            if term or trunc:
                                break
                        arms[arm] = {"ever_success": any(t["terminated"] for t in trace), "trace": trace,
                                     "wall_seconds": time.perf_counter() - started, "decisions": decisions}
                    rows.append({"episode": episode, "row": row,
                                 "initial_success": reference_data["rows"][index]["initial_success"], "arms": arms})
                    print(f"episode={episode} successes={[arms[a]['ever_success'] for a in ARMS]}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with OUTPUT.open("x") as stream:
        json.dump({"scope": "newly selected episodes; training membership and earlier exposure unknown",
                   "track": RGB, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "input_sha256": file_hash(INPUT), "reference_sha256": file_hash(REFERENCE),
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer, "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "reference", "feedback"):
        raise SystemExit("expected prepare|reference|feedback")
    {"prepare": prepare, "reference": reference, "feedback": feedback}[sys.argv[1]]()
