"""Paired actual RGB closed-loop episodes with and without exact-input encoding cache."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as prior
from experiments import pc_rgb_temporal_final_search_20261107_codex as program
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_shared_encoder_cache_20261107_codex import CachedJEPA

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_CACHED_CLOSED_LOOP_COST_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_rgb_cached_closed_loop_cost_20261107_codex.json"


def compare(a, b):
    """Report mismatches without suppressing the underlying paired trajectories."""
    issues = []
    if len(a["decisions"]) != len(b["decisions"]):
        issues.append("decision_count")
    if len(a["trace"]) != len(b["trace"]):
        issues.append("trace_length")
    max_action_error = 0.0
    max_distance_error = 0.0
    for da, db in zip(a["decisions"], b["decisions"]):
        if (da["choice"] != db["choice"] or da["queries"] != db["queries"]
                or da["packet_steps"] != db["packet_steps"] or da["remaining"] != db["remaining"]):
            issues.append("selection_clock_or_budget")
        aa, ab = np.asarray(da["actions"]), np.asarray(db["actions"])
        if aa.shape != ab.shape:
            issues.append("action_shape")
        else:
            max_action_error = max(max_action_error, float(np.max(np.abs(aa - ab), initial=0)))
    for ta, tb in zip(a["trace"], b["trace"]):
        max_distance_error = max(max_distance_error, abs(ta["distance"] - tb["distance"]))
        if ta["terminated"] != tb["terminated"] or ta["truncated"] != tb["truncated"]:
            issues.append("termination")
    if max_action_error > 1e-6:
        issues.append("action_values")
    if max_distance_error > 1e-5:
        issues.append("physical_distance")
    return {"same": not issues, "issues": sorted(set(issues)), "max_action_error": max_action_error,
            "max_distance_error": max_distance_error}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    historical = json.loads(prior.OUTPUT.read_text())
    if (historical["runner_sha256"] != file_hash(prior.__file__)
            or historical["protocol_sha256"] != file_hash(prior.PROTOCOL)
            or historical["input_sha256"] != file_hash(prior.INPUT)):
        raise ValueError("source provenance drift")
    model, scaler, transform, normalizer = prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != historical["model_sha256"] or normalizer != historical["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, prior.INPUT,
                         prior.common.CHECKPOINT / "weights.pt", prior.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and fixed TwoRoom action scaler",
                         clock="five environment steps per packet")
    cache = CachedJEPA(model)
    modules = {"original": model, "cached": cache}
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Run paired closed-loop effect and CEM16, identical seeds with and without encoder caching.", flush=True)
    try:
        with np.load(prior.INPUT, allow_pickle=False) as inputs, h5py.File(prior.common.DATA, "r") as data, torch.inference_mode():
            for index, (episode, row) in enumerate(zip(inputs["episodes"], inputs["rows"], strict=True)):
                episode, row = int(episode), int(row)
                if episode != 9000 + index:
                    raise ValueError("episode panel drift")
                goal = transform(inputs["goal"][index])[None, None, None].cuda()
                outcomes = {}
                for method in ("effect", "cem16"):
                    outcomes[method] = {}
                    order = ("original", "cached") if (index + (method == "cem16")) % 2 == 0 else ("cached", "original")
                    for version in order:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        env.reset(seed=42)
                        actor = env.unwrapped
                        actor._set_state(data["proprio"][row])
                        actor._set_goal_state(data["proprio"][row + 25])
                        boundary = RGBLeWM(modules[version], audit)
                        before_encoding = cache.encoder_calls
                        tail, trace, decisions = None, [], []
                        for iteration in range(5):
                            current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                       "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                            remaining = 25 - len(trace)
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            if method == "effect":
                                result = program.compile_rgb(boundary, current, scaler=scaler, old_tail=tail,
                                    candidate_seed=27121000 + episode * 10 + iteration,
                                    strategy_seed=27122000 + episode * 10 + iteration,
                                    strategy="effect", remaining_steps=remaining, final_rule="temporal")
                                choice = {"source": result["choice"].source, "index": result["choice"].index}
                                queries, packets = result["model_queries"], result["model_packet_steps"]
                            else:
                                result = prior.cem_temporal(boundary, current, scaler, tail, remaining,
                                                            8140000 + episode * 10 + iteration, 16)
                                choice = {"index": result["chosen"]}
                                queries, packets = result["queries"], result["packet_steps"]
                            torch.cuda.synchronize()
                            planning_seconds = time.perf_counter() - start
                            actions = result["actions"]
                            dispatched = []
                            for action in actions[:5]:
                                _, _, term, trunc, info = actor.step(action.copy())
                                dispatched.append(action.tolist())
                                trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                              "terminated": bool(term), "truncated": bool(trunc)})
                                if term or trunc:
                                    break
                            decisions.append({"remaining": remaining, "choice": choice, "queries": queries,
                                              "packet_steps": packets, "actions": actions.tolist(),
                                              "dispatched": dispatched, "planning_seconds": planning_seconds})
                            tail = actions[len(dispatched):].copy()
                            if term or trunc:
                                break
                        torch.cuda.synchronize()
                        outcomes[method][version] = {"trace": trace, "decisions": decisions,
                            "success": any(step["terminated"] for step in trace),
                            "task_seconds": time.perf_counter() - started,
                            "cache_encoding_calls": cache.encoder_calls - before_encoding if version == "cached" else None}
                    outcomes[method]["comparison"] = compare(outcomes[method]["original"], outcomes[method]["cached"])
                rows.append({"episode": episode, "row": row, "arms": outcomes})
                print(f"episode={episode} effect_same={outcomes['effect']['comparison']['same']} "
                      f"cem_same={outcomes['cem16']['comparison']['same']}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with OUT.open("x") as stream:
        json.dump({"scope": "exposed tasks, paired full feedback, not independent confirmation",
                   "track": RGB, "device": torch.cuda.get_device_name(), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "source_sha256": file_hash(prior.OUTPUT),
                   "input_sha256": file_hash(prior.INPUT), "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"sealed {len(rows)} episodes", flush=True)


if __name__ == "__main__":
    main()
