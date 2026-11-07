"""Checkpoint-bound actual RGB feedback replay on four exposed episodes."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as source
from experiments.pc_checkpoint_bound_rgb_20261107_codex import CheckpointBoundRGB
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.research_runtime.artifacts import atomic_write_json

PROTOCOL = source.ROOT.parent / "docs/PC_BOUND_RGB_CLOSED_LOOP_REPLAY_PROTOCOL_20261107_CODEX.md"
OUT = source.ROOT / "results/pc_bound_rgb_closed_loop_replay_20261107_codex.json"
ARMS = ("lookahead", "effect", "cem16")


def compare(arm, old, new):
    if len(old["decisions"]) != len(new["decisions"]) or len(old["trace"]) != len(new["trace"]):
        raise ValueError("replay length mismatch")
    max_action, max_distance = 0., 0.
    for a, b in zip(old["decisions"], new["decisions"], strict=True):
        for key in ("iteration", "remaining", "queries", "packet_steps", "forward_calls"):
            if a[key] != b[key]:
                raise ValueError(f"decision mismatch: {key}")
        if arm in ("effect", "lookahead"):
            if a["retained"] != b["retained"] or (a["choice"]["source"], a["choice"]["index"]) != (
                    b["choice"]["source"], b["choice"]["index"]):
                raise ValueError("program choice mismatch")
        elif a["chosen"] != b["chosen"]:
            raise ValueError("CEM choice mismatch")
        for key in ("actions", "dispatched"):
            x, y = np.asarray(a[key]), np.asarray(b[key])
            if x.shape != y.shape:
                raise ValueError(f"{key} shape mismatch")
            if x.size:
                max_action = max(max_action, float(np.max(np.abs(x-y))))
    for a, b in zip(old["trace"], new["trace"], strict=True):
        if a["terminated"] != b["terminated"] or a["truncated"] != b["truncated"]:
            raise ValueError("terminal mismatch")
        max_distance = max(max_distance, abs(a["distance"] - b["distance"]))
    if old["ever_success"] != new["ever_success"] or max_action > 1e-6 or max_distance > 1e-5:
        raise ValueError("actual action, physical distance, or success mismatch")
    return {"max_action_abs_diff": max_action, "max_distance_abs_diff": max_distance}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    old = json.loads(source.OUTPUT.read_text())
    reference = json.loads(source.REFERENCE.read_text())
    for key, path in (("runner_sha256", source.__file__), ("protocol_sha256", source.PROTOCOL),
                      ("input_sha256", source.INPUT), ("reference_sha256", source.REFERENCE)):
        if old[key] != file_hash(path):
            raise ValueError(f"source provenance mismatch: {key}")
    model, scaler, transform, normalizer = source.common.model_setup()
    model_hash = model_fingerprint(model)
    if model_hash != old["model_sha256"] or normalizer != old["normalizer_sha256"]:
        raise ValueError("model or normalizer mismatch")
    weights, config = source.common.CHECKPOINT / "weights.pt", source.common.CHECKPOINT / "config.json"
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.OUTPUT, source.INPUT,
                         source.REFERENCE, source.program.__file__, weights, config],
                         preprocessing="ImageNet 224, TwoRoom dataset scaler",
                         clock="5 environment steps per packet, 25 step deadline")
    torch.cuda.synchronize()
    start = time.perf_counter()
    boundary = CheckpointBoundRGB(model, audit, weights, config)
    torch.cuda.synchronize()
    initial_check_seconds = time.perf_counter() - start
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Replay 4 exposed episodes with a checkpoint-bound RGB planner.", flush=True)
    try:
        with (np.load(source.INPUT, allow_pickle=False) as inputs,
              h5py.File(source.common.DATA, "r") as data, torch.inference_mode()):
            for index in range(4):
                episode, row = int(inputs["episodes"][index]), int(inputs["rows"][index])
                if (episode != 9000 + index or old["rows"][index]["episode"] != episode
                        or reference["rows"][index]["episode"] != episode):
                    raise ValueError("episode binding mismatch")
                goal = transform(inputs["goal"][index])[None, None, None].cuda()
                arms = {}
                for arm in ARMS:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row + 25])
                    old_tail, decisions, trace = None, [], []
                    for iteration in range(5):
                        remaining = 25 - len(trace)
                        current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                   "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        receipt_start = len(boundary.receipts)
                        if arm in ("effect", "lookahead"):
                            result = source.program.compile_rgb(
                                boundary, current, scaler=scaler, old_tail=old_tail,
                                candidate_seed=27121000 + episode * 10 + iteration,
                                strategy_seed=27122000 + episode * 10 + iteration,
                                strategy=arm, remaining_steps=remaining, final_rule="temporal")
                            details = {"retained": result["retained_prefixes"],
                                       "choice": {"source": result["choice"].source,
                                                  "index": result["choice"].index,
                                                  "cost": result["choice"].cost}}
                            queries, packets = result["model_queries"], result["model_packet_steps"]
                        else:
                            result = source.cem_temporal(boundary, current, scaler, old_tail, remaining,
                                                         8140000 + episode * 10 + iteration, 16)
                            details = {"chosen": result["chosen"], "chosen_cost": result["chosen_cost"]}
                            queries, packets = result["queries"], result["packet_steps"]
                        actions, dispatched = result["actions"], []
                        for action in actions[:5]:
                            _, _, term, trunc, info = actor.step(action.copy())
                            dispatched.append(action.tolist())
                            trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                            if term or trunc:
                                break
                        decisions.append({"iteration": iteration, "remaining": remaining, "actions": actions.tolist(),
                                          "dispatched": dispatched, "queries": queries, "packet_steps": packets,
                                          "forward_calls": result["forward_calls"],
                                          "receipts": boundary.receipts[receipt_start:], **details})
                        old_tail = actions[len(dispatched):].copy()
                        if term or trunc:
                            break
                    actual = {"ever_success": any(t["terminated"] for t in trace), "trace": trace,
                              "decisions": decisions}
                    actual["parity"] = compare(arm, old["rows"][index]["arms"][arm], actual)
                    arms[arm] = actual
                rows.append({"episode": episode, "row": row, "arms": arms})
                print(f"episode={episode} all arms replayed", flush=True)
    finally:
        env.close()
    torch.cuda.synchronize()
    start = time.perf_counter()
    binding = boundary.finish()
    torch.cuda.synchronize()
    final_check_seconds = time.perf_counter() - start
    if model_fingerprint(model) != model_hash:
        raise ValueError("model drift")
    record = {"scope": "exposed RGB integration replay, not independent confirmation",
              "track": RGB, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
              "source_sha256": file_hash(source.OUTPUT), "model_sha256": model_hash,
              "normalizer_sha256": normalizer, "binding": binding,
              "initial_binding_seconds": initial_check_seconds, "final_binding_seconds": final_check_seconds,
              "rows": rows}
    atomic_write_json(OUT, record)
    print(json.dumps({"episodes": len(rows), "arms": len(rows) * len(ARMS),
                      "successful": sum(r["arms"][a]["ever_success"] for r in rows for a in ARMS),
                      "queries": sum(d["queries"] for r in rows for a in ARMS for d in r["arms"][a]["decisions"]),
                      "initial_binding_seconds": initial_check_seconds,
                      "final_binding_seconds": final_check_seconds}, indent=2))


if __name__ == "__main__":
    main()
