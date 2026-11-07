"""Four-arm RGB feedback comparison of final-time selection rules."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_rgb_hold_retention_feedback_20261107_codex as prior
from experiments import pc_rgb_temporal_final_search_20261107_codex as search
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_TEMPORAL_FINAL_SELECTION_FEEDBACK_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_tworoom_temporal_final_feedback_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    original = json.loads(prior.OUT.read_text())
    model, scaler, transform, normalizer = prior.prior.source.block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != original["model_sha256"] or normalizer != original["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    inputs_path = prior.prior.source.block.prior.inputs.INPUT
    common = prior.prior.source.block.prior.common
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, Path(search.__file__), inputs_path,
                                      common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="remaining deadline, five environment steps per packet")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Run paired RGB-only feedback; temporal rule changes final selection only.", flush=True)
    try:
        with np.load(inputs_path, allow_pickle=False) as inputs, h5py.File(common.DATA, "r") as data:
            with torch.inference_mode():
                for old in original["rows"]:
                    root, task = old["root"], old["row"]
                    if int(inputs["rows"][root]) != task:
                        raise ValueError("task binding mismatch")
                    goal = transform(inputs["goal"][root])[None, None, None].cuda()
                    arms, initial = {}, None
                    for strategy in ("lookahead", "effect"):
                        for rule in ("endpoint", "temporal"):
                            arm = strategy + "_" + rule
                            start = time.perf_counter()
                            env.reset(seed=42)
                            actor = env.unwrapped
                            actor._set_state(data["proprio"][task])
                            actor._set_goal_state(data["proprio"][task + 25])
                            initial_success = bool(float((actor.agent_position - actor.target_position).norm()) < 16.)
                            if initial is not None and initial_success != initial:
                                raise ValueError("initial state mismatch")
                            initial = initial_success
                            old_tail, decisions, trace = None, [], []
                            for iteration in range(5):
                                ctx = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                       "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                                boundary = RGBLeWM(model, audit)
                                torch.cuda.synchronize()
                                before = time.perf_counter()
                                result = search.compile_rgb(
                                    boundary, ctx, scaler=scaler, old_tail=old_tail,
                                    candidate_seed=27121000 + root * 10 + iteration,
                                    strategy_seed=27122000 + root * 10 + iteration,
                                    strategy=strategy, final_rule=rule, remaining_steps=25 - len(trace))
                                torch.cuda.synchronize()
                                elapsed = time.perf_counter() - before
                                chosen, dispatched = result["actions"], []
                                for action in chosen[:5]:
                                    _, _, term, trunc, info = actor.step(action.copy())
                                    dispatched.append(action.tolist())
                                    trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                                  "terminated": bool(term), "truncated": bool(trunc)})
                                    if term or trunc:
                                        break
                                decisions.append({"iteration": iteration, "retained": result["retained_prefixes"],
                                                  "choice": {"source": result["choice"].source,
                                                             "index": result["choice"].index,
                                                             "cost": result["choice"].cost},
                                                  "parent_cost": result["parent_cost"].tolist(),
                                                  "probe_cost": result["probe_cost"].tolist(),
                                                  "tail_cost": result["tail_cost"].tolist(),
                                                  "temporal_scores": result["temporal_scores"],
                                                  "actions": chosen.tolist(), "dispatched": dispatched,
                                                  "actions_sha256": prior.prior.action_hash(chosen),
                                                  "old_tail_sha256": None if old_tail is None else prior.prior.action_hash(old_tail),
                                                  "remaining_steps": result["remaining_steps"],
                                                  "model_queries": result["model_queries"],
                                                  "model_packet_steps": result["model_packet_steps"],
                                                  "forward_calls": result["forward_calls"],
                                                  "planning_seconds": elapsed, "receipts": boundary.receipts})
                                old_tail = chosen[len(dispatched):].copy()
                                if term or trunc:
                                    break
                            arms[arm] = {"decisions": decisions, "trace": trace,
                                         "ever_success": any(x["terminated"] for x in trace),
                                         "wall_seconds": time.perf_counter() - start}
                    rows.append({"root": root, "row": task, "initial_success": initial, "arms": arms})
                    print(f"root={root} success={[v['ever_success'] for v in arms.values()]}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weights changed")
    with OUT.open("x") as stream:
        json.dump({"track": RGB, "scope": "exposed development, actual RGB feedback",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "search_sha256": file_hash(Path(search.__file__)), "source_sha256": file_hash(prior.OUT),
                   "input_sha256": file_hash(inputs_path), "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
