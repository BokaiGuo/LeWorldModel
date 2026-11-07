"""Exposed development panel: zero-relative retention with actual RGB feedback."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_shared_rgb_feedback_smoke_20261107_codex as prior
from experiments import pc_rgb_hold_retention_search_20261107_codex as search
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_HOLD_RETENTION_FEEDBACK_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_tworoom_rgb_hold_retention_feedback_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(prior.source.PLAN.read_text())
    if plan["pool_sha256"] != file_hash(prior.source.POOL):
        raise ValueError("source pool drift")
    model, scaler, transform, normalizer = prior.source.block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    inputs_path = prior.source.block.prior.inputs.INPUT
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, Path(search.__file__), inputs_path,
                                      prior.source.block.prior.common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="remaining deadline, five environment steps per packet")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Run paired RGB feedback comparison on exposed roots; no independent gate claim.", flush=True)
    try:
        with (np.load(inputs_path, allow_pickle=False) as inputs,
              h5py.File(prior.source.block.prior.common.DATA, "r") as data, torch.inference_mode()):
            for root in range(10, 50):
                task = int(inputs["rows"][root])
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                arms = {}
                for arm in search.ARMS:
                    start = time.perf_counter()
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task + 25])
                    old_tail, decisions, trace = None, [], []
                    for iteration in range(5):
                        ctx = {"pixels": transform(actor.render().copy())[None, None, None].cuda(), "goal": goal,
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        candidate_seed = 27121000 + root * 10 + iteration
                        strategy_seed = 27122000 + root * 10 + iteration
                        boundary = RGBLeWM(model, audit)
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        remaining = 25 - len(trace)
                        result = search.compile_rgb(boundary, ctx, scaler=scaler, old_tail=old_tail,
                                                    candidate_seed=candidate_seed, strategy_seed=strategy_seed,
                                                    strategy=arm, remaining_steps=remaining)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - before
                        chosen = result["actions"]
                        dispatched = []
                        for action in chosen[:5]:
                            _, _, term, trunc, info = actor.step(action.copy())
                            dispatched.append(action.tolist())
                            trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                            if term or trunc:
                                break
                        decisions.append({"iteration": iteration, "candidate_seed": candidate_seed,
                                          "strategy_seed": strategy_seed, "retained": result["retained_prefixes"],
                                          "parent_cost": result["parent_cost"].tolist(),
                                          "probe_cost": result["probe_cost"].tolist(),
                                          "tail_cost": result["tail_cost"].tolist(),
                                          "effects": result["effects"].tolist(),
                                          "choice": {"source": result["choice"].source,
                                                     "index": result["choice"].index, "cost": result["choice"].cost},
                                          "actions": chosen.tolist(), "actions_sha256": prior.action_hash(chosen),
                                          "old_tail_sha256": None if old_tail is None else prior.action_hash(old_tail),
                                          "dispatched": dispatched, "queries": result["model_queries"],
                                          "mode": result["mode"], "prefix_steps": result["prefix_steps"],
                                          "remaining_steps": remaining, "model_packet_steps": result["model_packet_steps"],
                                          "forward_calls": result["forward_calls"], "planning_seconds": seconds,
                                          "receipts": boundary.receipts})
                        old_tail = chosen[len(dispatched):].copy()
                        if term or trunc:
                            break
                    arms[arm] = {"decisions": decisions, "trace": trace,
                                 "ever_success": any(x["terminated"] for x in trace),
                                 "truncated": any(x["truncated"] for x in trace),
                                 "terminal_distance": trace[-1]["distance"],
                                 "queries": sum(x["queries"] for x in decisions),
                                 "wall_before_serialization_seconds": time.perf_counter() - start}
                rows.append({"root": root, "row": task, "arms": arms})
                print(f"root={root} success={[arms[a]['ever_success'] for a in search.ARMS]}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with OUT.open("x") as stream:
        json.dump({"track": RGB, "scope": "exposed RGB-only development; no independent confirmation",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "search_sha256": file_hash(search.__file__), "source_plan_sha256": file_hash(prior.source.PLAN),
                   "input_sha256": file_hash(inputs_path), "audit_hash": audit.fingerprint,
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer, "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
