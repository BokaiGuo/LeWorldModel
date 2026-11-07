"""Same-core retention ablation with live RGB feedback on exposed TwoRoom roots."""
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from experiments import pc_shared_rgb_feedback_smoke_20261107_codex as prior
from experiments import pc_rgb_deadline_scoring_20261107_codex as deadline
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_RGB_DEADLINE_SCORING_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_tworoom_rgb_deadline_feedback_20261107_codex.json"
ARMS = ("parent", "lookahead", "vector", "shuffled")


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
    paths = [__file__, PROTOCOL, prior.__file__, prior.shared.__file__, deadline.__file__,
             prior.source.PLAN, prior.source.block.prior.inputs.INPUT,
             prior.source.block.prior.common.CHECKPOINT / "weights.pt"]
    audit = freeze_audit(RGB, "lewm", paths, preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="single observed frame; five packets of five environment steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(prior.source.block.prior.common.DATA, "r") as data, torch.inference_mode()):
            if len(inputs["rows"]) < 50:
                raise ValueError("task roster incomplete")
            for root in range(10, 50):
                task = int(inputs["rows"][root])
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                arms = {}
                for arm in ARMS:
                    start_task = time.perf_counter()
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task + 25])
                    old_tail, decisions, trace = None, [], []
                    for iteration in range(5):
                        image = actor.render().copy()
                        ctx = {"pixels": transform(image)[None, None, None].cuda(), "goal": goal,
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        candidate_seed = 27121000 + root * 10 + iteration
                        strategy_seed = 27122000 + root * 10 + iteration
                        parents, probes, tails = prior.build_candidates(candidate_seed, old_tail)
                        boundary = RGBLeWM(model, audit)
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        remaining = 25 - len(trace)
                        result = deadline.compile_rgb(boundary, ctx, parents, probes, tails, scaler=scaler,
                                                      prefix_steps=10, retain_count=2, strategy=arm,
                                                      budget=13, seed=strategy_seed, remaining_steps=remaining)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - started
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
                                          "choice": {"source": result["choice"].source,
                                                     "index": result["choice"].index,
                                                     "cost": result["choice"].cost},
                                          "actions": chosen.tolist(), "actions_sha256": prior.action_hash(chosen),
                                          "old_tail_sha256": None if old_tail is None else prior.action_hash(old_tail),
                                          "dispatched": dispatched, "queries": result["model_queries"],
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
                                 "wall_before_serialization_seconds": time.perf_counter() - start_task}
                rows.append({"root": root, "row": task, "arms": arms})
                print(f"root={root} success={[arms[a]['ever_success'] for a in ARMS]}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with OUT.open("x") as stream:
        json.dump({"track": RGB, "scope": "exposed RGB-only development; no tuned baseline or independent confirmation",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "shared_sha256": file_hash(prior.shared.__file__), "adapter_sha256": file_hash(deadline.__file__),
                   "source_plan_sha256": file_hash(prior.source.PLAN), "audit_hash": audit.fingerprint,
                   "model_sha256": fingerprint, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
