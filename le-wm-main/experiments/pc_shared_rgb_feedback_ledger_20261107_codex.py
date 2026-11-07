"""Complete scoring ledger for the RGB feedback engineering smoke."""
import json
import time

import h5py
import numpy as np
import torch

from experiments import pc_shared_rgb_feedback_smoke_20261107_codex as prior
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

PROTOCOL = prior.ROOT.parent / "docs/PC_SHARED_RGB_FEEDBACK_LEDGER_PROTOCOL_20261107_CODEX.md"
OUT = prior.ROOT / "results/pc_shared_rgb_feedback_ledger_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    old = json.loads(prior.OUT.read_text())
    if old["runner_sha256"] != file_hash(prior.__file__) or old["protocol_sha256"] != file_hash(prior.PROTOCOL):
        raise ValueError("prior source drift")
    plan = json.loads(prior.source.PLAN.read_text())
    model, scaler, transform, normalizer = prior.source.block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    deps = [__file__, PROTOCOL, prior.__file__, prior.shared.__file__, prior.adapter.__file__,
            prior.source.PLAN, prior.source.block.prior.inputs.INPUT,
            prior.source.block.prior.common.CHECKPOINT / "weights.pt"]
    audit = freeze_audit(RGB, "lewm", deps, preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="one observed frame; five packets of five environment steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(prior.source.block.prior.common.DATA, "r") as data, torch.inference_mode()):
            for root in (10, 11, 12):
                task = int(inputs["rows"][root])
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][task])
                actor._set_goal_state(data["proprio"][task + 25])
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                old_tail, rounds, trace = None, [], []
                started_task = time.perf_counter()
                for iteration in range(5):
                    ctx = {"pixels": transform(actor.render().copy())[None, None, None].cuda(), "goal": goal,
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    candidate_seed = 27121000 + root * 10 + iteration
                    strategy_seed = 27122000 + root * 10 + iteration
                    parents, probes, tails = prior.build_candidates(candidate_seed, old_tail)
                    boundary = RGBLeWM(model, audit)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = prior.adapter.compile_rgb(boundary, ctx, parents, probes, tails, scaler=scaler,
                                                       prefix_steps=10, retain_count=1, strategy="lookahead",
                                                       budget=11, seed=strategy_seed)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    chosen = result["actions"]
                    dispatched = []
                    for action in chosen[:5]:
                        _, _, term, trunc, info = actor.step(action.copy())
                        dispatched.append(action.tolist())
                        trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                        if term or trunc:
                            break
                    rounds.append({"iteration": iteration, "candidate_seed": candidate_seed,
                                   "strategy_seed": strategy_seed, "retained": result["retained_prefixes"],
                                   "parent_cost": result["parent_cost"].tolist(),
                                   "probe_cost": result["probe_cost"].tolist(),
                                   "tail_cost": result["tail_cost"].tolist(),
                                   "choice": {"source": result["choice"].source, "index": result["choice"].index,
                                              "cost": result["choice"].cost},
                                   "actions": chosen.tolist(), "actions_sha256": prior.action_hash(chosen),
                                   "old_tail_sha256": None if old_tail is None else prior.action_hash(old_tail),
                                   "dispatched": dispatched, "queries": result["model_queries"],
                                   "forward_calls": result["forward_calls"], "planning_seconds": seconds,
                                   "receipts": boundary.receipts})
                    old_tail = chosen[len(dispatched):].copy()
                    if term or trunc:
                        break
                rows.append({"root": root, "row": task, "rounds": rounds, "trace": trace,
                             "ever_success": any(x["terminated"] for x in trace),
                             "elapsed_before_serialization_seconds": time.perf_counter() - started_task})
                print(f"root={root} rounds={len(rounds)} steps={len(trace)}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    payload = {"track": RGB, "scope": "exposed engineering trace; not scientific confirmation",
               "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
               "prior_sha256": file_hash(prior.OUT), "shared_sha256": file_hash(prior.shared.__file__),
               "adapter_sha256": file_hash(prior.adapter.__file__), "model_sha256": fingerprint,
               "audit_hash": audit.fingerprint, "rows": rows}
    with OUT.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps({"tasks": len(rows), "rounds": sum(len(x["rounds"]) for x in rows),
                      "success": sum(x["ever_success"] for x in rows)}, indent=2))


if __name__ == "__main__":
    main()
