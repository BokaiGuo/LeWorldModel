"""Live RGB feedback engineering smoke for the shared fixed-prefix compiler."""
import hashlib
import json
import time

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_block_cem_50dev_20261107_codex as source
from experiments import pc_fixed_prefix_rgb_shared_adapter_20261107_codex as adapter
from experiments import pc_fixed_prefix_shared_20261107_codex as shared
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_SHARED_RGB_FEEDBACK_SMOKE_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_shared_rgb_feedback_smoke_20261107_codex.json"


def build_candidates(seed, old_tail):
    rng = np.random.default_rng(seed)
    parents = np.repeat(rng.uniform(-1, 1, (3, 5, 2)).astype(np.float32), 5, axis=1)
    if old_tail is not None:
        old_tail = np.asarray(old_tail, dtype=np.float32)
        if old_tail.ndim != 2 or old_tail.shape[1] != 2 or len(old_tail) > 25:
            raise ValueError("invalid old tail")
        parents[0] = np.pad(old_tail, ((0, 25 - len(old_tail)), (0, 0)))
    probes = np.repeat(rng.uniform(-1, 1, (2, 3, 2)).astype(np.float32), 5, axis=1)
    tails = np.repeat(rng.uniform(-1, 1, (2, 3, 2)).astype(np.float32), 5, axis=1)
    return parents, probes, tails


def action_hash(actions):
    return hashlib.sha256(np.ascontiguousarray(actions, dtype="<f4").tobytes()).hexdigest()


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(source.PLAN.read_text())
    if plan["pool_sha256"] != file_hash(source.POOL):
        raise ValueError("source pool drift")
    model, scaler, transform, normalizer = source.block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    paths = [__file__, PROTOCOL, shared.__file__, adapter.__file__, source.PLAN,
             source.block.prior.inputs.INPUT, source.block.prior.common.CHECKPOINT / "weights.pt"]
    audit = freeze_audit(RGB, "lewm", paths, preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="one observed frame; five packets of five environment steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(source.block.prior.common.DATA, "r") as data, torch.inference_mode()):
            for root in (10, 11, 12):
                row = int(inputs["rows"][root])
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][row])
                actor._set_goal_state(data["proprio"][row + 25])
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                previous_tail, rounds, trace, done = None, [], [], False
                for iteration in range(5):
                    # Only the live image and the frozen goal enter the planner.
                    image = actor.render().copy()
                    ctx = {"pixels": transform(image)[None, None, None].cuda(),
                           "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    parents, probes, tails = build_candidates(27121000 + root * 10 + iteration, previous_tail)
                    boundary = RGBLeWM(model, audit)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = adapter.compile_rgb(boundary, ctx, parents, probes, tails, scaler=scaler,
                                                 prefix_steps=10, retain_count=1, strategy="lookahead",
                                                 budget=11, seed=27122000 + root * 10 + iteration)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    chosen = result["actions"]
                    dispatched = []
                    for action in chosen[:5]:
                        _, _, term, trunc, info = actor.step(action.copy())
                        dispatched.append(action.tolist())
                        trace.append({"action": action.tolist(), "terminated": bool(term), "truncated": bool(trunc),
                                      "distance": float(info["distance_to_target"])})
                        if term or trunc:
                            done = True
                            break
                    rounds.append({"iteration": iteration, "choice": {"source": result["choice"].source,
                                   "index": result["choice"].index, "cost": result["choice"].cost},
                                   "selected_actions": chosen.tolist(), "selected_sha256": action_hash(chosen),
                                   "carried_tail_length": 0 if previous_tail is None else len(previous_tail),
                                   "carried_tail_sha256": None if previous_tail is None else action_hash(previous_tail),
                                   "parent0_sha256": action_hash(parents[0]), "dispatched": dispatched,
                                   "logical_queries": result["model_queries"], "forward_calls": result["forward_calls"],
                                   "synchronized_planning_seconds": seconds, "receipts": boundary.receipts})
                    previous_tail = chosen[len(dispatched):].copy()
                    if done:
                        break
                rows.append({"root": root, "row": row, "rounds": rounds, "trace": trace,
                             "ever_success": any(t["terminated"] for t in trace),
                             "truncated": any(t["truncated"] for t in trace)})
                print(f"root={root} rounds={len(rounds)} steps={len(trace)}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weights changed")
    record = {"track": RGB, "scope": "engineering smoke; exposed tasks; no baseline or gate claim",
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "shared_sha256": file_hash(shared.__file__), "adapter_sha256": file_hash(adapter.__file__),
              "source_plan_sha256": file_hash(source.PLAN), "audit_hash": audit.fingerprint,
              "model_sha256": fingerprint, "rows": rows,
              "total_logical_queries": sum(round_["logical_queries"] for row in rows for round_ in row["rounds"])}
    with OUT.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)
    print(json.dumps({"tasks": len(rows), "success": sum(row["ever_success"] for row in rows),
                      "queries": record["total_logical_queries"]}, indent=2))


if __name__ == "__main__":
    main()
