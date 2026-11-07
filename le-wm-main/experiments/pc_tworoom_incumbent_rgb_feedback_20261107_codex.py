"""Four exposed TwoRoom roots, actual RGB feedback, 680 candidate evaluations."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_rgb_hold_retention_feedback_20261107_codex as prior
from experiments import pc_rgb_block_incumbent_search_20261107_codex as planner
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_INCUMBENT_RGB_FEEDBACK_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_tworoom_incumbent_rgb_feedback_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    source = json.loads(prior.prior.source.PLAN.read_text())
    if source["pool_sha256"] != file_hash(prior.prior.source.POOL):
        raise ValueError("source candidate integrity failure")
    common = prior.prior.source.block.prior.common
    inputs_path = prior.prior.source.block.prior.inputs.INPUT
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != source["model_sha256"] or normalizer != source["normalizer_sha256"]:
        raise ValueError("model/normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, Path(planner.__file__),
                         inputs_path, common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="remaining deadline, five steps per packet")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(inputs_path, allow_pickle=False) as inputs,
              h5py.File(common.DATA, "r") as data, torch.inference_mode()):
            for root in range(10, 14):
                row = int(inputs["rows"][root])
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                arms = {}
                for arm in planner.ARMS:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row + 25])
                    trace, decisions, old_tail = [], [], None
                    task_start = time.perf_counter()
                    for iteration in range(5):
                        remaining = 25 - len(trace)
                        current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                   "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        boundary = RGBLeWM(model, audit)
                        forward_calls = 0

                        def score(actions):
                            nonlocal forward_calls
                            norm = scaler.transform(actions.reshape(-1, 2)).reshape(-1, remaining // 5, 10)
                            values = []
                            for offset in range(0, len(actions), 32):
                                packet = torch.as_tensor(norm[offset:offset + 32][None],
                                                         device="cuda", dtype=torch.float32)
                                value, _ = boundary.score(current, packet)
                                forward_calls += 1
                                values.extend(value[0].cpu().tolist())
                            return np.asarray(values)

                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        result = planner.search(score, remaining=remaining, old_tail=old_tail,
                                                seed=27121000 + root * 10 + iteration, arm=arm)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - t0
                        chosen = result["actions"]
                        dispatched = 0
                        for action in chosen[:5]:
                            _, _, term, trunc, info = actor.step(action.copy())
                            trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                            dispatched += 1
                            if term or trunc:
                                break
                        decisions.append({"iteration": iteration, "seed": 27121000 + root * 10 + iteration,
                                          "remaining": remaining, "actions": chosen.tolist(),
                                          "actions_sha256": result["actions_sha256"], "winner": result["winner"],
                                          "score": result["cost"], "all_scores": result["all_scores"].tolist(),
                                          "retained": result["retained"], "queries": result["model_queries"],
                                          "packet_steps": result["model_packet_steps"],
                                          "query_batches": result["query_batches"], "forward_calls": forward_calls,
                                          "old_tail_included": result["old_tail_included"],
                                          "planning_seconds": seconds, "dispatched": dispatched,
                                          "receipts": boundary.receipts})
                        old_tail = chosen[dispatched:].copy()
                        if term or trunc:
                            break
                    arms[arm] = {"trace": trace, "decisions": decisions,
                                 "ever_success": any(x["terminated"] for x in trace),
                                 "queries": sum(d["queries"] for d in decisions),
                                 "packet_steps": sum(d["packet_steps"] for d in decisions),
                                 "wall_seconds": time.perf_counter() - task_start}
                rows.append({"root": root, "row": row, "arms": arms})
                print(f"root={root} success={[arms[a]['ever_success'] for a in planner.ARMS]}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    result = {"track": RGB, "scope": "four exposed roots, development smoke only",
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "planner_sha256": file_hash(planner.__file__), "input_sha256": file_hash(inputs_path),
              "model_sha256": fingerprint, "normalizer_sha256": normalizer,
              "audit_hash": audit.fingerprint, "rows": rows}
    with OUT.open("x") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
