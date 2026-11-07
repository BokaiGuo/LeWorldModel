"""Paired, exposed-root RGB feedback comparison of the incumbent elite update."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_incumbent_rgb_feedback_20261107_codex as prior
from experiments import pc_rgb_block_incumbent_corrected_20261107_codex as corrected
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_INCUMBENT_ELITE_FIX_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_incumbent_elite_fix_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(prior.prior.prior.source.PLAN.read_text())
    if source["pool_sha256"] != file_hash(prior.prior.prior.source.POOL):
        raise ValueError("source candidate integrity failure")
    common = prior.prior.prior.source.block.prior.common
    inputs_path = prior.prior.prior.source.block.prior.inputs.INPUT
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != source["model_sha256"] or normalizer != source["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, Path(prior.planner.__file__),
                         Path(corrected.__file__), inputs_path, common.CHECKPOINT / "weights.pt",
                         common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="remaining deadline, five steps per packet")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(inputs_path, allow_pickle=False) as inputs,
              h5py.File(common.DATA, "r") as data, torch.inference_mode()):
            for root in range(10, 50):
                row = int(inputs["rows"][root])
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                arms = {}
                for name, planner in (("original", prior.planner), ("corrected", corrected)):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row + 25])
                    initial_rgb_sha256 = __import__("hashlib").sha256(actor.render().tobytes()).hexdigest()
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
                                                seed=27121000 + root * 10 + iteration, arm="baseline")
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
                        rank = (int(np.flatnonzero(np.argsort(result["all_scores"][:100], kind="stable") == 0)[0] + 1)
                                if old_tail is not None else None)
                        decisions.append({"iteration": iteration, "seed": 27121000 + root * 10 + iteration,
                                          "remaining": remaining, "actions": chosen.tolist(),
                                          "actions_sha256": result["actions_sha256"], "winner": result["winner"],
                                          "score": result["cost"], "all_scores": result["all_scores"].tolist(),
                                          "queries": result["model_queries"], "packet_steps": result["model_packet_steps"],
                                          "query_batches": result["query_batches"], "forward_calls": forward_calls,
                                          "old_tail_rank": rank, "old_tail_included": old_tail is not None,
                                          "old_tail_representable": result.get("old_tail_representable"),
                                          "old_tail_elite": result.get("old_tail_elite"),
                                          "planning_seconds": seconds, "dispatched": dispatched,
                                          "receipts": boundary.receipts})
                        old_tail = chosen[dispatched:].copy()
                        if term or trunc:
                            break
                    arms[name] = {"trace": trace, "decisions": decisions,
                                  "initial_rgb_sha256": initial_rgb_sha256,
                                  "ever_success": any(x["terminated"] for x in trace),
                                  "queries": sum(d["queries"] for d in decisions),
                                  "packet_steps": sum(d["packet_steps"] for d in decisions),
                                  "wall_seconds": time.perf_counter() - task_start}
                if arms["original"]["initial_rgb_sha256"] != arms["corrected"]["initial_rgb_sha256"]:
                    raise ValueError("initial image mismatch")
                rows.append({"root": root, "row": row, "arms": arms})
                print(f"root={root} original={arms['original']['ever_success']} "
                      f"corrected={arms['corrected']['ever_success']}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    paths = [Path(__file__), PROTOCOL, Path(prior.planner.__file__), Path(corrected.__file__),
             inputs_path, common.CHECKPOINT / "weights.pt"]
    with OUTPUT.open("x") as stream:
        json.dump({"track": RGB, "scope": "40 exposed roots; exploratory engineering comparison",
                   "source_sha256": {str(path): file_hash(path) for path in paths},
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer,
                   "audit_hash": audit.fingerprint, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
