"""Actual feedback with one cached RGB scoring session per replan."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_incumbent_rgb_feedback_20261107_codex as original
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession

ROOT = original.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_OBSERVATION_CACHE_ACTUAL_FEEDBACK_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_tworoom_rgb_cache_actual_feedback_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUT.exists():
        raise FileExistsError(OUT)
    common = original.prior.prior.source.block.prior.common
    source = json.loads(original.prior.prior.source.PLAN.read_text())
    input_path = original.prior.prior.source.block.prior.inputs.INPUT
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != source["model_sha256"] or normalizer != source["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, Path(original.planner.__file__),
                         Path(__file__).parent / "pc_rgb_observation_cache_20261107_codex.py",
                         input_path, common.CHECKPOINT / "weights.pt", common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="remaining deadline, five steps per packet")
    encoder_calls = [0]
    hook = model.encoder.register_forward_hook(lambda *args: encoder_calls.__setitem__(0, encoder_calls[0] + 1))
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(input_path, allow_pickle=False) as inputs,
              h5py.File(common.DATA, "r") as data, torch.inference_mode()):
            for root in range(10, 14):
                row = int(inputs["rows"][root])
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                arms = {}
                for arm in original.planner.ARMS:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row + 25])
                    trace, decisions, old_tail = [], [], None
                    start = time.perf_counter()
                    for iteration in range(5):
                        remaining = 25 - len(trace)
                        current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                   "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        count = encoder_calls[0]
                        boundary = RGBObservationScoringSession(model, audit)

                        def score(actions):
                            norm = scaler.transform(actions.reshape(-1, 2)).reshape(-1, remaining // 5, 10)
                            values = []
                            for offset in range(0, len(actions), 32):
                                chunk = torch.as_tensor(norm[offset:offset + 32][None], device="cuda",
                                                        dtype=torch.float32)
                                value, _ = boundary.score(current, chunk)
                                values.extend(value[0].cpu().tolist())
                            return np.asarray(values)

                        result = original.planner.search(score, remaining=remaining, old_tail=old_tail,
                                                         seed=27121000 + root * 10 + iteration, arm=arm)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - before
                        chosen, dispatched = result["actions"], 0
                        for action in chosen[:5]:
                            _, _, term, trunc, info = actor.step(action.copy())
                            trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                            dispatched += 1
                            if term or trunc:
                                break
                        decisions.append({"iteration": iteration, "remaining": remaining,
                                          "actions": chosen.tolist(), "actions_sha256": result["actions_sha256"],
                                          "winner": result["winner"], "all_scores": result["all_scores"].tolist(),
                                          "retained": result["retained"], "dispatched": dispatched,
                                          "queries": result["model_queries"], "packet_steps": result["model_packet_steps"],
                                          "encoder_calls": encoder_calls[0] - count,
                                          "cache_hits": boundary.cache_hits, "cache_misses": boundary.cache_misses,
                                          "forward_calls": len(boundary.receipts), "planning_seconds": seconds})
                        old_tail = chosen[dispatched:].copy()
                        if term or trunc:
                            break
                    arms[arm] = {"trace": trace, "decisions": decisions,
                                 "ever_success": any(t["terminated"] for t in trace),
                                 "wall_seconds": time.perf_counter() - start}
                rows.append({"root": root, "row": row, "arms": arms})
                print(f"root={root} success={[arms[a]['ever_success'] for a in original.planner.ARMS]}", flush=True)
    finally:
        hook.remove()
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weights changed")
    result = {"track": RGB, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "planner_sha256": file_hash(original.planner.__file__),
              "cache_sha256": file_hash(Path(__file__).parent / "pc_rgb_observation_cache_20261107_codex.py"),
              "input_sha256": file_hash(input_path), "model_sha256": fingerprint,
              "normalizer_sha256": normalizer, "rows": rows,
              "scope": "four exposed roots, actual feedback development regression"}
    with OUT.open("x") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
