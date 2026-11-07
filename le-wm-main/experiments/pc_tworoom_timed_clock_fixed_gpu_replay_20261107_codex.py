"""Run fixed-clock RGB feedback and compare with sealed historical decisions."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_rgb_timed_prefix_feedback_20261107_codex as historical
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_timed_prefix_clock_fixed_20261107_codex import compile_timed_clock_fixed

ROOT = historical.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_TIMED_CLOCK_FIXED_GPU_REPLAY_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_tworoom_timed_clock_fixed_gpu_replay_20261107_codex.json"
ROOTS = (10, 11, 17, 21)


def compare_decision(new, old):
    fields = ("parent_cost", "probe_cost", "tail_cost")
    errors = [float(np.max(np.abs(np.asarray(new[k]) - np.asarray(old[k]))))
              for k in fields if np.asarray(new[k]).size]
    old_choice = old["choice"]
    receipts = old["receipts"]
    new_receipts = new["receipts"]
    return {
        "chosen_source_index_equal": (new["choice"]["source"], new["choice"]["index"])
                                     == (old_choice["source"], old_choice["index"]),
        "chosen_action_equal": np.array_equal(np.asarray(new["actions"], np.float32),
                                              np.asarray(old["actions"][:new["remaining_steps"]], np.float32)),
        "score_max_abs_error": max(errors, default=0.0),
        "scores_close": all(np.allclose(new[k], old[k], rtol=1e-5, atol=1e-4) for k in fields),
        "input_receipts_equal": len(receipts) == len(new_receipts) and all(
            a["inputs"] == b["inputs"] for a, b in zip(receipts, new_receipts, strict=True)),
    }


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    source = historical.prior.source
    plan = json.loads(source.PLAN.read_text())
    if plan["pool_sha256"] != file_hash(source.POOL):
        raise ValueError("source pool drift")
    model, scaler, transform, normalizer = source.block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL,
                         Path(compile_timed_clock_fixed.__code__.co_filename),
                         historical.timed.__file__, historical.__file__, source.PLAN,
                         source.block.prior.inputs.INPUT, source.block.prior.common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="single observed frame; five environment steps per action packet")
    print("Run fixed-clock RGB feedback; historical records are read only after all actions.", flush=True)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(source.block.prior.common.DATA, "r") as data, torch.inference_mode()):
            for root in ROOTS:
                task = int(inputs["rows"][root])
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                arms = {}
                for arm in historical.ARMS:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task + 25])
                    old_tail, decisions, trace = None, [], []
                    for iteration in range(5):
                        image = actor.render().copy()
                        ctx = {"pixels": transform(image)[None, None, None].cuda(), "goal": goal,
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        remaining = 25 - len(trace)
                        boundary = RGBLeWM(model, audit)
                        result = compile_timed_clock_fixed(boundary, ctx, scaler=scaler, old_tail=old_tail,
                                                          candidate_seed=27121000 + root * 10 + iteration,
                                                          strategy_seed=27122000 + root * 10 + iteration,
                                                          strategy=arm, remaining_steps=remaining)
                        chosen = result["actions"]
                        if chosen.shape != (remaining, 2):
                            raise ValueError("selected actions violate deadline")
                        dispatched = []
                        for action in chosen[:5]:
                            _, _, term, trunc, info = actor.step(action.copy())
                            dispatched.append(action.tolist())
                            trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                            if term or trunc:
                                break
                        decisions.append({"remaining_steps": remaining, "actions": chosen.tolist(),
                                          "dispatched": dispatched,
                                          "choice": {"source": result["choice"].source,
                                                     "index": result["choice"].index,
                                                     "cost": result["choice"].cost},
                                          "parent_cost": result["parent_cost"].tolist(),
                                          "probe_cost": result["probe_cost"].tolist(),
                                          "tail_cost": result["tail_cost"].tolist(),
                                          "receipts": boundary.receipts})
                        old_tail = chosen[len(dispatched):].copy()
                        if term or trunc:
                            break
                    arms[arm] = {"decisions": decisions, "trace": trace}
                rows.append({"root": root, "row": task, "arms": arms})
                print(f"completed root={root}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    # Historical evidence is evaluation-only, after all planning and execution.
    old = json.loads(historical.OUT.read_text())
    if old["adapter_sha256"] != file_hash(historical.timed.__file__) or old["runner_sha256"] != file_hash(historical.__file__):
        raise ValueError("historical source drift")
    old_rows = {row["root"]: row for row in old["rows"]}
    comparison = []
    for row in rows:
        for arm in historical.ARMS:
            new, previous = row["arms"][arm], old_rows[row["root"]]["arms"][arm]
            count_equal = len(new["decisions"]) == len(previous["decisions"])
            decisions = [compare_decision(a, b) for a, b in zip(new["decisions"], previous["decisions"])]
            a, b = new["trace"], previous["trace"]
            trace_equal = len(a) == len(b) and all(
                x["action"] == y["action"] and x["terminated"] == y["terminated"]
                and x["truncated"] == y["truncated"] and abs(x["distance"] - y["distance"]) <= 1e-5
                for x, y in zip(a, b, strict=True))
            comparison.append({"root": row["root"], "arm": arm, "decision_count_equal": count_equal,
                               "trace_equal": trace_equal, "decisions": decisions})
    with OUT.open("x") as stream:
        json.dump({"track": RGB, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "fixed_adapter_sha256": file_hash(Path(compile_timed_clock_fixed.__code__.co_filename)),
                   "historical_sha256": file_hash(historical.OUT), "model_sha256": fingerprint,
                   "input_audit_hash": audit.fingerprint, "rows": rows, "comparison": comparison,
                   "scope": "exposed RGB development; equivalence regression, not independent performance"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
