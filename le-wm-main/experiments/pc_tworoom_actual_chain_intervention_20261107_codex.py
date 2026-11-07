"""Intervene on actual dispatched first packet; compare fixed and equal-budget replanned tails."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as run
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

PROTOCOL = run.ROOT.parent / "docs/PC_RGB_ACTUAL_CHAIN_INTERVENTION_PROTOCOL_20261107_CODEX.md"
OUTPUT = run.ROOT / "results/pc_tworoom_actual_chain_intervention_20261107_codex.json"
ARMS = ("actual", "lookahead", "zero", "random")


def make_prefixes(row):
    chain = np.asarray([x["action"] for x in row["arms"]["effect"]["trace"]], dtype=np.float32)
    if chain.ndim != 2 or chain.shape[1] != 2 or len(chain) > 25:
        raise ValueError("invalid executed chain")
    if len(chain) < 5:
        return chain, None
    look = np.asarray(row["arms"]["lookahead"]["decisions"][0]["actions"][:5], dtype=np.float32)
    rng = np.random.default_rng(7316000 + row["episode"])
    random = np.repeat(rng.uniform(-1, 1, (1, 2)).astype(np.float32), 5, axis=0)
    prefixes = {"actual": chain[:5], "lookahead": look, "zero": np.zeros((5, 2), np.float32), "random": random}
    if any(v.shape != (5, 2) or not np.isfinite(v).all() or np.abs(v).max() > 1 for v in prefixes.values()):
        raise ValueError("illegal intervention")
    return chain, prefixes


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    original = json.loads(run.OUTPUT.read_text())
    if (original["runner_sha256"] != file_hash(run.__file__)
            or original["protocol_sha256"] != file_hash(run.PROTOCOL)
            or original["input_sha256"] != file_hash(run.INPUT)):
        raise ValueError("source provenance drift")
    model, scaler, transform, normalizer = run.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != original["model_sha256"] or normalizer != original["normalizer_sha256"]:
        raise ValueError("model/normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, run.OUTPUT, run.INPUT,
                                      run.common.CHECKPOINT / "weights.pt", run.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 and TwoRoom dataset action scaler",
                         clock="five-step intervention, 20-step CEM16 continuation")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(run.INPUT, allow_pickle=False) as inputs,
              h5py.File(run.common.DATA, "r") as data, torch.inference_mode()):
            for i, row in enumerate(original["rows"]):
                if row["episode"] != int(inputs["episodes"][i]) or row["row"] != int(inputs["rows"][i]):
                    raise ValueError("episode mismatch")
                chain, prefixes = make_prefixes(row)
                result = {"episode": row["episode"], "initial_success": row["initial_success"],
                          "original_success": row["arms"]["effect"]["ever_success"],
                          "original_actions": chain.tolist(), "arms": {}}
                if prefixes is None:
                    result["ineligible_reason"] = "original execution shorter than five steps"
                    rows.append(result)
                    continue
                goal = transform(inputs["goal"][i])[None, None, None].cuda()

                def reset():
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    return actor

                def execute(actor, actions, stop=True):
                    trace = []
                    for action in actions:
                        _, _, term, trunc, info = actor.step(np.asarray(action, dtype=np.float32))
                        trace.append({"action": np.asarray(action).tolist(), "distance": float(info["distance_to_target"]),
                                      "success": bool(term), "truncated": bool(trunc)})
                        if trunc:
                            raise ValueError("unexpected truncation")
                        if term and stop:
                            break
                    return trace

                for name in ARMS:
                    actor = reset()
                    start = float((actor.agent_position - actor.target_position).norm())
                    prefix_trace = execute(actor, prefixes[name], stop=False)
                    current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(), "goal": goal,
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    boundary = RGBLeWM(model, audit)
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    plan = run.cem_temporal(boundary, current, scaler, None, 20, 9440000 + row["episode"], 16)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - before
                    replanned_trace = execute(actor, plan["actions"], stop=True)
                    actor = reset()
                    common_prefix = execute(actor, prefixes[name], stop=False)
                    np.testing.assert_allclose([x["distance"] for x in common_prefix],
                                               [x["distance"] for x in prefix_trace], atol=1e-5, rtol=0)
                    common_tail = execute(actor, chain[5:], stop=True) if not any(x["success"] for x in common_prefix) else []
                    if name == "actual":
                        reproduced = (common_prefix + common_tail)[:len(chain)]
                        recorded = row["arms"]["effect"]["trace"]
                        np.testing.assert_allclose([x["distance"] for x in reproduced],
                                                   [x["distance"] for x in recorded], atol=1e-5, rtol=0)
                        if [x["success"] for x in reproduced] != [x["terminated"] for x in recorded]:
                            raise ValueError("original actual trace not reproduced")
                    result["arms"][name] = {"prefix": prefixes[name].tolist(), "start_distance": start,
                                            "prefix_trace": prefix_trace, "common_tail_trace": common_tail,
                                            "replanned_actions": plan["actions"].tolist(),
                                            "replanned_trace": replanned_trace, "queries": plan["queries"],
                                            "packet_steps": plan["packet_steps"], "forward_calls": plan["forward_calls"],
                                            "planning_seconds": elapsed, "chosen": plan["chosen"],
                                            "chosen_cost": plan["chosen_cost"], "seed": 9440000 + row["episode"]}
                rows.append(result)
                print(f"episode={row['episode']} completed", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with OUTPUT.open("x") as stream:
        json.dump({"source_sha256": file_hash(run.OUTPUT), "input_sha256": file_hash(run.INPUT),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "model_sha256": fingerprint, "rows": rows}, stream, allow_nan=False)
    print(f"sealed {len(rows)} actual-chain intervention rows", flush=True)


if __name__ == "__main__":
    main()
