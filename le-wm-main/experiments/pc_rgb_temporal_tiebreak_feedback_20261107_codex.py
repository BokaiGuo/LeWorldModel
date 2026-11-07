"""Paired RGB effect feedback with original and endpoint-secondary tie rules."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_rgb_cached_closed_loop_cost_20261107_codex as source
from experiments import pc_rgb_temporal_tiebreak_20261107_codex as new
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_shared_encoder_cache_20261107_codex import CachedJEPA

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_TEMPORAL_TIEBREAK_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_rgb_temporal_tiebreak_feedback_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    old = json.loads(source.OUT.read_text())
    if (old["runner_sha256"] != file_hash(source.__file__)
            or old["protocol_sha256"] != file_hash(source.PROTOCOL)
            or old["input_sha256"] != file_hash(source.prior.INPUT)):
        raise ValueError("reference provenance drift")
    model, scaler, transform, normalizer = source.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != old["model_sha256"] or normalizer != old["normalizer_sha256"]:
        raise ValueError("model or normalizer changed")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.prior.INPUT,
                         source.prior.common.CHECKPOINT / "weights.pt",
                         source.prior.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 with fixed action normalizer",
                         clock="five environment steps per packet")
    modules = {"original": model, "cached": CachedJEPA(model)}
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Compare temporal tie rules in paired independent feedback executions.", flush=True)
    try:
        with np.load(source.prior.INPUT, allow_pickle=False) as inputs, \
                h5py.File(source.prior.common.DATA, "r") as data, torch.inference_mode():
            for index, root in enumerate(old["rows"]):
                episode, row = root["episode"], root["row"]
                if episode != 9000 + index or int(inputs["rows"][index]) != row:
                    raise ValueError("episode panel drift")
                goal = transform(inputs["goal"][index])[None, None, None].cuda()
                arms = {}
                for rule in ("baseline", "endpoint_tie"):
                    arms[rule] = {}
                    order = ("original", "cached") if (index + (rule == "endpoint_tie")) % 2 == 0 else ("cached", "original")
                    for version in order:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        env.reset(seed=42)
                        actor = env.unwrapped
                        actor._set_state(data["proprio"][row])
                        actor._set_goal_state(data["proprio"][row + 25])
                        boundary = RGBLeWM(modules[version], audit)
                        tail, decisions, trace = None, [], []
                        for iteration in range(5):
                            remaining = 25 - len(trace)
                            context = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                       "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                            kwargs = dict(scaler=scaler, old_tail=tail,
                                candidate_seed=27121000 + episode * 10 + iteration,
                                strategy_seed=27122000 + episode * 10 + iteration,
                                strategy="effect", remaining_steps=remaining)
                            if rule == "baseline":
                                result = source.program.compile_rgb(boundary, context, final_rule="temporal", **kwargs)
                            else:
                                result = new.compile_rgb(boundary, context, **kwargs)
                            actions = result["actions"]
                            dispatched = []
                            for action in actions[:5]:
                                _, _, term, trunc, info = actor.step(action.copy())
                                dispatched.append(action.tolist())
                                trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                              "terminated": bool(term), "truncated": bool(trunc)})
                                if term or trunc:
                                    break
                            decisions.append({"remaining": remaining,
                                "choice": {"source": result["choice"].source, "index": result["choice"].index},
                                "selected_temporal": result["choice"].cost,
                                "selected_endpoint": result.get("selected_endpoint_cost"),
                                "queries": result["model_queries"], "packet_steps": result["model_packet_steps"],
                                "actions": actions.tolist(), "dispatched": dispatched})
                            tail = actions[len(dispatched):].copy()
                            if term or trunc:
                                break
                        torch.cuda.synchronize()
                        arms[rule][version] = {"trace": trace, "decisions": decisions,
                            "success": any(x["terminated"] for x in trace),
                            "task_seconds": time.perf_counter() - start}
                    arms[rule]["comparison"] = source.compare(arms[rule]["original"], arms[rule]["cached"])
                for version in ("original", "cached"):
                    check = source.compare(root["arms"]["effect"][version], arms["baseline"][version])
                    if not check["same"]:
                        raise ValueError(f"baseline replay drift: episode={episode} version={version} {check}")
                rows.append({"episode": episode, "row": row, "arms": arms})
                print(f"episode={episode} baseline_same={arms['baseline']['comparison']['same']} "
                      f"tiebreak_same={arms['endpoint_tie']['comparison']['same']}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with OUT.open("x") as stream:
        json.dump({"scope": "exposed RGB development episodes; not independent confirmation",
                   "track": RGB, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "selector_sha256": file_hash(new.__file__), "source_sha256": file_hash(source.OUT),
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer, "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
