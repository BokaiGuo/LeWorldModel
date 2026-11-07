"""Replay real feedback decisions with and without decision-input encoding reuse."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_cem16_variance_floor_20261107_codex as source
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_shared_encoder_cache_20261107_codex import CachedJEPA
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

ROOT = source.base.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_CACHE_FEEDBACK_REPLAY_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_rgb_cache_feedback_replay_20261107_codex.json"


def compare_decisions(original, cached, saved):
    """Compare distributions, all scored samples, ties and final selected witness."""
    comparisons = []
    for a, b, old in zip(original["rounds"], cached["rounds"], saved["rounds"], strict=True):
        av, bv, ov = map(np.asarray, (a["costs"], b["costs"], old["costs"]))
        if not (np.allclose(av, bv, rtol=1e-5, atol=1e-3)
                and np.allclose(av, ov, rtol=1e-5, atol=1e-3)):
            raise ValueError("sample scores drift")
        elite_a = np.argsort(av, kind="stable")[:2]
        elite_b = np.argsort(bv, kind="stable")[:2]
        if not np.array_equal(elite_a, elite_b):
            raise ValueError("elite selection changed")
        comparisons.append(float(np.max(np.abs(av - bv))))
    if (original["chosen"] != cached["chosen"] or original["chosen"] != saved["chosen"]
            or not np.allclose(original["actions"], cached["actions"], atol=1e-6, rtol=0)
            or not np.allclose(original["actions"], saved["actions"], atol=1e-6, rtol=0)
            or not np.isclose(original["incumbent_cost"], cached["incumbent_cost"], atol=1e-3, rtol=1e-5)
            or not np.isclose(original["chosen_cost"], saved["chosen_cost"], atol=1e-3, rtol=1e-5)):
        raise ValueError("chosen action or incumbent drift")
    values = np.asarray([original["incumbent_cost"]] + [
        score for round_data in original["rounds"] for score in round_data["costs"]])
    ordered = np.sort(values, kind="stable")
    return {"max_sample_error": max(comparisons),
            "incumbent_error": abs(original["incumbent_cost"] - cached["incumbent_cost"]),
            "best_gap": float(ordered[1] - ordered[0]),
            "choice": original["chosen"]}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    historic = json.loads(source.OUTPUT.read_text())
    if (historic["runner_sha256"] != file_hash(source.__file__)
            or historic["protocol_sha256"] != file_hash(source.PROTOCOL)
            or historic["baseline_sha256"] != file_hash(source.base.OUTPUT)
            or historic["input_sha256"] != file_hash(source.base.INPUT)):
        raise ValueError("history provenance drift")
    model, scaler, transform, normalizer = source.base.common.model_setup()
    fingerprint = model_digest(model)
    if normalizer != historic["normalizer_sha256"] or model_fingerprint(model) != historic["model_sha256"]:
        raise ValueError("normalizer or historical checkpoint drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.OUTPUT, source.base.INPUT,
                         source.base.common.CHECKPOINT / "weights.pt",
                         source.base.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler",
                         clock="single observation, five physical steps per packet")
    proxy = CachedJEPA(model)
    def boundary(module):
        return SingleFrameRGB(module, audit, expected_model_digest=fingerprint,
                              action_mean=scaler.mean_, action_scale=scaler.scale_, packet_steps=5,
                              physical_low=-1, physical_high=1)
    original, cached = boundary(model), boundary(proxy)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Replay CEM16 on actual feedback observations and compare both complete planning paths.", flush=True)
    try:
        with (np.load(source.base.INPUT, allow_pickle=False) as inputs,
              h5py.File(source.base.common.DATA, "r") as dataset, torch.inference_mode()):
            for root, row in enumerate(historic["rows"]):
                if (int(inputs["episodes"][root]), int(inputs["rows"][root])) != (row["episode"], row["row"]):
                    raise ValueError("task identity drift")
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(dataset["proprio"][row["row"]])
                actor._set_goal_state(dataset["proprio"][row["row"] + 25])
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                tail, cursor = None, 0
                trace = row["arms"]["baseline"]["trace"]
                for iteration, decision in enumerate(row["arms"]["baseline"]["decisions"]):
                    remaining = 25 - cursor
                    if remaining != decision["remaining"]:
                        raise ValueError("clock mismatch")
                    context = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                               "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    start_encodes = proxy.encoder_calls
                    results, timings = {}, {}
                    order = (("original", original), ("cached", cached))
                    if (root + iteration) % 2:
                        order = order[::-1]
                    for name, wrapper in order:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        results[name] = source.base.cem_temporal(
                            wrapper, context, scaler, tail, remaining,
                            8140000 + row["episode"] * 10 + iteration, 16)
                        torch.cuda.synchronize()
                        timings[name] = time.perf_counter() - start
                    check = compare_decisions(results["original"], results["cached"], decision)
                    env_start = time.perf_counter()
                    for action in decision["dispatched"]:
                        _, _, term, trunc, info = actor.step(np.asarray(action, np.float32))
                        saved = trace[cursor]
                        if (abs(info["distance_to_target"] - saved["distance"]) > 1e-5
                                or bool(term) != saved["terminated"] or bool(trunc) != saved["truncated"]):
                            raise ValueError("physical trajectory drift")
                        np.testing.assert_allclose(action, saved["action"], atol=1e-6, rtol=0)
                        cursor += 1
                    env_seconds = time.perf_counter() - env_start
                    tail = np.asarray(decision["actions"], np.float32)[len(decision["dispatched"]):]
                    rows.append({"episode": row["episode"], "iteration": iteration, "remaining": remaining,
                                 "new_encoding_calls": proxy.encoder_calls - start_encodes,
                                 "timing": timings, "environment_seconds": env_seconds, **check})
                if cursor != len(trace):
                    raise ValueError("incomplete physical trace")
                print(f"episode={row['episode']} decisions={len(rows)}", flush=True)
    finally:
        env.close()
    original.verify_unchanged()
    cached.verify_unchanged()
    with OUT.open("x") as stream:
        json.dump({"scope": "paired replay at actual feedback observations, not a new control trial",
                   "device": torch.cuda.get_device_name(), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)), "source_sha256": file_hash(source.OUTPUT),
                   "cache_sha256": file_hash(Path(__file__).with_name("pc_rgb_shared_encoder_cache_20261107_codex.py")),
                   "model_digest": fingerprint, "normalizer_sha256": normalizer, "rows": rows},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
