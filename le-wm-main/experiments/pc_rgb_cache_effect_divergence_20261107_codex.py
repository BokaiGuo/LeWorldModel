"""Diagnose first cache-related effect selection divergence on identical feedback inputs."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_rgb_cached_closed_loop_cost_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_shared_encoder_cache_20261107_codex import CachedJEPA

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_CACHE_EFFECT_DIVERGENCE_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_rgb_cache_effect_divergence_20261107_codex.json"


def different(a, b):
    return a["choice"] != b["choice"] or not np.allclose(a["actions"], b["actions"], atol=1e-6, rtol=0)


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    historical = json.loads(source.OUT.read_text())
    if (historical["runner_sha256"] != file_hash(source.__file__)
            or historical["protocol_sha256"] != file_hash(source.PROTOCOL)
            or historical["input_sha256"] != file_hash(source.prior.INPUT)):
        raise ValueError("feedback provenance mismatch")
    model, scaler, transform, normalizer = source.prior.common.model_setup()
    digest = model_fingerprint(model)
    if digest != historical["model_sha256"] or normalizer != historical["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.prior.INPUT,
                         source.prior.common.CHECKPOINT / "weights.pt",
                         source.prior.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, frozen action scaler",
                         clock="five environment steps per packet")
    boundaries = {"original": RGBLeWM(model, audit), "cached": RGBLeWM(CachedJEPA(model), audit)}
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Replay first effect-cache divergence at identical legal observations and old tails.", flush=True)
    try:
        with np.load(source.prior.INPUT, allow_pickle=False) as inputs, \
                h5py.File(source.prior.common.DATA, "r") as data, torch.inference_mode():
            for index, root in enumerate(historical["rows"]):
                arms = root["arms"]["effect"]
                if arms["comparison"]["same"]:
                    continue
                a, b = arms["original"], arms["cached"]
                divergent = next(i for i, (da, db) in enumerate(zip(a["decisions"], b["decisions"], strict=True))
                                 if different(da, db))
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][root["row"]])
                actor._set_goal_state(data["proprio"][root["row"] + 25])
                cursor = 0
                tail = None
                for decision in a["decisions"][:divergent]:
                    for action in decision["dispatched"]:
                        _, _, term, trunc, info = actor.step(np.asarray(action, np.float32))
                        old = a["trace"][cursor]
                        if (abs(float(info["distance_to_target"]) - old["distance"]) > 1e-5
                                or bool(term) != old["terminated"] or bool(trunc) != old["truncated"]):
                            raise ValueError("reference physical drift")
                        cursor += 1
                    tail = np.asarray(decision["actions"], np.float32)[len(decision["dispatched"]):]
                context = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                           "goal": transform(inputs["goal"][index])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                results = {}
                for name, boundary in boundaries.items():
                    result = source.program.compile_rgb(boundary, context, scaler=scaler, old_tail=tail,
                        candidate_seed=27121000 + root["episode"] * 10 + divergent,
                        strategy_seed=27122000 + root["episode"] * 10 + divergent,
                        strategy="effect", remaining_steps=25-cursor, final_rule="temporal")
                    expected = arms[name]["decisions"][divergent]
                    choice = {"source": result["choice"].source, "index": result["choice"].index}
                    if choice != expected["choice"]:
                        raise ValueError("sealed choice did not replay")
                    np.testing.assert_allclose(result["actions"], expected["actions"], atol=1e-6, rtol=0)
                    results[name] = {"choice": choice, "actions": result["actions"].tolist(),
                        "retained": result["retained_prefixes"], "temporal_scores": result["temporal_scores"],
                        "parent_cost": result["parent_cost"].tolist(), "probe_cost": result["probe_cost"].tolist(),
                        "tail_cost": result["tail_cost"].tolist()}
                first, second = results["original"], results["cached"]
                if first["retained"] != second["retained"]:
                    score_error = None
                else:
                    score_error = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)), initial=0))
                                      for x, y in zip(first["temporal_scores"], second["temporal_scores"], strict=True))
                scores = np.concatenate([np.asarray(x) for x in first["temporal_scores"]])
                ordered = np.sort(scores)
                aa, bb = np.asarray(first["actions"]), np.asarray(second["actions"])
                rows.append({"episode": root["episode"], "iteration": divergent, "remaining": 25-cursor,
                    "first_five_max_abs": float(np.max(np.abs(aa[:5]-bb[:5]), initial=0)),
                    "tail_max_abs": float(np.max(np.abs(aa[5:]-bb[5:]), initial=0)),
                    "original_top_gap": float(ordered[1]-ordered[0]),
                    "same_retained": first["retained"] == second["retained"], "max_score_diff": score_error,
                    "original": first, "cached": second})
                print(f"episode={root['episode']} iteration={divergent} gap={ordered[1]-ordered[0]:.8g}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != digest:
        raise ValueError("model drift")
    with OUT.open("x") as stream:
        json.dump({"protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "source_sha256": file_hash(source.OUT), "rows": rows}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
