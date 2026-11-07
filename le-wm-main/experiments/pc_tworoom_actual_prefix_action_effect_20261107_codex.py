"""Evaluate a common five-step action bank after real executed-prefix interventions."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_actual_chain_intervention_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

run = source.run
PROTOCOL = run.ROOT.parent / "docs/PC_RGB_ACTUAL_PREFIX_ACTION_EFFECT_PROTOCOL_20261107_CODEX.md"
OUT = run.ROOT / "results/pc_tworoom_actual_prefix_action_effect_20261107_codex.json"
FEATURES = run.ROOT / "results/pc_tworoom_actual_prefix_action_effect_20261107_codex.npz"


def probes(original):
    directions = np.asarray([[0., 0.], [.6, 0.], [-.6, 0.], [0., .6], [0., -.6]], np.float32)
    result = np.repeat(directions[:, None, :], 5, axis=1)
    if len(original) >= 10:
        result = np.concatenate((result, np.asarray(original[5:10], np.float32)[None]))
    if not np.isfinite(result).all() or np.abs(result).max() > 1:
        raise ValueError("invalid probe")
    return result


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists() or FEATURES.exists():
        raise FileExistsError("result exists")
    prior = json.loads(source.OUTPUT.read_text())
    base = json.loads(run.OUTPUT.read_text())
    if (prior["source_sha256"] != file_hash(run.OUTPUT) or prior["input_sha256"] != file_hash(run.INPUT)
            or prior["protocol_sha256"] != file_hash(source.PROTOCOL)
            or prior["runner_sha256"] != file_hash(source.__file__)):
        raise ValueError("source drift")
    model, scaler, transform, normalizer = run.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != prior["model_sha256"] or normalizer != base["normalizer_sha256"]:
        raise ValueError("model/normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.OUTPUT, run.INPUT,
                                      run.common.CHECKPOINT / "weights.pt", run.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, TwoRoom dataset action scaler",
                         clock="5-step prefix and 5-step probe")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    records, vectors = [], {}
    try:
        with (np.load(run.INPUT, allow_pickle=False) as inputs,
              h5py.File(run.common.DATA, "r") as data, torch.inference_mode()):
            index = {int(ep): i for i, ep in enumerate(inputs["episodes"])}
            for row in prior["rows"]:
                if "ineligible_reason" in row:
                    records.append({"episode": row["episode"], "ineligible_reason": row["ineligible_reason"]})
                    continue
                idx = index[row["episode"]]
                offset = int(inputs["rows"][idx])
                goal = transform(inputs["goal"][idx])[None, None, None].cuda()
                bank = probes(row["original_actions"])
                output = {"episode": row["episode"], "initial_success": row["initial_success"],
                          "original_success": row["original_success"], "probes": bank.tolist(), "arms": {}}

                def prefix_state(name):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][offset])
                    actor._set_goal_state(data["proprio"][offset + 25])
                    steps = []
                    for action in row["arms"][name]["prefix"]:
                        _, _, term, trunc, info = actor.step(np.asarray(action, np.float32))
                        if trunc:
                            raise ValueError("unexpected prefix truncation")
                        steps.append({"distance": float(info["distance_to_target"]), "success": bool(term)})
                    recorded = row["arms"][name]["prefix_trace"]
                    np.testing.assert_allclose([x["distance"] for x in steps], [x["distance"] for x in recorded],
                                               atol=1e-5, rtol=0)
                    if [x["success"] for x in steps] != [x["success"] for x in recorded]:
                        raise ValueError("prefix replay mismatch")
                    return actor, steps

                for name in source.ARMS:
                    actor, before = prefix_state(name)
                    current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(), "goal": goal,
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    normal = scaler.transform(bank.reshape(-1, 2)).reshape(1, len(bank), 1, 10)
                    endpoint, context = RGBLeWM(model, audit).score(
                        current, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                    predicted = context["predicted_emb"][0, :, -1]
                    target = context["goal_emb"][0, -1]
                    trace, frames = [], []
                    for action_chain in bank:
                        actor, _ = prefix_state(name)
                        steps = []
                        for action in action_chain:
                            _, _, term, trunc, info = actor.step(action.copy())
                            if trunc:
                                raise ValueError("unexpected probe truncation")
                            steps.append({"distance": float(info["distance_to_target"]), "success": bool(term)})
                        trace.append(steps)
                        frames.append(transform(actor.render().copy()))
                    actual = model.encode({"pixels": torch.stack(frames).cuda()[:, None]})["emb"][:, 0]
                    actual_cost = (actual - target).square().sum(-1)
                    calculated = (predicted - target).square().sum(-1)
                    if not torch.allclose(endpoint[0], calculated, atol=1e-3, rtol=0):
                        raise ValueError("prediction score mismatch")
                    key = f"e{row['episode']}_{name}"
                    vectors[key + "_predicted"] = predicted.cpu().numpy()
                    vectors[key + "_actual"] = actual.cpu().numpy()
                    vectors[key + "_goal"] = target.cpu().numpy()
                    output["arms"][name] = {"prefix_trace": before, "trace": trace,
                                            "predicted_cost": endpoint[0].cpu().tolist(),
                                            "encoded_cost": actual_cost.cpu().tolist()}
                records.append(output)
                print(f"episode {row['episode']} probes={len(bank)}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **vectors)
    with OUT.open("x") as stream:
        json.dump({"source_sha256": file_hash(source.OUTPUT), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "feature_sha256": file_hash(FEATURES),
                   "model_sha256": fingerprint, "rows": records}, stream, allow_nan=False)
    print(f"sealed {len(records)} tasks", flush=True)


if __name__ == "__main__":
    main()
