"""Align predicted latents, encoded physical frames, and task outcomes in fixed pools."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_retention_counterfactual_pool_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

run = source.run
PROTOCOL = run.ROOT.parent / "docs/PC_RGB_POOL_TEMPORAL_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
OUT = run.ROOT / "results/pc_tworoom_pool_temporal_alignment_20261107_codex.json"
FEATURES = run.ROOT / "results/pc_tworoom_pool_temporal_alignment_20261107_codex.npz"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists() or FEATURES.exists():
        raise FileExistsError("result exists")
    pool = json.loads(source.OUTPUT.read_text())
    original = json.loads(run.OUTPUT.read_text())
    if (pool["source_sha256"] != file_hash(run.OUTPUT) or pool["input_sha256"] != file_hash(run.INPUT)
            or pool["runner_sha256"] != file_hash(source.__file__)
            or pool["protocol_sha256"] != file_hash(source.PROTOCOL)):
        raise ValueError("source artifact drift")
    model, scaler, transform, normalizer = run.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != pool["model_sha256"] or normalizer != original["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.OUTPUT, run.INPUT,
                                      run.common.CHECKPOINT / "weights.pt", run.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, dataset action scaler",
                         clock="five environment steps per packet")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, arrays = [], {}
    try:
        with (np.load(run.INPUT, allow_pickle=False) as inputs,
              h5py.File(run.common.DATA, "r") as data, torch.inference_mode()):
            indices = {int(ep): i for i, ep in enumerate(inputs["episodes"])}
            for k, entry in enumerate(pool["entries"]):
                idx = indices[entry["episode"]]
                row = int(inputs["rows"][idx])

                def reset_and_history():
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row + 25])
                    for action in entry["history"]:
                        _, _, term, trunc, _ = actor.step(np.asarray(action, dtype=np.float32))
                        if term or trunc:
                            raise ValueError("history already ended")
                    return actor

                actor = reset_and_history()
                goal = transform(inputs["goal"][idx])[None, None, None].cuda()
                ctx = {"pixels": transform(actor.render().copy())[None, None, None].cuda(), "goal": goal,
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
                chains = np.asarray(entry["actions"], dtype=np.float32)
                horizon = entry["remaining"]
                if chains.shape != (6, horizon, 2) or horizon % 5:
                    raise ValueError("invalid candidate clock")
                normal = scaler.transform(chains.reshape(-1, 2)).reshape(1, 6, horizon // 5, 10)
                endpoint, context = RGBLeWM(model, audit).score(
                    ctx, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                predicted = context["predicted_emb"][0, :, 1:]
                target = context["goal_emb"][0, -1]
                predicted_cost = (predicted - target).square().sum(-1).cpu().numpy()
                np.testing.assert_allclose(predicted_cost.min(-1), entry["temporal_cost"], atol=1e-3, rtol=0)
                np.testing.assert_allclose(endpoint[0].cpu().numpy(), entry["endpoint_cost"], atol=1e-3, rtol=0)
                images, physics = [], []
                for j, chain in enumerate(chains):
                    actor = reset_and_history()
                    trace = []
                    for t, action in enumerate(chain):
                        _, _, term, trunc, info = actor.step(action.copy())
                        if trunc:
                            raise ValueError("unexpected truncation")
                        item = {"distance": float(info["distance_to_target"]), "success": bool(term)}
                        trace.append(item)
                        if t < len(entry["physical"][j]):
                            expected = entry["physical"][j][t]
                            if (abs(item["distance"] - expected["distance"]) > 1e-5
                                    or item["success"] != expected["success"]):
                                raise ValueError("candidate physical replay mismatch")
                        if (t + 1) % 5 == 0:
                            images.append(transform(actor.render().copy()))
                    physics.append(trace)
                batch = torch.stack(images).cuda()
                actual = model.encode({"pixels": batch[:, None]})["emb"][:, 0].reshape(6, horizon // 5, -1)
                if actual.shape != predicted.shape or not torch.isfinite(actual).all():
                    raise ValueError("encoded time or feature dimension mismatch")
                actual_cost = (actual - target).square().sum(-1).cpu().numpy()
                arrays[f"p{k}"] = predicted.cpu().numpy()
                arrays[f"a{k}"] = actual.cpu().numpy()
                arrays[f"g{k}"] = target.cpu().numpy()
                rows.append({"episode": entry["episode"], "iteration": entry["iteration"],
                             "initial_success": entry["initial_success"], "prefix": entry["prefix"],
                             "predicted_cost": predicted_cost.tolist(), "actual_cost": actual_cost.tolist(),
                             "physical": physics})
                if (k + 1) % 20 == 0:
                    print(f"aligned {k+1}/{len(pool['entries'])} pools", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with OUT.open("x") as stream:
        json.dump({"source_sha256": file_hash(source.OUTPUT), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "feature_sha256": file_hash(FEATURES),
                   "model_sha256": fingerprint, "rows": rows}, stream, allow_nan=False)
    print(f"sealed {len(rows)} aligned pools", flush=True)


if __name__ == "__main__":
    main()
