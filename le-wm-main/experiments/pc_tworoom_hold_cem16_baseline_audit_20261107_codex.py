"""Reconstruct CEM16 candidates, rescore on GPU, and replay recorded actions."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_hold_cem16_baseline_20261107_codex as src
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit, tensor_receipt
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

OUT = Path(str(src.PREFIX) + "_audit.json")


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUT.exists():
        raise FileExistsError(OUT)
    run = json.loads(src.OUTPUT.read_text())
    if (run["pool_sha256"] != file_hash(src.POOL) or run["protocol_sha256"] != file_hash(src.PROTOCOL)
            or run["runner_sha256"] != file_hash(src.__file__)
            or run["comparison_sha256"] != file_hash(src.hold.OUT)):
        raise ValueError("source binding drift")
    common = src.hold.prior.source.block.prior.common
    inputs_path = src.hold.prior.source.block.prior.inputs.INPUT
    model, scaler, transform, normalizer = common.model_setup()
    if model_fingerprint(model) != run["model_sha256"] or normalizer != run["normalizer_sha256"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [__file__, src.OUTPUT, src.POOL, inputs_path,
                         common.CHECKPOINT / "weights.pt"], preprocessing="ImageNet 224",
                         clock="remaining deadline, five environment steps per packet")
    low, high = (-1 - scaler.mean_) / scaler.scale_, (1 - scaler.mean_) / scaler.scale_
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    max_diff, max_distance, checked = 0.0, 0.0, 0
    try:
        with np.load(src.POOL, allow_pickle=False) as pool, np.load(inputs_path, allow_pickle=False) as inputs, \
                h5py.File(common.DATA, "r") as data, torch.inference_mode():
            for task in run["rows"]:
                root, row = task["root"], task["row"]
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][row])
                actor._set_goal_state(data["proprio"][row + 25])
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                tail, position = None, 0
                for step in task["decisions"]:
                    remaining = 25 - position
                    if remaining != step["remaining"] or step["queries"] != 16:
                        raise ValueError("clock or query mismatch")
                    ctx = {"pixels": transform(actor.render().copy())[None, None, None].cuda(), "goal": goal,
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    for key in ("pixels", "goal", "action"):
                        if tensor_receipt(ctx[key]) != step["receipts"][0]["inputs"][key]:
                            raise ValueError("RGB context mismatch")
                    normal, recorded = pool[step["pool_key"] + "_normal"], pool[step["pool_key"] + "_cost"]
                    old = np.asarray(scaler.transform(src.cem.old_or_hold(tail)), np.float32)
                    np.testing.assert_allclose(normal[0], old, atol=1e-6, rtol=0)
                    rng = np.random.default_rng(step["seed"])
                    mean = np.broadcast_to(scaler.transform(np.zeros((1, 2)))[0], (5, 2)).copy()
                    std = np.ones((5, 2))
                    for index in range(3):
                        sample = src.cem.candidates(rng, mean, std, low, high, 5)
                        np.testing.assert_allclose(np.repeat(sample, 5, axis=1), normal[1+5*index:6+5*index],
                                                   rtol=0, atol=1e-6)
                        np.testing.assert_allclose(step["rounds"][index]["mean"], mean, rtol=0, atol=1e-8)
                        np.testing.assert_allclose(step["rounds"][index]["std"], std, rtol=0, atol=1e-8)
                        elite = sample[np.argsort(recorded[1+5*index:6+5*index], kind="stable")[:2]]
                        mean, std = elite.mean(0), np.maximum(elite.std(0), 1e-5)
                    boundary = RGBLeWM(model, audit)
                    values, _ = boundary.score(ctx, torch.as_tensor(normal[:, :remaining].reshape(
                        1, 16, remaining // 5, 10), dtype=torch.float32, device="cuda"))
                    values = values[0].cpu().numpy()
                    max_diff = max(max_diff, float(np.abs(values - recorded).max()))
                    if np.max(np.abs(values - recorded)) > 0.005 or int(np.argmin(recorded)) != step["chosen"]:
                        raise ValueError("score or selection mismatch")
                    if int(np.argmin(values)) != step["chosen"]:
                        raise ValueError("batch changed selected chain")
                    raw = np.clip(scaler.inverse_transform(normal[step["chosen"]]).astype(np.float32)[:remaining], -1, 1)
                    np.testing.assert_array_equal(raw, np.asarray(step["actions"], np.float32))
                    for action in step["dispatched"]:
                        _, _, term, trunc, info = actor.step(np.asarray(action, np.float32))
                        expected = task["trace"][position]
                        delta = abs(float(info["distance_to_target"]) - expected["distance"])
                        max_distance = max(max_distance, delta)
                        if delta > 1e-5 or bool(term) != expected["terminated"] or bool(trunc) != expected["truncated"]:
                            raise ValueError("physical replay mismatch")
                        position += 1
                    tail = raw[len(step["dispatched"]):].copy()
                    checked += 1
                if position != len(task["trace"]):
                    raise ValueError("incomplete trace")
                print(f"root={root} verified", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != run["model_sha256"]:
        raise ValueError("model mutation")
    with OUT.open("x") as f:
        json.dump({"source_sha256": file_hash(src.OUTPUT), "pool_sha256": file_hash(src.POOL),
                   "runner_sha256": file_hash(__file__), "decisions": checked,
                   "max_score_diff": max_diff, "max_distance_diff": max_distance}, f, indent=2)


if __name__ == "__main__":
    main()
