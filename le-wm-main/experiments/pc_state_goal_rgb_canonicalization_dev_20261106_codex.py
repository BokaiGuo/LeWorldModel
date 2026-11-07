"""Run paired RGB canonicalization scoring before any physical label access."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pc_goal_rgb_canonicalization import goal_shift, translate
from experiments.pc_input_permissions import StateAssistedPointMaze, file_hash, manifest
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import ROOT, load_model

SOURCE = ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz"
OLD = ROOT / "results/pc_state_effect_local_dev_20261106_codex.json"
OUT = ROOT / "results/pc_state_goal_rgb_canonicalization_dev_20261106_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_STATE_GOAL_RGB_CANONICALIZATION_DEV_PROTOCOL_20261106_CODEX.md"


def digest(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    previous = json.loads(OLD.read_text())
    if previous["task_manifest_hash"] != file_hash(SOURCE):
        raise ValueError("physical source drift")
    print("Paired visual-only alignment, state-assisted JEPA scoring; labels excluded until audit.", flush=True)
    model, dataset, audit = load_model()
    fingerprint = model_fingerprint(model)
    wrapper = StateAssistedPointMaze(model, audit)
    mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    observations = []
    with np.load(SOURCE, allow_pickle=False) as data:
        for root in range(4):
            for camera in range(2):
                observations.append({
                    "root": 1880 + root, "camera": camera,
                    "current": data["images"][2*root, 0, camera, 0].copy(),
                    "goal": data["images"][2*root+1, 0, camera, 0].copy(),
                    "proprio": data["traces"][2*root, 0, camera, 0].copy(),
                })
    rows = []
    with torch.inference_mode():
        for observation in observations:
            root, camera = observation["root"], observation["camera"]
            arms = {}
            for arm in ("raw", "goal_rgb_centered"):
                torch.cuda.synchronize()
                started = time.perf_counter()
                current, goal = observation["current"], observation["goal"]
                transform_started = time.perf_counter()
                shift = goal_shift(goal)
                if arm == "goal_rgb_centered":
                    current, current_valid = translate(current, shift["dx"], shift["dy"])
                    goal, goal_valid = translate(goal, shift["dx"], shift["dy"])
                    if not np.array_equal(current_valid, goal_valid):
                        raise ValueError("target/current translation mismatch")
                    valid_pixels = int(current_valid.sum())
                else:
                    valid_pixels = 224 * 224
                transform_seconds = time.perf_counter() - transform_started
                inputs = {"visual": rgb_tensor(current),
                          "proprio": torch.tensor(observation["proprio"], device="cuda",
                                                  dtype=torch.float32)[None, None]}
                goal_emb = model.encode(rgb_tensor(goal))[0]
                values = []
                for x in (0., .1, -.1):
                    raw = torch.tensor([[x, 0.]] * 5, device="cuda")
                    packet = ((raw - mean) / std).reshape(1, 1, 10)
                    pred = wrapper.predict(inputs, packet)["visual"][-1]
                    value = float((pred - goal_emb).square().mean())
                    if not np.isfinite(value):
                        raise ValueError("nonfinite cost")
                    values.append(value)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                arms[arm] = {"scores": values, "selected": int(np.argmin(values)), "shift": shift,
                             "valid_pixels": valid_pixels, "transform_seconds": transform_seconds,
                             "planning_seconds": seconds, "prediction_calls": 3, "goal_encoding_calls": 1,
                             "current_image_sha256": digest(current), "goal_image_sha256": digest(goal)}
            previous_row = next(row for row in previous["rows"] if row["root"] == root and row["camera"] == camera)
            np.testing.assert_allclose(arms["raw"]["scores"], previous_row["predicted_costs_zero_plus_minus"],
                                       atol=1e-6, rtol=1e-5)
            rows.append({"root": root, "camera": camera, "arms": arms})
            print(f"root={root} camera={camera} raw={arms['raw']['selected']} "
                  f"centered={arms['goal_rgb_centered']['selected']}", flush=True)
    if fingerprint != model_fingerprint(model):
        raise ValueError("model fingerprint changed")
    record = manifest(audit, algorithm_version="goal_rgb_centered_dev_v1", protocol_path=PROTOCOL,
                      task_manifest_path=SOURCE, receipts=wrapper.receipts)
    record.update(rows=rows, model_fingerprint=fingerprint, old_result_sha256=file_hash(OLD),
                  runner_sha256=file_hash(__file__),
                  transform_sha256=file_hash(Path(__file__).with_name("pc_goal_rgb_canonicalization.py")),
                  scope="state-assisted same-exposed-roots paired selection; labels unavailable to scorer")
    with OUT.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(OUT), "score_calls": len(wrapper.receipts), "conditions": len(rows)}))


if __name__ == "__main__":
    main()
