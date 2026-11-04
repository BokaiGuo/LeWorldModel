"""Static-camera RGB position with explicit unknown-velocity default."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from experiments.a0_pointmaze_native_feedback_time_20261102 import encode
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a1_pointmaze_rgb_task_spectral_dev_20261102 import marker
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import score_plans
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

CAMERA = ROOT / "results/a0_pointmaze_camera_metadata_20261103_DEV01.json"
CAMERA_SHA = "1b7e13ee87608cbc2fd5e98f3c42b098d0416c9425ec879e2cdd0bc963fc5ecf"
SOURCE = ROOT / "results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json"
SOURCE_SHA = "ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade"
REPLAY = ROOT / "results/a0_pointmaze_cem_all_candidates_20261102_DEV01.json"
REPLAY_SHA = "7dbf437b3f3473c467f6245ac788bd7de07d330f1ac17479b12e9f4b0b1f60fb"
PROPRIO = ROOT / "results/a0_pointmaze_proprio_action_response_20261103_DEV01.json"
PROPRIO_SHA = "ad34847ebcd6c4e6896cdda08b53838f0b16c1932a5e5816a7504dae8c9251e5"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_RGB_GEOMETRY_STATE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a0_pointmaze_rgb_geometry_state_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def estimate_state(image, metadata, body_offset):
    """Use image and fixed camera/scene geometry only; no measured state."""
    camera = metadata
    if camera["azimuth"] != 90.0 or camera["elevation"] != -90.0:
        raise ValueError("camera is not the declared top-down orientation")
    offset = np.asarray(body_offset, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("invalid static body offset")
    x, y = marker(image)
    distance, fovy = float(camera["camera_distance"]), float(camera["fovy_degrees"])
    center = np.asarray(camera["camera_lookat"], dtype=np.float64)
    if distance <= 0 or not 0 < fovy < 180 or center.shape != (3,):
        raise ValueError("invalid projection")
    # The legacy sim.render array is bottom-up. Its pixel y axis follows +world y.
    plane_z = float(offset[2] + camera["marker_geom_relative_position"][2])
    camera_z = center[2] + distance
    depth = camera_z - plane_z
    if depth <= 0:
        raise ValueError("marker plane behind camera")
    scale = 2 * depth * np.tan(np.deg2rad(fovy) / 2) / 224
    xy = center[:2] + (np.array([x, y]) - 111.5) * scale - offset[:2]
    return np.asarray([xy[0], xy[1], 0., 0.], np.float32), {"marker_xy": [x, y], "world_units_per_pixel": scale}


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((CAMERA, CAMERA_SHA), (SOURCE, SOURCE_SHA), (REPLAY, REPLAY_SHA), (PROPRIO, PROPRIO_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed source: {path}")
    camera_doc, source, replay, prop = (json.loads(p.read_text()) for p in (CAMERA, SOURCE, REPLAY, PROPRIO))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (REPLAY.with_suffix(".npz"), replay["arrays_sha256"]),
                           (DEFAULT_CHECKPOINT, source["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", source["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", source["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed input: {path}")
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera_doc["xml_sha256"]:
        raise RuntimeError("camera scene changed")
    scene = mujoco.MjModel.from_xml_string(xml)
    body = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_BODY, "particle")
    if body < 0:
        raise ValueError("particle body missing")
    offset = scene.body_pos[body].copy()
    src, rep = (np.load(p.with_suffix(".npz"), allow_pickle=False) for p in (SOURCE, REPLAY))
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, started = {}, [], time.monotonic()
    print("Test analytic RGB position with unknown initial velocity, fixed candidate banks.", flush=True)
    with lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"] or before != prop["model_before"]:
            raise RuntimeError("model identity mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        for old, actual, prior in zip(source["rows"], replay["rows"], prop["rows"]):
            root = old["root"]
            if not root == actual["root"] == prior["root"]:
                raise RuntimeError("root mismatch")
            image = src[f"r{root}_rgb0"]
            estimate, metadata = estimate_state(image, camera_doc["metadata"], offset)
            truth = dataset.states[root, 0].cpu().numpy().astype(np.float32)
            privileged_pos = np.array([truth[0], truth[1], 0, 0], dtype=np.float32)
            goal = encode(model, src[f"r{root}_goal_rgb"], device)
            mask = src[f"r{root}_initial_mask"]
            contexts = {name: encode_with_state(model, image, state, device)
                        for name, state in (("rgb_position_zero_velocity", estimate),
                                            ("privileged_position_zero_velocity", privileged_pos))}
            zero_visual = encode(model, image, device)["visual"]
            if not all(torch.equal(ctx["visual"], zero_visual) for ctx in contexts.values()):
                raise RuntimeError("visual input changed across state conditions")
            branches = []
            for prior_branch, rep_branch in zip(prior["branches"], actual["branches"]):
                method = prior_branch["method"]
                if method != rep_branch["method"]:
                    raise RuntimeError("method mismatch")
                key = f"r{root}_{method}"
                proposals = rep[f"{key}_proposals"]
                metrics = {}
                for name, context in contexts.items():
                    scores = score_plans(model, context, goal, mask, proposals, mean, std)
                    index = int(np.argmin(scores))
                    arrays[f"{key}_{name}_scores"] = scores
                    metrics[name] = {"choice": index, "distance_eval_only": rep_branch["distance_eval_only"][index]}
                metrics["zero_state"] = {"choice": prior_branch["original_choice"],
                                         "distance_eval_only": prior_branch["original_distance"]}
                metrics["privileged_full_state"] = {"choice": prior_branch["state_choice"],
                                                    "distance_eval_only": prior_branch["state_distance"]}
                branches.append({"method": method, "conditions": metrics})
            rows.append({"root": root, "rgb_state_estimate": estimate.tolist(), "estimate_metadata": metadata,
                         "privileged_initial_state_eval_only": truth.tolist(),
                         "position_error_eval_only": float(np.linalg.norm(estimate[:2]-truth[:2])),
                         "branches": branches})
            print(f"root {root}: RGB state and privileged diagnostic scores completed", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_rgb_geometry_state.v1", "camera_sha256": CAMERA_SHA,
              "source_sha256": SOURCE_SHA, "replay_sha256": REPLAY_SHA, "proprio_sha256": PROPRIO_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "body_offset_from_static_scene": offset.tolist(),
              "model_before": before, "model_after": after, "device": torch.cuda.get_device_name(device),
              "wall_seconds": time.monotonic()-started,
              "scope": "known-root fixed-bank development; RGB position with zero velocity; no P1", "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
