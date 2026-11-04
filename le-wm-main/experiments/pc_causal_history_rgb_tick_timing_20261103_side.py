"""Instrument the legacy RGB bridge loop; never supply measurements to planning."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_causal_history_rgb_tick_timing_20261103_side"
SOURCE = ROOT / "results/pc_causal_history_full_bank_replan_20261103_DEV01.json"
CAMERA = ROOT / "results/a0_pointmaze_camera_metadata_20261103_DEV01.json"
OLD = ROOT / "results/a1_pointmaze_rgb_state_full_planning_20261103_DEV01.json"
STATES = ROOT / "data/point_maze_v16_obstacle_challenge_extension_20261021/states.pth"
BRIDGE = ROOT / "experiments/pointmaze_native_rgb_bridge_20261102.py"
PROTOCOL = ROOT.parent / "docs/PC_CAUSAL_HISTORY_RGB_TICK_TIMING_SIDE_PROTOCOL_20261103.md"
ROOTS = (8343, 8345, 8349, 8351)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(request, xml, output):
    import mujoco_py

    with np.load(request, allow_pickle=False) as f:
        starts = f["starts"]
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    cam = sim.render_contexts[0].cam
    cam.azimuth, cam.elevation = 90, -90
    cam.lookat[0], cam.lookat[1] = 3., 3.
    target = model.site_name2id("target_site")
    body = model.body_name2id("particle")
    geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body]
    if len(geoms) != 1:
        raise ValueError("expected a single particle geom")
    geom = geoms[0]
    all_images, all_ticks, all_boundary = [], [], []

    def state():
        return np.concatenate((sim.data.qpos.copy(), sim.data.qvel.copy(),
                               sim.data.geom_xpos[geom].copy(), [sim.data.time]))

    for initial in starts:
        sim.reset()
        sim.data.qpos[:], sim.data.qvel[:] = initial[:2], initial[2:]
        sim.forward()
        sim.data.site_xpos[target] = [11, 11, 0]
        images = [np.ascontiguousarray(sim.render(224, 224), dtype=np.uint8)]
        boundaries, ticks = [state()], []
        for _ in range(2):
            sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
            sim.forward()
            sim.data.ctrl[:] = [0., 0.]
            for _ in range(5):
                sim.step()
                ticks.append(state())
            sim.data.site_xpos[target] = [11, 11, 0]
            images.append(np.ascontiguousarray(sim.render(224, 224), dtype=np.uint8))
            boundaries.append(state())
        all_images.append(images)
        all_ticks.append(ticks)
        all_boundary.append(boundaries)
    np.savez_compressed(output, images=np.asarray(all_images), ticks=np.asarray(all_ticks),
                        boundaries=np.asarray(all_boundary),
                        timestep=np.asarray(model.opt.timestep), body_offset=model.body_pos[body].copy(),
                        geom_offset=model.geom_pos[geom].copy())


def main():
    if OUT.with_suffix(".json").exists() or OUT.with_suffix(".npz").exists():
        raise FileExistsError("refusing overwrite")
    import torch
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    from experiments.a0_pointmaze_rgb_geometry_state_20261103 import estimate_state
    from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import causal_state

    source = json.loads(SOURCE.read_text())
    if sha(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]:
        raise ValueError("source arrays hash mismatch")
    manifest = json.loads((STATES.parent / "challenge_manifest.json").read_text())
    if sha(STATES) != manifest["states_sha256"]:
        raise ValueError("states hash mismatch")
    camera, old = json.loads(CAMERA.read_text()), json.loads(OLD.read_text())
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise ValueError("XML mismatch")
    states = torch.load(STATES, map_location="cpu", weights_only=True)
    starts = np.stack([states[root, 40].numpy().astype(np.float64) for root in ROOTS])
    with tempfile.TemporaryDirectory(prefix="rgb_tick_side_") as tmp:
        tmp = Path(tmp)
        np.savez(tmp / "request.npz", starts=starts)
        (tmp / "scene.xml").write_text(xml)
        runtime = ROOT / ".legacy_mujoco210_py310/vendor/mujoco-2.1.2"
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", MUJOCO_PY_MUJOCO_PATH=str(runtime),
                   LD_LIBRARY_PATH=f"{runtime}/lib:{runtime}/bin:/usr/lib/nvidia:" + env.get("LD_LIBRARY_PATH", ""))
        subprocess.run([str(ROOT / ".legacy_mujoco210_py310/bin/python"), str(Path(__file__)),
                        "--worker", str(tmp / "request.npz"), str(tmp / "scene.xml"),
                        str(tmp / "output.npz")], env=env, check=True)
        with np.load(tmp / "output.npz", allow_pickle=False) as f:
            data = {k: f[k].copy() for k in f.files}
    if not np.isclose(float(data["timestep"]), .01, rtol=0, atol=1e-12):
        raise ValueError("unexpected tick duration")
    if not np.allclose(data["body_offset"], old["body_offset_from_static_scene"], atol=1e-12, rtol=0):
        raise ValueError("body offset mismatch")
    rows = []
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as saved:
        for i, root in enumerate(ROOTS):
            images, ticks, bounds = data["images"][i], data["ticks"][i], data["boundaries"][i]
            original = saved[f"r{root}_p0_history"][:3]
            matches = [bool(np.array_equal(a, b)) for a, b in zip(images, original)]
            actions = saved[f"r{root}_p0_actions"][:2]
            if not np.array_equal(actions, np.zeros((2, 2), np.float32)):
                raise ValueError("source prelude not zero")
            estimates, centers = [], []
            for image in images:
                estimate, meta = estimate_state(image, camera["metadata"], data["body_offset"])
                estimates.append(estimate[:2].astype(np.float64))
                centers.append(meta["marker_xy"])
            intervals = []
            for j in range(2):
                velocity = causal_state(images[j], images[j+1], camera["metadata"],
                                        data["body_offset"], .05)[2:].astype(np.float64)
                intervals.append({"rgb_velocity": velocity.tolist(),
                                  "qpos_interval_velocity": ((bounds[j+1,:2]-bounds[j,:2])/.05).tolist(),
                                  "endpoint_qvel": bounds[j+1,2:4].tolist(),
                                  "rgb_minus_qpos_velocity": (velocity-(bounds[j+1,:2]-bounds[j,:2])/.05).tolist()})
            rows.append({"root": root, "source_image_matches": matches, "actions": actions.tolist(),
                         "centroids_xy_pixels": centers, "rgb_positions": np.asarray(estimates).tolist(),
                         "boundary_states_qpos_qvel_geomxyz_time": bounds.tolist(),
                         "rgb_minus_boundary_qpos": (np.asarray(estimates)-bounds[:,:2]).tolist(),
                         "rgb_minus_render_geom_xy": (np.asarray(estimates)-
                             (bounds[:,4:6]-data["body_offset"][:2])).tolist(),
                         "intervals": intervals,
                         "tick_states_qpos_qvel_geomxyz_time": ticks.tolist()})
    with OUT.with_suffix(".npz").open("xb") as f:
        np.savez_compressed(f, **data, starts=starts)
    report = {"scope": "exposed roots; offline diagnostic; never planning input",
              "roots": rows, "states_sha256": sha(STATES), "source_sha256": sha(SOURCE),
              "source_npz_sha256": sha(SOURCE.with_suffix(".npz")), "camera_sha256": sha(CAMERA),
              "offset_source_sha256": sha(OLD), "bridge_sha256": sha(BRIDGE),
              "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
              "arrays_sha256": sha(OUT.with_suffix(".npz")), "xml_sha256": camera["xml_sha256"],
              "tick_seconds": float(data["timestep"]), "body_offset": data["body_offset"].tolist(),
              "geom_offset": data["geom_offset"].tolist(),
              "all_source_images_match": all(all(r["source_image_matches"]) for r in rows)}
    with OUT.with_suffix(".json").open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(OUT.with_suffix(".json")),
                      "all_source_images_match": report["all_source_images_match"]}))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        main()
