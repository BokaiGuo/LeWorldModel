"""Native-camera control with separately replayed physical trajectories."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_real_camera_effect_dev_20261106_codex"
PROTOCOL = ROOT.parent / "docs/PC_REAL_CAMERA_EFFECT_DEV_PROTOCOL_20261106_CODEX.md"
CAMERAS = ((3., 2.5), (3.5, 2.5))
ACTIONS = ((0., 0.), (.1, 0.), (-.1, 0.))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backend(request, xml, response):
    import mujoco_py
    states = np.load(request, allow_pickle=False)["states"]
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    cam = sim.render_contexts[0].cam
    cam.azimuth, cam.elevation = 90, -90
    camera_default = np.r_[cam.azimuth, cam.elevation, cam.distance, cam.lookat].copy()
    traces, images, contacts, times = [], [], [], []
    for state in states:
        state_traces, state_images, state_contacts, state_times = [], [], [], []
        for action in ACTIONS:
            action_traces, action_images, action_contacts, action_times = [], [], [], []
            for lookat in CAMERAS:
                sim.reset()
                sim.data.qpos[:] = state[:2]
                sim.data.qvel[:] = state[2:]
                sim.forward()
                cam.lookat[0], cam.lookat[1] = lookat
                trace, frames, counts, clock = [], [], [], []
                for step in range(6):
                    before = np.r_[sim.data.qpos, sim.data.qvel, sim.data.time].copy()
                    sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
                    frames.append(sim.render(224, 224).copy())
                    np.testing.assert_array_equal(
                        before, np.r_[sim.data.qpos, sim.data.qvel, sim.data.time])
                    trace.append(before[:4])
                    counts.append(int(sim.data.ncon))
                    clock.append(float(sim.data.time))
                    if step < 5:
                        sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
                        sim.forward()
                        sim.data.ctrl[:] = action
                        for _ in range(5):
                            sim.step()
                action_traces.append(trace)
                action_images.append(frames)
                action_contacts.append(counts)
                action_times.append(clock)
            state_traces.append(action_traces)
            state_images.append(action_images)
            state_contacts.append(action_contacts)
            state_times.append(action_times)
        traces.append(state_traces)
        images.append(state_images)
        contacts.append(state_contacts)
        times.append(state_times)
    np.savez_compressed(response, traces=traces, images=images, contacts=contacts,
                        times=times, camera_default=camera_default,
                        binding=np.asarray(mujoco_py.__version__),
                        runtime=np.asarray(mujoco_py.get_version()))


def main():
    import torch
    from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    report_path, payload_path = OUT.with_suffix(".json"), OUT.with_suffix(".npz")
    if report_path.exists() or payload_path.exists():
        raise FileExistsError(OUT)
    ids = [(root, frame) for root in range(1880, 1884) for frame in (0, 5)]
    source = DATA / "states.pth"
    archive = torch.load(source, map_location="cpu", weights_only=True)
    states = np.stack([archive[root, frame].numpy() for root, frame in ids])
    print("Native camera diagnostic: fixed exposed roots; physics must be view invariant.", flush=True)
    with tempfile.TemporaryDirectory(prefix="pc_real_camera_") as temp:
        folder = Path(temp)
        request, xml = folder / "request.npz", folder / "scene.xml"
        np.savez_compressed(request, states=states)
        xml_text = official_xml()
        xml.write_text(xml_text)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                          "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(LEGACY / "bin/python"), str(Path(__file__).resolve()), "--backend",
                        str(request), str(xml), str(payload_path)], env=env, check=True, timeout=180)
    with np.load(payload_path, allow_pickle=False) as data:
        traces, images, times = data["traces"], data["images"], data["times"]
        if traces.shape != (8, 3, 2, 6, 4) or images.shape != (8, 3, 2, 6, 224, 224, 3):
            raise ValueError("unexpected tensor shapes")
        np.testing.assert_allclose(traces[:, :, 0], traces[:, :, 1], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(times[:, :, 0], times[:, :, 1])
        delta = np.abs(images[:, :, 0].astype(float) - images[:, :, 1])
        rows = [{"root": root, "frame": frame,
                 "camera_physics_max_difference": float(np.abs(traces[i, :, 0] - traces[i, :, 1]).max()),
                 "camera_pixel_mae": float(delta[i].mean()),
                 "final_state_by_action": traces[i, :, 0, -1].tolist()}
                for i, (root, frame) in enumerate(ids)]
        if not all(row["camera_pixel_mae"] > 0 for row in rows):
            raise ValueError("camera did not change image")
        report = {"scope": "exposed-root native camera physics control, not model or search evidence",
                  "roots": 4, "states": 8, "trajectories": 48, "rows": rows,
                  "actions": ACTIONS, "cameras": CAMERAS, "camera_default": data["camera_default"].tolist(),
                  "binding": str(data["binding"]), "runtime": str(data["runtime"]),
                  "source_sha256": sha(source), "code_sha256": sha(__file__),
                  "protocol_sha256": sha(PROTOCOL), "payload_sha256": sha(payload_path),
                  "xml_sha256": hashlib.sha256(xml_text.encode()).hexdigest()}
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--backend":
        backend(*sys.argv[2:])
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("unexpected arguments")
