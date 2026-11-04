"""Compare archived frames to legacy step and synchronized render timing."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/a0_pointmaze_replay_render_timing_20261102_DEV01.json"
ROOTS, FRAMES = (1880, 1881, 1882, 1883), (0, 5, 10)


def backend(request: Path, xml: Path, response: Path) -> None:
    import mujoco_py

    with np.load(request, allow_pickle=False) as payload:
        states, actions = payload["states"], payload["actions"]
    model = mujoco_py.load_model_from_xml(xml.read_text())
    raw_images, synced_images, replay_states = [], [], []
    for source, action_sequence in zip(states, actions):
        sim = mujoco_py.MjSim(model)
        sim.render(224, 224)
        cam = sim.render_contexts[0].cam
        cam.azimuth, cam.elevation = 90, -90
        cam.lookat[0], cam.lookat[1] = 3.0, 3.0
        sim.data.qpos[:], sim.data.qvel[:] = source[:2], source[2:]
        sim.forward()
        sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
        raw_root, synced_root, states_root = [], [], []
        for step in range(11):
            if step in FRAMES:
                # MazeEnv._render_frame does not explicitly forward after step.
                raw_root.append(sim.render(224, 224).copy())
                states_root.append(np.r_[sim.data.qpos[:2], sim.data.qvel[:2]].copy())
                sim.forward()
                sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
                synced_root.append(sim.render(224, 224).copy())
            if step < 10:
                sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
                sim.forward()
                sim.data.ctrl[:] = np.clip(action_sequence[step], -1, 1)
                for _ in range(5):
                    sim.step()
                sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
        raw_images.append(raw_root)
        synced_images.append(synced_root)
        replay_states.append(states_root)
    np.savez_compressed(response, raw=np.asarray(raw_images), synced=np.asarray(synced_images),
                        states=np.asarray(replay_states))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import torch
    from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_camera_validation_20261102 import OUTPUT as camera_result

    expected = "89a87203999b382a764b15da399a515199db0ba2ce6a6dcac661ffc20787d2d6"
    if OUTPUT.exists() or sha(camera_result) != expected:
        raise RuntimeError("output exists or camera result changed")
    states_path, actions_path = DATA / "states.pth", DATA / "actions.pth"
    states = torch.load(states_path, map_location="cpu", weights_only=True).numpy()
    actions = torch.load(actions_path, map_location="cpu", weights_only=True).numpy()
    archived = []
    episode_hashes = {}
    for root in ROOTS:
        path = DATA / "obses" / f"episode_{root:03d}.pth"
        episode_hashes[str(root)] = sha(path)
        archived.append(torch.load(path, map_location="cpu", weights_only=True).numpy()[list(FRAMES)])
    archived = np.stack(archived)
    with tempfile.TemporaryDirectory(prefix="pc_pointmaze_replay_render_") as tmp:
        directory = Path(tmp)
        request, xml, response = directory / "request.npz", directory / "maze.xml", directory / "result.npz"
        np.savez_compressed(request, states=states[list(ROOTS), 0], actions=actions[list(ROOTS), :10])
        xml_text = official_xml()
        xml.write_text(xml_text)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        proc = subprocess.run([str(LEGACY / "bin/python"), str(Path(__file__).resolve()), "--backend",
                               str(request), str(xml), str(response)], env=env, text=True,
                              capture_output=True, timeout=90)
        if proc.returncode or not response.is_file():
            raise RuntimeError(proc.stderr[-3000:])
        with np.load(response, allow_pickle=False) as payload:
            raw, synced, replay = (payload[key].copy() for key in ("raw", "synced", "states"))
    if raw.shape != archived.shape or synced.shape != archived.shape or replay.shape != (len(ROOTS), len(FRAMES), 4):
        raise RuntimeError("backend returned malformed arrays")
    rows = []
    for i, root in enumerate(ROOTS):
        for j, frame in enumerate(FRAMES):
            result = {"root": root, "frame": frame}
            for name, image in (("raw", raw[i, j]), ("synced", synced[i, j])):
                diff = np.abs(image.astype(np.int16) - archived[i, j].astype(np.int16))
                result[name] = {"mae": float(diff.mean()),
                                "different_pixels": int(np.any(diff != 0, axis=2).sum()),
                                "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
            state_error = replay[i, j].astype(np.float64) - states[root, frame].astype(np.float64)
            result["position_error_l2"] = float(np.linalg.norm(state_error[:2]))
            result["velocity_error_l2"] = float(np.linalg.norm(state_error[2:]))
            result["archive_rgb_sha256"] = hashlib.sha256(archived[i, j].tobytes()).hexdigest()
            rows.append(result)
    report = {
        "schema": "a0_pointmaze_replay_render_timing.v1",
        "scope": "exposed development roots; replay diagnostic, not P1 or independent confirmation",
        "camera_selection_sha256": expected,
        "contract": "stored action clipped; velocity clipped before each archive action; five mujoco ticks per action",
        "raw": "render immediately after step; no explicit forward",
        "synced": "render after sim.forward at same state; evaluation intervention",
        "states_sha256": sha(states_path), "actions_sha256": sha(actions_path),
        "episode_sha256": episode_hashes, "script_sha256": sha(Path(__file__)),
        "runtime_sha256": sha(RUNTIME / "lib/libmujoco.so.2.1.2"),
        "xml_sha256": hashlib.sha256(xml_text.encode()).hexdigest(),
        "roots": list(ROOTS), "frames": list(FRAMES), "rows": rows,
    }
    raw_json = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw_json)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw_json).hexdigest(),
                      "raw_exact": sum(row["raw"]["different_pixels"] == 0 for row in rows),
                      "synced_exact": sum(row["synced"]["different_pixels"] == 0 for row in rows),
                      "max_position_error": max(row["position_error_l2"] for row in rows)}))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--backend":
        backend(*(Path(arg) for arg in sys.argv[2:]))
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("unexpected arguments")
