"""Exploratory fixed camera-offset grid on one already exposed archive frame."""
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
OUTPUT = ROOT / "results/a0_pointmaze_camera_offset_probe_20261102_DEV01.json"
CAMERAS = ((3.0, 2.5), (2.5, 2.5), (3.5, 2.5), (3.0, 2.0), (3.0, 3.0))


def backend(request: Path, xml: Path, response: Path) -> None:
    import mujoco_py

    state = np.load(request, allow_pickle=False)["state"]
    model = mujoco_py.load_model_from_xml(xml.read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    camera = sim.render_contexts[0].cam
    camera.azimuth, camera.elevation = 90, -90
    sim.data.qpos[:] = state[:2]
    sim.data.qvel[:] = state[2:]
    sim.forward()
    sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
    images = []
    for x, y in CAMERAS:
        camera.lookat[0], camera.lookat[1] = x, y
        images.append(sim.render(224, 224).copy())
    np.savez_compressed(response, images=np.stack(images))


def main() -> None:
    import torch
    from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    state_path, image_path = DATA / "states.pth", DATA / "obses/episode_1880.pth"
    state = torch.load(state_path, map_location="cpu", weights_only=True)[1880, 0].numpy()
    reference = torch.load(image_path, map_location="cpu", weights_only=True)[0].numpy()
    with tempfile.TemporaryDirectory(prefix="pc_pointmaze_camera_offset_") as temp:
        folder = Path(temp)
        request, xml, response = folder / "state.npz", folder / "scene.xml", folder / "out.npz"
        np.savez_compressed(request, state=state)
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
            images = payload["images"].copy()
    if images.shape != (len(CAMERAS), 224, 224, 3):
        raise RuntimeError("bad image tensor")
    rows = []
    for camera, image in zip(CAMERAS, images):
        diff = np.abs(image.astype(np.int16) - reference.astype(np.int16))
        rows.append({"lookat_xy": list(camera), "mae": float(diff.mean()),
                     "different_pixels": float(np.any(diff != 0, axis=2).mean()),
                     "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
    report = {"schema": "a0_pointmaze_camera_offset_probe.v1",
              "scope": "post-hoc exploratory grid on exposed root 1880 frame 0; not a camera certification",
              "camera_grid": [list(value) for value in CAMERAS],
              "state_archive_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
              "image_archive_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
              "xml_sha256": hashlib.sha256(xml_text.encode()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest(), "rows": rows}))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--backend":
        backend(*(Path(arg) for arg in sys.argv[2:]))
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("unexpected arguments")
