"""Validate the development-selected fixed camera on other exposed roots."""
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
OUTPUT = ROOT / "results/a0_pointmaze_camera_validation_20261102_DEV01.json"
ROOTS, FRAMES = (1881, 1882, 1883), (0, 5, 10)


def backend(request: Path, xml: Path, response: Path) -> None:
    import mujoco_py
    with np.load(request, allow_pickle=False) as payload:
        states = payload["states"]
    model = mujoco_py.load_model_from_xml(xml.read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    camera = sim.render_contexts[0].cam
    camera.azimuth, camera.elevation = 90, -90
    camera.lookat[0], camera.lookat[1] = 3.0, 3.0
    images = []
    for state in states:
        sim.data.qpos[:], sim.data.qvel[:] = state[:2], state[2:]
        sim.forward()
        sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
        images.append(sim.render(224, 224).copy())
    np.savez_compressed(response, images=np.stack(images))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import torch
    from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_camera_offset_probe_20261102 import OUTPUT as calibration

    expected = "1c0fff1cc2e2ea3feee1e0cc1e798c2baa2e969230d9563fc0b5dd737bde0e06"
    if OUTPUT.exists() or sha(calibration) != expected:
        raise RuntimeError("output exists or development camera source changed")
    state_path = DATA / "states.pth"
    states = torch.load(state_path, map_location="cpu", weights_only=True).numpy()
    refs, starts, identities, episode_hashes = [], [], [], {}
    for root in ROOTS:
        path = DATA / "obses" / f"episode_{root:03d}.pth"
        images = torch.load(path, map_location="cpu", weights_only=True).numpy()
        episode_hashes[str(root)] = sha(path)
        for frame in FRAMES:
            identities.append((root, frame))
            refs.append(images[frame])
            starts.append(states[root, frame])
    with tempfile.TemporaryDirectory(prefix="pc_pointmaze_camera_validate_") as tmp:
        folder = Path(tmp)
        request, xml, response = folder / "states.npz", folder / "maze.xml", folder / "output.npz"
        np.savez_compressed(request, states=np.stack(starts))
        xml_text = official_xml()
        xml.write_text(xml_text)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        proc = subprocess.run([str(LEGACY / "bin/python"), str(Path(__file__).resolve()), "--backend",
                               str(request), str(xml), str(response)], capture_output=True, text=True,
                              env=env, timeout=90)
        if proc.returncode or not response.is_file():
            raise RuntimeError(proc.stderr[-3000:])
        with np.load(response, allow_pickle=False) as payload:
            rendered = payload["images"].copy()
    if rendered.shape != (len(identities), 224, 224, 3):
        raise RuntimeError("render output shape mismatch")
    rows = []
    for (root, frame), actual, ref in zip(identities, rendered, refs):
        difference = np.abs(actual.astype(np.int16) - ref.astype(np.int16))
        rows.append({"root": root, "frame": frame, "mae": float(difference.mean()),
                     "different_pixels": int(np.any(difference != 0, axis=2).sum()),
                     "render_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                     "archive_sha256": hashlib.sha256(ref.tobytes()).hexdigest()})
    report = {"schema": "a0_pointmaze_camera_validation.v1",
              "scope": "development-selected camera; other exposed roots; no planner or independent P1",
              "selection": "lookat 3,3 selected post hoc on root 1880 frame 0",
              "calibration_sha256": expected, "roots": list(ROOTS), "frames": list(FRAMES),
              "code_sha256": sha(Path(__file__)), "states_sha256": sha(state_path),
              "episode_sha256": episode_hashes,
              "xml_sha256": hashlib.sha256(xml_text.encode()).hexdigest(),
              "legacy_runtime_sha256": sha(RUNTIME / "lib/libmujoco.so.2.1.2"),
              "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest(),
                      "max_mae": max(row["mae"] for row in rows),
                      "exact": sum(row["different_pixels"] == 0 for row in rows)}))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--backend":
        backend(*(Path(arg) for arg in sys.argv[2:]))
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("unexpected arguments")
