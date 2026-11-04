"""Compare original mujoco-py render path with fixed archive frames.

The torch-capable parent delegates rendering to the installed legacy
interpreter. This is a development diagnostic, not a confirmation run.
"""
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
OUTPUT = ROOT / "results/a0_pointmaze_legacy_render_probe_20261102_DEV01.json"
LEGACY = ROOT / ".legacy_mujoco210_py310"
RUNTIME = LEGACY / "vendor/mujoco-2.1.2"
FRAMES = (0, 5, 10)
ROOTS = (1880, 1881, 1882, 1883)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backend(subset: Path, xml_path: Path, output: Path) -> None:
    import mujoco_py

    with np.load(subset, allow_pickle=False) as payload:
        states = payload["states"]
    model = mujoco_py.load_model_from_xml(xml_path.read_text())
    sim = mujoco_py.MjSim(model)
    # Match prepare_for_render: first create a context, then set its angles.
    sim.render(224, 224)
    camera = sim.render_contexts[0].cam
    camera.azimuth = 90
    camera.elevation = -90
    camera_values = np.r_[camera.azimuth, camera.elevation, camera.distance, camera.lookat].copy()
    frames = []
    for state in states:
        sim.data.qpos[:] = state[:2]
        sim.data.qvel[:] = state[2:]
        sim.forward()
        sim.data.site_xpos[model.site_name2id("target_site")] = [11.0, 11.0, 0.0]
        frames.append(sim.render(224, 224).copy())
    np.savez_compressed(output, images=np.stack(frames), camera=camera_values,
                        mujoco_py_version=np.asarray(mujoco_py.__version__),
                        mujoco_runtime=np.asarray(mujoco_py.get_version()))


def main() -> None:
    import torch
    from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    state_source = DATA / "states.pth"
    states = torch.load(state_source, map_location="cpu", weights_only=True).numpy()
    images, selected_states, identifiers, archive_hashes = [], [], [], {}
    for root in ROOTS:
        path = DATA / "obses" / f"episode_{root:03d}.pth"
        archive_hashes[str(root)] = sha(path)
        episode = torch.load(path, map_location="cpu", weights_only=True).numpy()
        for frame in FRAMES:
            identifiers.append((root, frame))
            images.append(episode[frame])
            selected_states.append(states[root, frame])
    with tempfile.TemporaryDirectory(prefix="pc_pointmaze_legacy_render_") as tmp:
        folder = Path(tmp)
        request, xml_path, response = folder / "request.npz", folder / "official.xml", folder / "response.npz"
        np.savez_compressed(request, states=np.stack(selected_states))
        xml = official_xml()
        xml_path.write_text(xml)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        cmd = [str(LEGACY / "bin/python"), str(Path(__file__).resolve()), "--backend",
               str(request), str(xml_path), str(response)]
        completed = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=90)
        if completed.returncode or not response.is_file():
            raise RuntimeError(f"legacy renderer failed ({completed.returncode}): {completed.stderr[-3000:]}")
        with np.load(response, allow_pickle=False) as payload:
            rendered = payload["images"].copy()
            camera = payload["camera"].tolist()
            versions = {"mujoco_py": str(payload["mujoco_py_version"].item()),
                        "runtime": str(payload["mujoco_runtime"].item())}
    if rendered.shape != (len(identifiers), 224, 224, 3):
        raise RuntimeError(f"unexpected rendered shape {rendered.shape}")
    rows = []
    for (root, frame), image, reference in zip(identifiers, rendered, images):
        row = {"root": root, "frame": frame,
               "archive_rgb_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
               "render_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
        for orientation, trial in (("raw", image), ("vertical_flip", image[::-1])):
            diff = np.abs(trial.astype(np.int16) - reference.astype(np.int16))
            row[orientation] = {"mae": float(diff.mean()),
                                "fraction_nonidentical_pixels": float(np.any(diff != 0, axis=2).mean())}
        rows.append(row)
    report = {
        "schema": "a0_pointmaze_legacy_render_probe.v1",
        "scope": "exposed roots; native renderer diagnostic only; neither training independence nor P1 confirmation",
        "versions": versions, "camera_azimuth_elevation_distance_lookat": camera,
        "official_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "legacy_runtime_sha256": sha(RUNTIME / "lib/libmujoco.so.2.1.2"),
        "runner_sha256": sha(Path(__file__)), "states_sha256": sha(state_source),
        "episode_sha256": archive_hashes,
        "orientation_analysis": "raw and vertically flipped separately; exploratory, not selected for deployment",
        "rows": rows,
    }
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest(),
                      "raw_mae_range": [min(r["raw"]["mae"] for r in rows),
                                        max(r["raw"]["mae"] for r in rows)]}))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--backend":
        backend(*(Path(arg) for arg in sys.argv[2:]))
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("usage: runner with no arguments (backend mode is internal)")
