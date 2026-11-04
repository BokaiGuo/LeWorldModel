"""Localize legacy/archive pixel differences on a single exposed development root."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
import torch

from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/a0_pointmaze_visual_difference_20261102_DEV01.json"
MONTAGE = ROOT / "results/a0_pointmaze_visual_difference_20261102_DEV01.png"
FRAMES = (0, 5, 10)
EPISODE = 1880


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if RESULT.exists() or MONTAGE.exists():
        raise FileExistsError("refusing to overwrite diagnostic outputs")
    episode_path = DATA / "obses" / f"episode_{EPISODE:03d}.pth"
    images = torch.load(episode_path, map_location="cpu", weights_only=True).numpy()[list(FRAMES)]
    states = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True).numpy()[EPISODE, list(FRAMES)]
    with tempfile.TemporaryDirectory(prefix="pc_pointmaze_visual_diff_") as tmp:
        folder = Path(tmp)
        request, xml, response = folder / "state.npz", folder / "maze.xml", folder / "rgb.npz"
        np.savez_compressed(request, states=states)
        xml_text = official_xml()
        xml.write_text(xml_text)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        backend = ROOT / "experiments/a0_pointmaze_legacy_render_probe_20261102.py"
        command = [str(LEGACY / "bin/python"), str(backend), "--backend",
                   str(request), str(xml), str(response)]
        process = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
        if process.returncode or not response.is_file():
            raise RuntimeError(process.stderr[-3000:])
        with np.load(response, allow_pickle=False) as data:
            rendered = data["images"].copy()
    if rendered.shape != images.shape or images.shape != (3, 224, 224, 3):
        raise ValueError("unexpected image dimensions")
    # A per-pixel variation mask is derived from the three frames themselves.
    # It localizes all changed pixels, not specifically the point marker.
    archive_motion = np.any(images != images[0:1], axis=(0, 3))
    render_motion = np.any(rendered != rendered[0:1], axis=(0, 3))
    motion_union = archive_motion | render_motion
    difference = np.abs(images.astype(np.int16) - rendered.astype(np.int16))
    background = ~motion_union
    panels = [np.concatenate((images[i], rendered[i], difference[i].astype(np.uint8)), axis=1)
              for i in range(len(FRAMES))]
    Image.fromarray(np.concatenate(panels, axis=0)).save(MONTAGE)
    report = {
        "schema": "a0_pointmaze_visual_difference.v1",
        "scope": "exploratory fixed development root; no thresholds, model, or planning outcome",
        "episode": EPISODE, "frames": list(FRAMES),
        "image_archive_sha256": sha(episode_path),
        "state_archive_sha256": sha(DATA / "states.pth"),
        "backend_sha256": sha(backend),
        "official_xml_sha256": hashlib.sha256(xml_text.encode()).hexdigest(),
        "script_sha256": sha(Path(__file__)),
        "montage_sha256": sha(MONTAGE),
        "montage_layout": "rows: frames 0/5/10; columns: archive, legacy, absolute RGB difference",
        "archive_changed_fraction": float(archive_motion.mean()),
        "legacy_changed_fraction": float(render_motion.mean()),
        "motion_union_fraction": float(motion_union.mean()),
        "mae_outside_motion_union": float(difference[:, background, :].mean()),
        "mae_inside_motion_union": float(difference[:, motion_union, :].mean()) if motion_union.any() else None,
    }
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with RESULT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"result": str(RESULT), "sha256": hashlib.sha256(raw).hexdigest(), **report}))


if __name__ == "__main__":
    main()
