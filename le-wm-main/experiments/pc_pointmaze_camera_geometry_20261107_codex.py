"""Development-only camera grid audit on exposed archive initial images."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from experiments import pc_pointmaze_packetwise_three_way_20261107_codex as prior
from experiments.pc_input_permissions import file_hash

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CAMERA_GEOMETRY_CALIBRATION_PROTOCOL_20261107_CODEX.md"
WORKER = ROOT / "experiments/pc_pointmaze_camera_grid_worker_20261107_codex.py"
BASE = ROOT / "results/pc_pointmaze_camera_geometry_20261107_codex"
FRAMES = Path(str(BASE) + "_frames.npz")
RESULT = Path(str(BASE) + "_result.json")
FOLDER = ROOT / "data/point_maze_tail1800_1899"


def camera_grid():
    return np.asarray([(x, y) for x in (2.5, 2.75, 3., 3.25, 3.5)
                       for y in (2., 2.25, 2.5, 2.75, 3.)], dtype=np.float64)


def errors(frames, reference):
    frames, reference = np.asarray(frames), np.asarray(reference)
    if frames.shape != (8, 25, 224, 224, 3) or reference.shape != (8, 224, 224, 3):
        raise ValueError("unexpected image shapes")
    return np.abs(frames.astype(np.int16) - reference[:, None].astype(np.int16)).mean(axis=(2, 3, 4))


def main():
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    if FRAMES.exists() or RESULT.exists():
        raise FileExistsError("camera results exist")
    print("Compare fixed camera grid against archived initial frames; no outcome-guided actions.", flush=True)
    prior.check_plan()
    render = json.loads(prior.RENDER.read_text())
    if render["frames"] != file_hash(prior.FRAMES):
        raise ValueError("prior render drift")
    states_path = FOLDER / "states.pth"
    states = torch.load(states_path, map_location="cpu", weights_only=True)[1884:1892, 0].numpy()
    paths = [FOLDER / "obses" / f"episode_{r:03d}.pth" for r in range(1884, 1892)]
    reference = np.stack([torch.load(p, map_location="cpu", weights_only=True)[0].numpy() for p in paths])
    cameras = camera_grid()
    with tempfile.TemporaryDirectory(prefix="pc_camera_grid_") as tmp:
        folder = Path(tmp)
        request, xml, output = folder / "request.npz", folder / "scene.xml", folder / "output.npz"
        with request.open("xb") as f:
            np.savez_compressed(f, states=states, cameras=cameras)
        xml_text = official_xml()
        xml.write_text(xml_text)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(LEGACY / "bin/python"), str(WORKER), str(request), str(xml), str(output)],
                       env=env, check=True, timeout=180)
        with np.load(output, allow_pickle=False) as f:
            frames, defaults = f["frames"].copy(), f["defaults"].copy()
    fixed = int(np.flatnonzero(np.all(cameras == [3., 2.5], axis=1))[0])
    with np.load(prior.FRAMES, allow_pickle=False) as previous:
        for i in range(8):
            np.testing.assert_array_equal(frames[i, fixed], previous[f"r{1884+i}"][0, 0])
    mae = errors(frames, reference)
    choice = int(np.argmin(mae[:4].mean(axis=0)))
    with FRAMES.open("xb") as f:
        np.savez_compressed(f, states=states, cameras=cameras, frames=frames, reference=reference, mae=mae)
    record = {"scope": "exposed state-assisted camera geometry only", "protocol": file_hash(PROTOCOL),
              "runner": file_hash(__file__), "worker": file_hash(WORKER), "frames": file_hash(FRAMES),
              "source_frames": file_hash(prior.FRAMES), "states": file_hash(states_path),
              "archive_frames": {str(1884+i): file_hash(p) for i, p in enumerate(paths)},
              "xml_sha256": hashlib.sha256(xml_text.encode()).hexdigest(), "defaults": defaults.tolist(),
              "fixed_index": fixed, "selected_index": choice, "selected_camera": cameras[choice].tolist(),
              "calibration_roots": [1884, 1885, 1886, 1887], "check_roots": [1888, 1889, 1890, 1891],
              "fixed_mae_per_root": mae[:, fixed].tolist(), "selected_mae_per_root": mae[:, choice].tolist(),
              "mean_mae_per_camera_calibration": mae[:4].mean(axis=0).tolist(),
              "mean_mae_per_camera_check": mae[4:].mean(axis=0).tolist()}
    with RESULT.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)
    print(json.dumps({k: record[k] for k in ("defaults", "selected_camera", "fixed_mae_per_root",
                                             "selected_mae_per_root")}, indent=2))


if __name__ == "__main__":
    main()
