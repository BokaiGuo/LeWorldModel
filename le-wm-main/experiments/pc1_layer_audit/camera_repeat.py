"""Read-only diagnostic of render-repeat parity; no changing audit settings."""
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from experiments.pc1_layer_audit import camera
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
from experiments.pc_input_permissions import file_hash
from experiments.research_runtime.artifacts import atomic_write_json


def main():
    roots = (1832, 1884)
    out = camera.OUT / "render_repeat"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise FileExistsError(out)
    archive = torch.load(camera.DATA / "states.pth", weights_only=True, map_location="cpu")
    states = np.stack([archive[r, [0, 5]].numpy() for r in roots]).reshape(-1, 4)
    worker = Path(__file__).with_name("camera_repeat_worker.py")
    with tempfile.TemporaryDirectory(prefix="pc1_render_repeat_") as tmp:
        folder = Path(tmp)
        request, xml = folder / "input.npz", folder / "scene.xml"
        with request.open("xb") as f:
            np.savez_compressed(f, states=states)
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(LEGACY / "bin/python"), str(worker), str(request), str(xml), str(out / "frames.npz")],
                       env=env, check=True, timeout=90)
    with np.load(out / "frames.npz", allow_pickle=False) as f:
        frames = f["frames"].reshape(2, 2, 6, 224, 224, 3)
    rows = []
    for i, root in enumerate(roots):
        ref = torch.load(camera.DATA / "obses" / f"episode_{root:03d}.pth", weights_only=True, map_location="cpu")[[0, 5]].numpy()
        delta = np.abs(frames[i].astype(float)-ref[:, None].astype(float))
        rows.append(dict(root=root, mae=delta.mean((2, 3, 4)).tolist(),
                         max_error=delta.max((2, 3, 4)).tolist()))
    record = dict(rows=rows, sources={str(p): file_hash(p) for p in
                                     (Path(__file__), worker, camera.PROTOCOL)},
                  arrays_sha256=file_hash(out / "frames.npz"))
    atomic_write_json(out / "result.json", record)
    print(rows, flush=True)


if __name__ == "__main__":
    main()
