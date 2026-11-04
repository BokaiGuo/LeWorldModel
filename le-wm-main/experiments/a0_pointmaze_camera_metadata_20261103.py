"""Capture fixed legacy-renderer geometry, without dynamic state labels."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/a0_pointmaze_camera_metadata_20261103_DEV01.json"
LEGACY = ROOT / ".legacy_mujoco210_py310"


def worker(path):
    import mujoco_py
    model = mujoco_py.load_model_from_xml(Path(path).read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    camera = sim.render_contexts[0].cam
    camera.azimuth, camera.elevation = 90, -90
    camera.lookat[0], camera.lookat[1] = 3.0, 3.0
    marker = model.geom_name2id("particle_geom")
    print(json.dumps({"camera_distance": float(camera.distance), "camera_lookat": camera.lookat.tolist(),
                      "azimuth": float(camera.azimuth), "elevation": float(camera.elevation),
                      "fovy_degrees": float(model.vis.global_.fovy),
                      "marker_geom_relative_position": model.geom_pos[marker].tolist(),
                      "marker_geom_radius": float(model.geom_size[marker][0]),
                      "mujoco_py_version": mujoco_py.__version__}, allow_nan=False))


def main():
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    xml = official_xml()
    with tempfile.TemporaryDirectory(prefix="pc_cam_metadata_") as tmp:
        xml_path = Path(tmp) / "scene.xml"
        xml_path.write_text(xml)
        env = os.environ.copy()
        runtime = LEGACY / "vendor/mujoco-2.1.2"
        env["MUJOCO_PY_MUJOCO_PATH"] = str(runtime)
        env["LD_LIBRARY_PATH"] = ":".join((str(runtime / "lib"), str(runtime / "bin"), "/usr/lib/nvidia",
                                           env.get("LD_LIBRARY_PATH", "")))
        proc = subprocess.run([str(LEGACY / "bin/python"), str(Path(__file__).resolve()), "--worker", str(xml_path)],
                              env=env, text=True, capture_output=True, check=False)
        if proc.returncode:
            raise RuntimeError(proc.stderr[-1500:])
        records = [line for line in proc.stdout.splitlines() if line.startswith("{")]
        if len(records) != 1:
            raise RuntimeError("legacy camera worker returned an ambiguous metadata record")
        metadata = json.loads(records[0])
    report = {"schema": "a0_pointmaze_camera_metadata.v1", "metadata": metadata,
              "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(), "script_sha256": sha256_file(Path(__file__)),
              "scope": "static bridge camera/geometry; camera xy was development-selected"}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"metadata": metadata, "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
