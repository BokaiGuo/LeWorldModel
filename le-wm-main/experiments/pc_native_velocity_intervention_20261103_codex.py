"""Launch sealed native fixed-bank velocity intervention."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_geometry_native_suffix_20261103_codex as source
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base

WORKER = base.ROOT / "experiments/pc_native_velocity_intervention_worker_20261103_codex.py"
PROTOCOL = base.PROJECT / "docs/PC_NATIVE_VELOCITY_INTERVENTION_PROTOCOL_20261103_CODEX.md"
OUTPUT = base.ROOT / "results/pc_native_velocity_intervention_20261103_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    previous = json.loads(source.OUTPUT.read_text())
    if previous["worker_sha256"] != base.digest(source.WORKER):
        raise ValueError("source worker changed")
    with tempfile.TemporaryDirectory(prefix="pc_velocity_") as temp:
        path = Path(temp)
        xml, result = path / "scene.xml", path / "result.json"
        xml.write_text(official_xml())
        if base.digest(xml) != previous["xml_sha256"]:
            raise ValueError("XML mismatch")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"),
                                            str(backend.RUNTIME / "bin"),
                                            "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY / "bin/python"), str(WORKER), str(source.OUTPUT),
                        str(xml), str(result)], env=env, check=True, timeout=120)
        data = json.loads(result.read_text())
    errors = [abs(e) for row in data["rows"] for e in row["reconstruction_difference"]]
    data.update({"schema": "pc_native_velocity_intervention.v1",
                 "source_sha256": base.digest(source.OUTPUT), "protocol_sha256": base.digest(PROTOCOL),
                 "worker_sha256": base.digest(WORKER), "launcher_sha256": base.digest(Path(__file__)),
                 "max_reconstruction_distance_error": max(errors),
                 "reconstruction_tolerance": 1e-8,
                 "reconstruction_pass": max(errors) <= 1e-8})
    with OUTPUT.open("x") as handle:
        json.dump(data, handle, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: data[key] for key in ("max_reconstruction_distance_error",
                                                 "reconstruction_pass")}))


if __name__ == "__main__":
    main()
