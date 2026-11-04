"""Launch sealed 2.1.2 privileged suffix diagnostic in isolated interpreter."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base

WORKER = base.ROOT / "experiments/pc_geometry_native_suffix_worker_20261103_codex.py"
PROTOCOL = base.PROJECT / "docs/PC_GEOMETRY_NATIVE_FIXED_BANK_SUFFIX_PROTOCOL_20261103_CODEX.md"
OUTPUT = base.ROOT / "results/pc_geometry_native_fixed_bank_suffix_20261103_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    starts = [s.tolist() for s in base.obstacle_intersection_starts()]
    if starts != [[.8, .8], [.8, 1.8], [2.8, .8], [2.8, 1.8]]:
        raise ValueError("static geometry roots changed")
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pc_native_suffix_") as temporary:
        tmp = Path(temporary)
        request, xml, output = tmp / "request.json", tmp / "official.xml", tmp / "result.json"
        request.write_text(json.dumps({"starts": starts}))
        xml.write_text(official_xml())
        xml_hash = base.digest(xml)
        if xml_hash != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("official executor XML changed")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"), str(backend.RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY / "bin/python"), str(WORKER), str(request),
                        str(xml), str(output)], check=True, env=env, timeout=1200)
        result = json.loads(output.read_text())
    if len(result["rows"]) != 48 or any(len(r["search"]["evaluations"]) != 100 for r in result["rows"]):
        raise ValueError("incomplete fixed bank or budget")
    result.update({"schema": "pc_geometry_native_fixed_bank_suffix.v1",
                   "protocol_sha256": base.digest(PROTOCOL), "xml_sha256": xml_hash,
                   "worker_sha256": base.digest(WORKER), "launcher_sha256": base.digest(Path(__file__)),
                   "elapsed_seconds": time.monotonic() - t0})
    with OUTPUT.open("x") as handle:
        json.dump(result, handle, allow_nan=False)
        handle.write("\n")
    print(f"saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
