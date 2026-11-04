"""Post-seal physical replay for equal-token RGB feedback experiment."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_equal_resource_dev_20261103_codex as run
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

OUTPUT = run.PREFIX.with_name(run.PREFIX.name+"_physical.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    root, cp5, execution = (json.loads(p.read_text()) for p in (run.ROOT_PLAN, run.CP_PLAN, run.EXEC))
    if (root["runner_sha256"] != sha(Path(run.__file__)) or root["protocol_sha256"] != sha(run.PROTOCOL)
            or cp5["root_plan_sha256"] != sha(run.ROOT_PLAN) or cp5["context_sha256"] != sha(run.CONTEXT)
            or execution["cp5_sha256"] != sha(run.CP_PLAN) or execution["images_sha256"] != sha(run.IMAGES)):
        raise ValueError("sealed dependency mismatch")
    requests = [{"start": list(run.old.STARTS[row["root"]]), "policy": row["method"],
                 "actions": run.old.WARMUP.tolist()+row["prefix"]+row["suffix"]} for row in cp5["rows"]]
    with tempfile.TemporaryDirectory(prefix="pc_equal_resource_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp/"req.json", tmp/"scene.xml", tmp/"out.json"
        req.write_text(json.dumps({"rows": requests}))
        xml.write_text(official_xml())
        if sha(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("unexpected physical scene")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME/"lib"), str(backend.RUNTIME/"bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY/"bin/python"), str(backend.WORKER), "native",
                        str(req), str(xml), str(out)], env=env, check=True, timeout=120)
        result = json.loads(out.read_text())
    if result["binding"] != "2.1.2.14" or len(result["rows"]) != len(cp5["rows"]):
        raise ValueError("incomplete replay")
    rows = []
    for row, replay in zip(cp5["rows"], result["rows"], strict=True):
        if row["method"] != replay["policy"] or len(replay["trace"]) != 31:
            raise ValueError("physical role/clock mismatch")
        distance = np.linalg.norm(np.asarray([t["state"][:2] for t in replay["trace"]])-[1.8, 2.8], axis=1)
        rows.append({"root": row["root"], "method": row["method"], "initial_distance": float(distance[15]),
                     "prefix_distance": float(distance[20]), "terminal_distance": float(distance[30]),
                     "trace": replay["trace"]})
    run.old.write_json(OUTPUT, {"rows": rows, "root_sha256": sha(run.ROOT_PLAN), "cp5_sha256": sha(run.CP_PLAN),
                                "execution_sha256": sha(run.EXEC), "runner_sha256": sha(Path(__file__)),
                                "worker_sha256": sha(backend.WORKER)})
    print(f"saved {len(rows)} physical replays", flush=True)


if __name__ == "__main__":
    main()
