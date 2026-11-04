"""Evaluate sealed RGB-replanned suffixes on the native physical backend."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_spectral_bank_cp5_20261103_codex as run
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

OUTPUT = run.BASE.with_name(run.BASE.name+"_physical.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    plans = json.loads(run.DECISION.read_text())
    execution = json.loads(run.EXECUTION.read_text())
    if (plans["runner_sha256"] != sha(Path(run.__file__)) or plans["source_sha256"] != sha(run.prior.OUTPUT)
            or plans["context_sha256"] != sha(run.CONTEXT)
            or execution["decision_sha256"] != sha(run.DECISION) or execution["arrays_sha256"] != sha(run.IMAGES)):
        raise ValueError("cp5 evidence mismatch")
    requests = [{"start": list(run.old.STARTS[x["root"]]), "policy": x["role"],
                 "actions": run.old.WARMUP.tolist()+x["prefix"]+x["suffix"]} for x in plans["rows"]]
    with tempfile.TemporaryDirectory(prefix="pc_spectral_cp5_physical_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp/"req.json", tmp/"scene.xml", tmp/"out.json"
        req.write_text(json.dumps({"rows": requests}))
        xml.write_text(official_xml())
        if sha(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("scene mismatch")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME/"lib"), str(backend.RUNTIME/"bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY/"bin/python"), str(backend.WORKER), "native",
                        str(req), str(xml), str(out)], env=env, check=True, timeout=120)
        physical = json.loads(out.read_text())
    if physical["binding"] != "2.1.2.14" or len(physical["rows"]) != len(plans["rows"]):
        raise ValueError("incomplete physical replay")
    rows = []
    for plan, result in zip(plans["rows"], physical["rows"], strict=True):
        if result["policy"] != plan["role"] or len(result["trace"]) != 31:
            raise ValueError("role or clock mismatch")
        distance = np.linalg.norm(np.asarray([x["state"][:2] for x in result["trace"]]) - [1.8, 2.8], axis=1)
        rows.append({"root": plan["root"], "role": plan["role"], "initial_distance": float(distance[15]),
                     "prefix_distance": float(distance[20]), "terminal_distance": float(distance[30]),
                     "trace": result["trace"]})
    run.old.write_json(OUTPUT, {"rows": rows, "decision_sha256": sha(run.DECISION),
                                "execution_sha256": sha(run.EXECUTION), "runner_sha256": sha(Path(__file__)),
                                "worker_sha256": sha(backend.WORKER)})
    print(f"saved {len(rows)} cp5 physical traces", flush=True)


if __name__ == "__main__":
    main()
