"""Post-seal native physical evaluation of RGB-replanned cp5 suffixes."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_motion_cp5_replan_20261103_codex as cp5
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

OUTPUT = cp5.OUT.with_name(cp5.OUT.name + "_physical.json")
GOAL = np.array([1.8, 2.8])


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    decisions, executed = (json.loads(p.read_text()) for p in (cp5.DECISION, cp5.RECEIPT))
    if (executed["decision_sha256"] != cp5.prior.prior.sha(cp5.DECISION)
            or executed["arrays_sha256"] != cp5.prior.prior.sha(cp5.EXECUTION)
            or decisions["context_sha256"] != cp5.prior.prior.sha(cp5.CONTEXT)
            or decisions["runner_sha256"] != cp5.prior.prior.sha(Path(cp5.__file__))):
        raise ValueError("cp5 receipt mismatch")
    requests = [{"start": list(cp5.prior.STARTS[row["root_index"]]), "policy": row["role"],
                 "actions": cp5.prior.WARMUP.tolist() + row["prefix"] + row["suffix"]}
                for row in decisions["rows"]]
    with tempfile.TemporaryDirectory(prefix="pc_cp5_physical_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp / "req.json", tmp / "scene.xml", tmp / "out.json"
        req.write_text(json.dumps({"rows": requests}))
        xml.write_text(official_xml())
        if cp5.prior.prior.sha(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("wrong XML")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME/"lib"), str(backend.RUNTIME/"bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY/"bin/python"), str(backend.WORKER), "native",
                        str(req), str(xml), str(out)], env=env, check=True, timeout=120)
        data = json.loads(out.read_text())
    if len(data["rows"]) != 12 or data["binding"] != "2.1.2.14":
        raise ValueError("incomplete native replay")
    rows = []
    for decision, result in zip(decisions["rows"], data["rows"], strict=True):
        if result["policy"] != decision["role"] or len(result["trace"]) != 31:
            raise ValueError("physical role/clock mismatch")
        distances = [float(np.linalg.norm(np.asarray(t["state"][:2])-GOAL)) for t in result["trace"]]
        rows.append({"root_index": decision["root_index"], "role": decision["role"],
                     "initial_distance": distances[15], "prefix_distance": distances[20],
                     "terminal_distance": distances[30], "trace": result["trace"]})
    cp5.prior.write_json(OUTPUT, {"schema": "pc_motion_cp5_physical.v1", "rows": rows,
                                  "decision_sha256": cp5.prior.prior.sha(cp5.DECISION),
                                  "execution_sha256": cp5.prior.prior.sha(cp5.RECEIPT),
                                  "worker_sha256": cp5.prior.prior.sha(backend.WORKER),
                                  "runner_sha256": cp5.prior.prior.sha(Path(__file__))})
    print("saved 12 physical cp5 suffix replays", flush=True)


if __name__ == "__main__":
    main()
