"""Post-seal follow-up physical and common suffix replay."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_spectral_followup_20261103_codex as run
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

OUTPUT = run.PREFIX.with_name(run.PREFIX.name+"_physical.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    decisions = json.loads(run.DECISIONS.read_text())
    receipt = json.loads(run.RECEIPT.read_text())
    if (decisions["runner_sha256"] != sha(Path(run.__file__)) or decisions["protocol_sha256"] != sha(run.PROTOCOL)
            or decisions["source_sha256"] != sha(run.base.OUTPUT) or decisions["context_sha256"] != sha(run.CONTEXT)
            or receipt["decision_sha256"] != sha(run.DECISIONS) or receipt["arrays_sha256"] != sha(run.IMAGES)):
        raise ValueError("followup evidence mismatch")
    rng = np.random.default_rng(2026110350)
    common = np.concatenate((np.zeros((1, 10, 2)), rng.uniform(-1, 1, (5, 10, 2))))
    requests, desc = [], []
    for row in decisions["rows"]:
        for kind, suffix in [("original", row["old_suffix"]), ("replanned", row["suffix"])] + [
                (f"common_{i}", action.tolist()) for i, action in enumerate(common)]:
            requests.append({"start": list(run.old.STARTS[row["root"]]), "policy": row["role"],
                             "actions": run.old.WARMUP.tolist()+row["prefix"]+suffix})
            desc.append({"root": row["root"], "role": row["role"], "kind": kind})
    with tempfile.TemporaryDirectory(prefix="pc_spectral_followup_") as directory:
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
                        str(req), str(xml), str(out)], env=env, check=True, timeout=180)
        physical = json.loads(out.read_text())
    if physical["binding"] != "2.1.2.14" or len(physical["rows"]) != len(desc):
        raise ValueError("incomplete replay")
    rows = []
    for key, result in zip(desc, physical["rows"], strict=True):
        if result["policy"] != key["role"] or len(result["trace"]) != 31:
            raise ValueError("physical clock or role mismatch")
        distance = np.linalg.norm(np.asarray([x["state"][:2] for x in result["trace"]]) - [1.8, 2.8], axis=1)
        rows.append({**key, "initial_distance": float(distance[15]), "prefix_distance": float(distance[20]),
                     "terminal_distance": float(distance[30]), "trace": result["trace"]})
    run.old.write_json(OUTPUT, {"rows": rows, "common_suffixes": common.tolist(),
                                "decision_sha256": sha(run.DECISIONS), "execution_sha256": sha(run.RECEIPT),
                                "worker_sha256": sha(backend.WORKER), "runner_sha256": sha(Path(__file__))})
    print(f"saved {len(rows)} physical traces", flush=True)


if __name__ == "__main__":
    main()
