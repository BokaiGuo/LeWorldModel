"""Physical replay of sealed, equally optimized fixed candidate chains."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_budget_candidate_quality_20261103_codex as run
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

OUTPUT = run.PREFIX.with_name(run.PREFIX.name + "_physical.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    source = json.loads(run.ROOT_PLAN.read_text())
    if (source["runner_sha256"] != sha(Path(run.__file__))
            or source["protocol_sha256"] != sha(run.PROTOCOL)):
        raise ValueError("sealed dependency mismatch")
    requests, labels = [], []
    for row in source["rows"]:
        for index, program in enumerate(row["programs"]):
            requests.append({"start": list(run.old.STARTS[row["root"]]), "policy": str(index),
                             "actions": run.old.WARMUP.tolist() + program["chain"]})
            labels.append({"root": row["root"], "candidate_index": index,
                           "name": program["name"], "source_bank": program["source_bank"],
                           "model_score": program["terminal_score"]})
    with tempfile.TemporaryDirectory(prefix="pc_quality_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp/"req.json", tmp/"scene.xml", tmp/"out.json"
        req.write_text(json.dumps({"rows": requests}))
        xml.write_text(official_xml())
        if sha(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("physical scene mismatch")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME/"lib"), str(backend.RUNTIME/"bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY/"bin/python"), str(backend.WORKER), "native",
                        str(req), str(xml), str(out)], env=env, check=True, timeout=180)
        physical = json.loads(out.read_text())
    if physical["binding"] != "2.1.2.14" or len(physical["rows"]) != len(labels):
        raise ValueError("physical replay incomplete")
    results = []
    for label, replay in zip(labels, physical["rows"], strict=True):
        if replay["policy"] != str(label["candidate_index"]) or len(replay["trace"]) != 31:
            raise ValueError("physical replay clock mismatch")
        distance = np.linalg.norm(np.asarray([x["state"][:2] for x in replay["trace"]])-[1.8, 2.8], axis=1)
        results.append({**label, "terminal_distance": float(distance[-1]), "trace": replay["trace"]})
    run.old.write_json(OUTPUT, {"rows": results, "root_sha256": sha(run.ROOT_PLAN),
                               "worker_sha256": sha(backend.WORKER), "runner_sha256": sha(Path(__file__))})
    print(f"saved {len(results)} physical traces", flush=True)


if __name__ == "__main__":
    main()
