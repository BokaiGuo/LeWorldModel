"""Post-seal physical evaluation and common-suffix diagnostic."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_multinominal_compare_20261103_codex as run
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

OUTPUT = run.PREFIX.with_name(run.PREFIX.name+"_physical.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    root, cp10, execution = (json.loads(path.read_text()) for path in (run.ROOT_PLAN, run.CP_PLAN, run.EXEC))
    if (root["runner_sha256"] != sha(Path(run.__file__)) or root["protocol_sha256"] != sha(run.PROTOCOL)
            or root["generator_sha256"] != sha(Path(run.compiler.__file__))
            or cp10["root_plan_sha256"] != sha(run.ROOT_PLAN) or cp10["context_sha256"] != sha(run.CONTEXT)
            or execution["cp10_sha256"] != sha(run.CP_PLAN) or execution["images_sha256"] != sha(run.IMAGES)):
        raise ValueError("sealed dependency mismatch")
    requests, desc = [], []
    for row in cp10["rows"]:
        requests.append({"start": list(run.old.STARTS[row["root"]]), "policy": row["method"],
                         "actions": run.old.WARMUP.tolist()+row["first10"]+row["suffix"]})
        desc.append({"root": row["root"], "budget": row["budget"], "method": row["method"], "kind": "feedback",
                     "prefix_length": 10})
    rng = np.random.default_rng(2026110650)
    common = np.concatenate((np.zeros((1, 5, 2)), rng.uniform(-1, 1, (4, 5, 2))))
    for anchor in (x for x in root["rows"] if x["method"] == "spectral"):
        root_id, budget, horizon = anchor["root"], anchor["budget"], anchor["chosen"]["horizon"]
        if horizon not in (5, 10):
            raise ValueError("invalid chosen spectral horizon")
        peers = [x for x in root["rows"] if x["root"] == root_id and x["budget"] == budget]
        prefixes = {x["method"]: x["chosen"]["chain"][:horizon] for x in peers}
        prefixes["zero"] = np.zeros((horizon, 2)).tolist()
        for method, prefix in prefixes.items():
            for index, suffix in enumerate(common):
                requests.append({"start": list(run.old.STARTS[root_id]), "policy": method,
                                 "actions": run.old.WARMUP.tolist()+prefix+suffix.tolist()})
                desc.append({"root": root_id, "budget": budget, "method": method, "kind": "common",
                             "prefix_length": horizon, "suffix_index": index})
    with tempfile.TemporaryDirectory(prefix="pc_multinominal_physical_") as directory:
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
    if physical["binding"] != "2.1.2.14" or len(physical["rows"]) != len(desc):
        raise ValueError("physical replay incomplete")
    rows = []
    for info, replay in zip(desc, physical["rows"], strict=True):
        trace = replay["trace"]
        expected = 31 if info["kind"] == "feedback" else 21+info["prefix_length"]
        if replay["policy"] != info["method"] or len(trace) != expected:
            raise ValueError("physical role or clock mismatch")
        distance = np.linalg.norm(np.asarray([x["state"][:2] for x in trace])-[1.8, 2.8], axis=1)
        rows.append({**info, "initial_distance": float(distance[15]),
                     "prefix_distance": float(distance[15+info["prefix_length"]]),
                     "terminal_distance": float(distance[-1]), "trace": trace})
    run.old.write_json(OUTPUT, {"rows": rows, "common_suffixes": common.tolist(),
                                "root_sha256": sha(run.ROOT_PLAN), "cp10_sha256": sha(run.CP_PLAN),
                                "execution_sha256": sha(run.EXEC), "runner_sha256": sha(Path(__file__)),
                                "worker_sha256": sha(backend.WORKER)})
    print(f"saved {len(rows)} physical traces", flush=True)


if __name__ == "__main__":
    main()
