"""Post-seal open-loop and paired common-suffix physical diagnostic."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_spectral_bank_rgb_dev_20261103_codex as run
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

OUTPUT = run.ROOT / "results/pc_spectral_bank_rgb_dev_20261103_codex_physical.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    decisions = json.loads(run.OUTPUT.read_text())
    execution = json.loads(run.EXECUTION.read_text())
    if (execution["decision_sha256"] != sha(run.OUTPUT)
            or execution["arrays_sha256"] != sha(run.IMAGES)
            or decisions["protocol_sha256"] != sha(run.PROTOCOL)
            or decisions["runner_sha256"] != sha(Path(run.__file__))
            or decisions["core_sha256"] != sha(Path(run.core.__file__))):
        raise ValueError("sealed dependency mismatch")
    rng = np.random.default_rng(2026110350)
    common = np.concatenate((np.zeros((1, 10, 2)), rng.uniform(-1, 1, (5, 10, 2))))
    requests, descriptors = [], []

    def append(root, name, prefix, suffix, descriptor):
        requests.append({"start": list(run.old.STARTS[root]), "policy": name,
                         "actions": run.old.WARMUP.tolist()+prefix+suffix})
        descriptors.append(descriptor)

    for row in decisions["rows"]:
        root = row["root"]
        for item in row["plan"]["programs"]:
            append(root, item["name"], item["prefix"], item["suffix"],
                   {"root": root, "kind": "candidate", "name": item["name"]})
        for role, item in row["arms"].items():
            append(root, item["name"], item["prefix"], item["suffix"],
                   {"root": root, "kind": "arm", "role": role, "name": item["name"]})
            for index, suffix in enumerate(common):
                append(root, item["name"], item["prefix"], suffix.tolist(),
                       {"root": root, "kind": "common", "role": role, "suffix_index": index})
    with tempfile.TemporaryDirectory(prefix="pc_spectral_rgb_physical_") as directory:
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
                        str(req), str(xml), str(out)], env=env, check=True, timeout=180)
        physical = json.loads(out.read_text())
    if physical["binding"] != "2.1.2.14" or len(physical["rows"]) != len(descriptors):
        raise ValueError("incomplete physical replay")
    rows = []
    for desc, replay in zip(descriptors, physical["rows"], strict=True):
        trace = replay["trace"]
        if len(trace) != 31:
            raise ValueError("wrong action clock")
        distances = np.linalg.norm(np.asarray([x["state"][:2] for x in trace]) - [1.8, 2.8], axis=1)
        rows.append({**desc, "initial_distance": float(distances[15]), "prefix_distance": float(distances[20]),
                     "terminal_distance": float(distances[30]), "trace": trace})
    run.old.write_json(OUTPUT, {"schema": "pc_spectral_bank_rgb_physical.v1", "rows": rows,
                                "common_suffixes": common.tolist(), "decision_sha256": sha(run.OUTPUT),
                                "execution_sha256": sha(run.EXECUTION), "runner_sha256": sha(Path(__file__)),
                                "worker_sha256": sha(backend.WORKER)})
    print(f"saved {len(rows)} paired physical traces", flush=True)


if __name__ == "__main__":
    main()
