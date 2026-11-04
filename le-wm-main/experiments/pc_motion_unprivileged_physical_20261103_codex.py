"""Post-seal native diagnostic for fixed moving-history programs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_motion_unprivileged_dev_20261103_codex as run
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

ROOT = run.ROOT
PROTOCOL = ROOT.parent / "docs/PC_MOTION_UNPRIVILEGED_PHYSICAL_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_motion_unprivileged_physical_20261103_codex.json"
GOAL = np.array([1.8, 2.8])


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    decision = json.loads(run.DECISIONS.read_text())
    execution = json.loads(run.RECEIPT.read_text())
    if (execution["decision_sha256"] != run.prior.sha(run.DECISIONS)
            or execution["arrays_sha256"] != run.prior.sha(run.EXECUTION)
            or decision["runner_sha256"] != run.prior.sha(Path(run.__file__))
            or decision["generator_sha256"] != run.prior.sha(Path(run.policy.__file__))):
        raise ValueError("sealed dependency mismatch")
    rng = np.random.default_rng(2026110350)
    common = np.concatenate((np.zeros((1, 10, 2)), rng.uniform(-1, 1, (5, 10, 2))))
    requests, descriptors = [], []
    for row in decision["rows"]:
        root = row["root_index"]
        for index, candidate in enumerate(row["candidates"]):
            program = candidate["program"]
            requests.append({"start": list(run.STARTS[root]), "policy": program["name"],
                             "actions": row["warmup"] + program["prefix"] + program["suffix"]})
            descriptors.append({"kind": "model_chain", "root": root, "candidate_index": index})
        names = [row["choice"]["selected_name"], "zero", "random_0"]
        for role, name in zip(("selected", "zero", "random_0"), names, strict=True):
            prefix = next(c["program"]["prefix"] for c in row["candidates"] if c["program"]["name"] == name)
            for suffix_index, suffix in enumerate(common):
                requests.append({"start": list(run.STARTS[root]), "policy": name,
                                 "actions": row["warmup"] + prefix + suffix.tolist()})
                descriptors.append({"kind": "common_suffix", "root": root, "role": role,
                                    "name": name, "suffix_index": suffix_index})
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pc_motion_phys_") as directory:
        temp = Path(directory)
        req, xml, out = temp / "request.json", temp / "official.xml", temp / "out.json"
        req.write_text(json.dumps({"rows": requests}))
        xml.write_text(official_xml())
        if run.prior.sha(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("native XML mismatch")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"), str(backend.RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY / "bin/python"), str(backend.WORKER), "native",
                        str(req), str(xml), str(out)], env=env, check=True, timeout=120)
        native = json.loads(out.read_text())
    if len(native["rows"]) != len(descriptors) or native["binding"] != "2.1.2.14":
        raise ValueError("incomplete replay")
    rows = []
    for descriptor, replay in zip(descriptors, native["rows"], strict=True):
        trace = replay["trace"]
        if len(trace) != 31:
            raise ValueError("unexpected physical action clock")
        positions = np.asarray([t["state"][:2] for t in trace])
        distances = np.linalg.norm(positions - GOAL, axis=1)
        rows.append({**descriptor, "initial_distance": float(distances[15]),
                     "prefix_distance": float(distances[20]), "terminal_distance": float(distances[30]),
                     "prefix_position": positions[20].tolist(), "trace": trace})
    report = {"schema": "pc_motion_unprivileged_physical.v1", "rows": rows,
              "common_suffixes": common.tolist(), "decision_sha256": run.prior.sha(run.DECISIONS),
              "execution_sha256": run.prior.sha(run.RECEIPT), "worker_sha256": run.prior.sha(backend.WORKER),
              "protocol_sha256": run.prior.sha(PROTOCOL), "runner_sha256": run.prior.sha(Path(__file__)),
              "elapsed_seconds": time.monotonic()-started, "physical_chains": len(rows),
              "note": "post-seal diagnostic; physical states unavailable to RGB planner"}
    with OUTPUT.open("x") as handle:
        json.dump(report, handle, allow_nan=False)
        handle.write("\n")
    print(f"saved {len(rows)} physical chains", flush=True)


if __name__ == "__main__":
    main()
