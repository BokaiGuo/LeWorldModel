"""Post-seal physical replay of the goal-mode RGB smoke candidate bank."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_goal_mode_rgb_smoke_20261103_codex as smoke
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

OUTPUT = smoke.PREFIX.with_name(smoke.PREFIX.name + "_physical.json")
PROTOCOL = smoke.ROOT.parent / "docs/PC_GOAL_MODE_RGB_PHYSICAL_DIAGNOSTIC_PROTOCOL_20261103_CODEX.md"


def validate(plan, execution, plan_sha, images_sha):
    if execution["plan_sha256"] != plan_sha or execution["images_sha256"] != images_sha:
        raise ValueError("sealed execution hash mismatch")
    selected = plan["result"]["selected"]
    programs = plan["result"]["programs"]
    names = [item["name"] for item in programs]
    if len(names) != len(set(names)) or names.count(selected["name"]) != 1:
        raise ValueError("invalid candidate identities")
    chains = {}
    for item in programs:
        chain = np.asarray(item["prefix"] + item["suffix"], dtype=float)
        if chain.shape != (15, 2) or not np.isfinite(chain).all() or np.any(np.abs(chain) > 1):
            raise ValueError("invalid sealed action chain")
        chains[item["name"]] = chain.tolist()
    if chains[selected["name"]] != plan["chain"] or plan["chain"] != execution["chain"]:
        raise ValueError("selected execution mismatch")
    return chains


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = smoke.old.prior.sha
    plan, execution = (json.loads(path.read_text()) for path in (smoke.PLAN, smoke.EXECUTION))
    if (plan["runner_sha256"] != sha(Path(smoke.__file__))
            or plan["compiler_sha256"] != sha(Path(smoke.compiler.__file__))
            or plan["protocol_sha256"] != sha(smoke.PROTOCOL)
            or plan["history_sha256"] != sha(smoke.old.HISTORY)):
        raise ValueError("sealed source mismatch")
    chains = validate(plan, execution, sha(smoke.PLAN), sha(smoke.IMAGES))
    requests = [{"start": list(smoke.old.STARTS[0]), "policy": name,
                 "actions": smoke.old.WARMUP.tolist() + chain}
                for name, chain in chains.items()]
    with tempfile.TemporaryDirectory(prefix="pc_goal_mode_physical_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp / "req.json", tmp / "scene.xml", tmp / "out.json"
        req.write_text(json.dumps({"rows": requests}))
        xml.write_text(official_xml())
        if sha(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("physical scene mismatch")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"), str(backend.RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY / "bin/python"), str(backend.WORKER), "native",
                        str(req), str(xml), str(out)], env=env, check=True, timeout=180)
        physical = json.loads(out.read_text())
    if physical["binding"] != "2.1.2.14" or len(physical["rows"]) != len(requests):
        raise ValueError("physical backend or row count mismatch")
    records = []
    for request, replay in zip(requests, physical["rows"], strict=True):
        trace = replay["trace"]
        if replay["policy"] != request["policy"] or len(trace) != 31:
            raise ValueError("replay identity or clock mismatch")
        distance = np.linalg.norm(np.asarray([row["state"][:2] for row in trace]) - [1.8, 2.8], axis=1)
        records.append({"name": request["policy"],
                        "selected": request["policy"] == plan["result"]["selected"]["name"],
                        "same_bank_greedy": request["policy"] == plan["result"]["same_bank_greedy_name"],
                        "initial_distance": float(distance[15]), "prefix_distance": float(distance[20]),
                        "terminal_distance": float(distance[30]), "trace": trace})
    smoke.old.write_json(OUTPUT, {"scope": "post-seal single-root fixed-bank physical diagnostic",
                                 "rows": records, "backend": physical["binding"],
                                 "plan_sha256": sha(smoke.PLAN), "execution_sha256": sha(smoke.EXECUTION),
                                 "images_sha256": sha(smoke.IMAGES), "worker_sha256": sha(backend.WORKER),
                                 "protocol_sha256": sha(PROTOCOL), "runner_sha256": sha(Path(__file__))})
    print(f"saved {len(records)} post-seal physical traces", flush=True)


if __name__ == "__main__":
    main()
