"""Replay all sealed extra-budget CEM trajectories in physical dynamics."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_cem_extra_compute_feedback_20261103_codex as run
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

OUTPUT = run.PREFIX.with_name(run.PREFIX.name + "_physical.json")
PROTOCOL = run.ROOT.parent / "docs/PC_CEM_EXTRA_COMPUTE_FEEDBACK_PHYSICAL_PROTOCOL_20261103_CODEX.md"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    requests, bindings = [], []
    for root in range(4):
        for budget in run.BUDGETS:
            decisions, execution, images = run.paths(root, budget)
            record = json.loads(execution.read_text())
            if (record["root"] != root or record["budget"] != budget or len(record["rows"]) != 3
                    or record["images_sha256"] != sha(images)):
                raise ValueError("execution or image mismatch")
            actions = []
            with np.load(images, allow_pickle=False) as image_file:
                rgb = image_file["rgb"]
                if rgb.shape[0] != 31:
                    raise ValueError("RGB clock mismatch")
                for step, row in enumerate(record["rows"]):
                    decision = json.loads(decisions[step].read_text())
                    if (row["step"] != step or row["decision_sha256"] != sha(decisions[step])
                            or decision["root"] != root or decision["budget"] != budget
                            or decision["step"] != step or decision["runner_sha256"] != sha(Path(run.__file__))
                            or decision["protocol_sha256"] != sha(run.PROTOCOL)
                            or decision["optimizer_sha256"] != sha(Path(run.optimize.__code__.co_filename))
                            or decision["token_counts"]["prediction_token_samples"] != budget
                            or row["actions"] != decision["chain"][:5]
                            or row["rgb_sha256"] != [run.old.digest_image(x)
                                                    for x in rgb[16 + 5*step:21 + 5*step]]):
                        raise ValueError("decision, action or RGB mismatch")
                    actions.extend(row["actions"])
            if actions != record["executed_actions"] or np.asarray(actions).shape != (15, 2):
                raise ValueError("action clock mismatch")
            requests.append({"start": list(run.old.STARTS[root]), "policy": str(budget),
                             "actions": run.old.WARMUP.tolist() + actions})
            bindings.append({"root": root, "budget": budget, "execution_sha256": sha(execution),
                             "images_sha256": sha(images),
                             "decision_sha256": [sha(path) for path in decisions]})
    with tempfile.TemporaryDirectory(prefix="pc_cem_extra_physical_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp / "req.json", tmp / "scene.xml", tmp / "out.json"
        req.write_text(json.dumps({"rows": requests}))
        xml.write_text(official_xml())
        if sha(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("scene mismatch")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"), str(backend.RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY / "bin/python"), str(backend.WORKER), "native",
                        str(req), str(xml), str(out)], env=env, check=True, timeout=180)
        physical = json.loads(out.read_text())
    if physical["binding"] != "2.1.2.14" or len(physical["rows"]) != len(requests):
        raise ValueError("physical backend or count mismatch")
    rows = []
    for binding, replay in zip(bindings, physical["rows"], strict=True):
        trace = replay["trace"]
        if replay["policy"] != str(binding["budget"]) or len(trace) != 31:
            raise ValueError("physical clock mismatch")
        distances = np.linalg.norm(np.asarray([x["state"][:2] for x in trace]) - [1.8, 2.8], axis=1)
        rows.append({**binding, "distances": [float(distances[i]) for i in (15, 20, 25, 30)], "trace": trace})
    run.old.write_json(OUTPUT, {"scope": "four reused roots, post-seal extra-budget CEM development",
                               "rows": rows, "backend": physical["binding"], "protocol_sha256": sha(PROTOCOL),
                               "worker_sha256": sha(backend.WORKER), "runner_sha256": sha(Path(__file__))})
    print(f"saved {len(rows)} physical traces", flush=True)


if __name__ == "__main__":
    main()
