"""Post-seal physical replay of the direct-CEM RGB feedback baseline."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_goal_mode_rgb_mpc_cem_20261103_codex as run
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

OUTPUT = run.PREFIX.with_name(run.PREFIX.name + "_physical.json")
PROTOCOL = run.ROOT.parent / "docs/PC_GOAL_MODE_RGB_MPC_CEM_PHYSICAL_PROTOCOL_20261103_CODEX.md"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    execution = json.loads(run.EXECUTION.read_text())
    if len(execution["rows"]) != 3 or execution["images_sha256"] != sha(run.IMAGES):
        raise ValueError("execution or image mismatch")
    actions = []
    with np.load(run.IMAGES, allow_pickle=False) as images:
        rgb = images["rgb"]
        if rgb.shape[0] != 31:
            raise ValueError("RGB clock mismatch")
        for step, row in enumerate(execution["rows"]):
            decision = json.loads(run.DECISIONS[step].read_text())
            if (row["step"] != step or row["decision_sha256"] != sha(run.DECISIONS[step])
                    or decision["runner_sha256"] != sha(Path(run.__file__))
                    or decision["protocol_sha256"] != sha(run.PROTOCOL)
                    or row["actions"] != decision["chain"][:5]
                    or row["rgb_sha256"] != [run.old.digest_image(x) for x in rgb[16+5*step:21+5*step]]):
                raise ValueError("decision, action or RGB mismatch")
            actions.extend(row["actions"])
    if actions != execution["executed_actions"] or np.asarray(actions).shape != (15, 2):
        raise ValueError("execution action mismatch")
    request = {"start": list(run.old.STARTS[0]), "policy": "cem",
               "actions": run.old.WARMUP.tolist() + actions}
    with tempfile.TemporaryDirectory(prefix="pc_rgb_cem_phys_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp / "req.json", tmp / "scene.xml", tmp / "out.json"
        req.write_text(json.dumps({"rows": [request]}))
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
    if physical["binding"] != "2.1.2.14" or len(physical["rows"]) != 1:
        raise ValueError("physical backend mismatch")
    trace = physical["rows"][0]["trace"]
    if len(trace) != 31 or physical["rows"][0]["policy"] != "cem":
        raise ValueError("physical replay clock mismatch")
    distances = np.linalg.norm(np.asarray([item["state"][:2] for item in trace]) - [1.8, 2.8], axis=1)
    run.old.write_json(OUTPUT, {"scope": "single reused root, post-seal physical replay",
                               "distances_at_warmup_and_feedback": [float(distances[i]) for i in (15, 20, 25, 30)],
                               "trace": trace, "backend": physical["binding"],
                               "execution_sha256": sha(run.EXECUTION), "images_sha256": sha(run.IMAGES),
                               "decision_sha256": [sha(path) for path in run.DECISIONS],
                               "protocol_sha256": sha(PROTOCOL), "worker_sha256": sha(backend.WORKER),
                               "runner_sha256": sha(Path(__file__))})
    print("saved CEM physical trace and feedback-boundary distances", flush=True)


if __name__ == "__main__":
    main()
