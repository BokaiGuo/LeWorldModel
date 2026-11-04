"""Audit fixed-terminal real-RGB receipts and replay exact actions in the native backend."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_fixed_terminal_fair_feedback_dev_20261103_codex as run
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

OUTPUT = run.PREFIX.with_name(run.PREFIX.name + "_physical.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    requests, bindings = [], []
    for root in range(4):
        for arm in run.ARMS:
            decisions, execution, images = run.paths(root, arm)
            recorded = json.loads(execution.read_text())
            if (recorded["root"] != root or recorded["arm"] != arm or len(recorded["rows"]) != 3
                    or recorded["images_sha256"] != sha(images)):
                raise ValueError("execution binding mismatch")
            full, previous_chain, tokens, svd, planning_seconds, encoder_seconds = [], None, [], [], [], []
            with np.load(images, allow_pickle=False) as file:
                rgb = file["rgb"]
                if rgb.shape[0] != 31:
                    raise ValueError("RGB length mismatch")
                for step, item in enumerate(recorded["rows"]):
                    decision = json.loads(decisions[step].read_text())
                    result = decision["result"]
                    chain = result["selected"]["chain"]
                    if (decision["root"] != root or decision["arm"] != arm or decision["step"] != step
                            or decision["runner_sha256"] != sha(Path(run.__file__))
                            or decision["protocol_sha256"] != sha(run.PROTOCOL)
                            or decision["optimizer_sha256"] != sha(Path(run.optimize.__code__.co_filename))
                            or decision["generator_sha256"] != sha(Path(run.generator.__file__))
                            or item["decision_sha256"] != sha(decisions[step]) or item["actions"] != chain[:5]
                            or result["horizon"] != 15 - 5 * step
                            or decision["counts"]["prediction_token_samples"] > run.CAP
                            or item["rgb_sha256"] != [run.old.digest_image(x) for x in rgb[16 + step * 5:21 + step * 5]]):
                        raise ValueError("decision/action/image mismatch")
                    if step and result["carry"] != previous_chain[5:]:
                        raise ValueError("carry not bound to previous selected tail")
                    if step and result["selected"]["terminal_score"] > result["carry_score"] + 1e-12:
                        raise ValueError("selected score worse than rescored carry")
                    if decision["input_rgb_sha256"] != [run.old.digest_image(rgb[t]) for t in decision["ticks"]]:
                        raise ValueError("decision context image mismatch")
                    previous_chain = chain
                    full.extend(item["actions"])
                    tokens.append(decision["counts"]["prediction_token_samples"])
                    svd.append(result["spectral_counts"].get("svd", 0))
                    planning_seconds.append(decision["planning_seconds"])
                    encoder_seconds.append(decision["encoding_seconds"])
            if full != recorded["executed_actions"] or np.asarray(full).shape != (15, 2):
                raise ValueError("executed sequence mismatch")
            requests.append({"start": list(run.old.STARTS[root]), "policy": f"{root}-{arm}",
                             "actions": run.old.WARMUP.tolist() + full})
            bindings.append({"root": root, "arm": arm, "execution_sha256": sha(execution),
                             "images_sha256": sha(images), "decision_sha256": [sha(p) for p in decisions],
                             "tokens": tokens, "svd": svd, "planning_seconds": planning_seconds,
                             "encoding_seconds": encoder_seconds})
    with tempfile.TemporaryDirectory(prefix="pc_fixed_terminal_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp / "request.json", tmp / "scene.xml", tmp / "out.json"
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
    if physical["binding"] != "2.1.2.14" or len(physical["rows"]) != len(bindings):
        raise ValueError("backend mismatch")
    rows = []
    for binding, replay in zip(bindings, physical["rows"], strict=True):
        if replay["policy"] != f'{binding["root"]}-{binding["arm"]}' or len(replay["trace"]) != 31:
            raise ValueError("replay clock mismatch")
        distances = np.linalg.norm(np.asarray([x["state"][:2] for x in replay["trace"]]) - [1.8, 2.8], axis=1)
        rows.append({**binding, "distances": [float(distances[t]) for t in (15, 20, 25, 30)],
                     "trace": replay["trace"]})
    run.old.write_json(OUTPUT, {"rows": rows, "backend": physical["binding"],
                               "protocol_sha256": sha(run.PROTOCOL), "runner_sha256": sha(Path(__file__)),
                               "worker_sha256": sha(backend.WORKER), "full_cost_certified": False})
    print("Native physical replay complete for 8 sealed feedback trajectories.", flush=True)


if __name__ == "__main__":
    main()
