"""Audit sealed RGB decisions, then replay their actions on the native backend."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_multidirection_rgb_fixed_terminal_dev_20261103_codex as policy
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

run = policy.runner
OUTPUT = policy.PREFIX.with_name(policy.PREFIX.name + "_physical.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    run.PREFIX, run.ARMS = policy.PREFIX, policy.ARMS
    sha = run.old.prior.sha
    requests, bindings = [], []
    for root in range(4):
        for arm in run.ARMS:
            decisions, execution, images = run.paths(root, arm)
            receipt = json.loads(execution.read_text())
            if receipt["root"] != root or receipt["arm"] != arm or receipt["images_sha256"] != sha(images):
                raise ValueError("execution binding mismatch")
            full, previous, counts, max_rounding = [], None, [], 0.0
            with np.load(images, allow_pickle=False) as archive:
                rgb = archive["rgb"]
                if len(rgb) != 31 or len(receipt["rows"]) != 3:
                    raise ValueError("image clock mismatch")
                for step, row in enumerate(receipt["rows"]):
                    decision = json.loads(decisions[step].read_text())
                    result = decision["result"]
                    chain = np.asarray(result["selected"]["chain"], dtype=float)
                    if (decision["root"] != root or decision["arm"] != arm or decision["step"] != step
                            or decision["protocol_sha256"] != sha(policy.PROTOCOL)
                            or result["actual_source_sha256"] != policy.source_hashes()
                            or row["decision_sha256"] != sha(decisions[step])
                            or chain.shape != (15 - 5 * step, 2)
                            or row["actions"] != chain[:5].tolist()
                            or row["rgb_sha256"] != [run.old.digest_image(x) for x in rgb[16 + 5 * step:21 + 5 * step]]
                            or decision["input_rgb_sha256"] != [run.old.digest_image(rgb[t]) for t in decision["ticks"]]):
                        raise ValueError("decision binding mismatch")
                    if step and not np.array_equal(np.asarray(result["carry"]), previous[5:]):
                        raise ValueError("carry mismatch")
                    trace = result["prediction_trace"]
                    total = 0
                    for batch in trace:
                        actions = np.asarray(batch["requested_actions"])
                        converted = np.asarray(batch["model_raw_float32_actions"])
                        if not np.array_equal(actions.astype(np.float32), converted.astype(np.float32)):
                            raise ValueError("float32 prediction action mismatch")
                        total += len(actions) * actions.shape[1] // 5
                        if batch["tokens"] != len(actions) * actions.shape[1] // 5:
                            raise ValueError("batch token mismatch")
                    if total != decision["counts"]["prediction_token_samples"] or total > run.CAP:
                        raise ValueError("total token mismatch")
                    max_rounding = max(max_rounding, float(np.max(np.abs(chain - chain.astype(np.float32)))))
                    counts.append(total)
                    previous = chain
                    full.extend(row["actions"])
            if full != receipt["executed_actions"]:
                raise ValueError("executed actions mismatch")
            requests.append({"start": list(run.old.STARTS[root]), "policy": f"{root}-{arm}",
                             "actions": run.old.WARMUP.tolist() + full})
            bindings.append({"root": root, "arm": arm, "execution_sha256": sha(execution),
                             "images_sha256": sha(images), "decision_sha256": [sha(p) for p in decisions],
                             "tokens": counts, "max_float32_rounding": max_rounding})
    with tempfile.TemporaryDirectory(prefix="pc_multidirection_rgb_replay_") as directory:
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
                               "protocol_sha256": sha(policy.PROTOCOL),
                               "replay_sha256": sha(Path(__file__)), "worker_sha256": sha(backend.WORKER),
                               "full_cost_certified": False})
    print("Audited and replayed eight RGB feedback trajectories.", flush=True)


if __name__ == "__main__":
    main()
