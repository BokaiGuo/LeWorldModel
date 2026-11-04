"""Reconstruct successor Jacobians and replay sealed candidate physics."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc1_rgb_multidirection_successor_response_20261103_codex as source

OUTPUT = source.OUTPUT.with_name(source.OUTPUT.stem + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = source.run.old.prior.sha
    data = json.loads(source.OUTPUT.read_text())
    if data["dependencies"] != {path: sha(Path(path)) for path in data["dependencies"]}:
        raise ValueError("source drift")
    source.run.PREFIX, source.run.ARMS = source.policy.PREFIX, source.policy.ARMS
    requests, expected, max_j_error = [], [], 0.0
    for row in data["rows"]:
        root, branch = row["root"], row["branch"]
        decisions, _, _ = source.run.paths(root, "multidirection")
        decision = json.loads(decisions[0].read_text())["result"]
        parent, nominal = np.asarray(row["parent"]), np.asarray(row["nominal"])
        columns = []
        for direction in np.eye(10).reshape(10, 5, 2):
            plus = np.concatenate((parent, nominal + .05 * direction))
            minus = np.concatenate((parent, nominal - .05 * direction))
            a = source.prior.trace_prediction(decision["prediction_trace"], plus)[0].astype(np.float32)
            b = source.prior.trace_prediction(decision["prediction_trace"], minus)[0].astype(np.float32)
            columns.append((a - b) / .1)
        max_j_error = max(max_j_error, float(np.max(np.abs(np.asarray(columns).T - row["jacobian"]))))
        entries = [("nominal", row["nominal"], row["nominal_trace"])]
        entries.extend((f"candidate{i}", item["tail"], item["trace"]) for i, item in enumerate(row["candidates"]))
        for label, tail, trace in entries:
            requests.append({"start": list(source.run.old.STARTS[root]), "policy": f"{branch}-{label}",
                             "actions": source.run.old.WARMUP.tolist() + row["parent"] + tail})
            expected.append(trace)
        scores = [item["model_score"] for item in row["candidates"]]
        physical = [item["physical_distance"] for item in row["candidates"]]
        order = np.argsort(physical, kind="stable").tolist()
        if (int(np.argmin(scores)) != row["model_best"] or order != row["physical_order"]
                or (order[0] in row["retained"]) != row["physical_best_retained"]):
            raise ValueError("ranking mismatch")
        for item in row["candidates"]:
            model_delta = np.asarray(item["model_position"]) - row["center"]
            physical_delta = np.asarray(item["physical_position"]) - row["physical_nominal_position"]
            if (item["linear_vs_model"] != source.prior.compare(item["model_linear_delta"], model_delta)
                    or item["model_vs_physical"] != source.prior.compare(model_delta, physical_delta)):
                raise ValueError("response metric mismatch")
    if max_j_error != 0.0 or len(requests) != data["physical_requests"]:
        raise ValueError("Jacobian or request count mismatch")
    with tempfile.TemporaryDirectory(prefix="pc1_rgb_successor_audit_") as directory:
        temp = Path(directory)
        req, xml, out = temp / "request.json", temp / "scene.xml", temp / "out.json"
        req.write_text(json.dumps({"rows": requests}, allow_nan=False))
        xml.write_text(source.prior.official_xml())
        if sha(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("scene mismatch")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(source.backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(source.backend.RUNTIME / "lib"), str(source.backend.RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(source.backend.LEGACY / "bin/python"), str(source.backend.WORKER), "native",
                        str(req), str(xml), str(out)], env=env, check=True, timeout=180)
        replayed = json.loads(out.read_text())
    if replayed["binding"] != "2.1.2.14" or len(replayed["rows"]) != len(requests):
        raise ValueError("backend mismatch")
    for request, original, actual in zip(requests, expected, replayed["rows"], strict=True):
        if request["policy"] != actual["policy"] or original != actual["trace"]:
            raise ValueError("physical replay mismatch")
    with OUTPUT.open("x") as stream:
        json.dump({"source_sha256": sha(source.OUTPUT), "audit_sha256": sha(Path(__file__)),
                   "physical_replays": len(requests), "branches": len(data["rows"]),
                   "max_jacobian_reconstruction_error": max_j_error}, stream, indent=2, allow_nan=False)
    print(f"Replayed {len(requests)} successor physical paths and reconstructed {len(data['rows'])} Jacobians.", flush=True)


if __name__ == "__main__":
    main()
