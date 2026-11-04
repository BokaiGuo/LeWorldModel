"""Independent deterministic replay of the sealed candidate-response diagnostic."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc1_rgb_multidirection_candidate_response_20261103_codex as source

OUTPUT = source.OUTPUT.with_name(source.OUTPUT.stem + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = source.run.old.prior.sha
    data = json.loads(source.OUTPUT.read_text())
    if data["dependencies"] != {path: sha(Path(path)) for path in data["dependencies"]}:
        raise ValueError("diagnostic source drift")
    source.run.PREFIX, source.run.ARMS = source.policy.PREFIX, source.policy.ARMS
    requests, expected, max_j_error = [], [], 0.0
    grouped = {}
    for item in data["rows"]:
        grouped.setdefault((item["root"], item["step"]), []).append(item)
    for (root, step), rows in sorted(grouped.items()):
        decisions, execution, _ = source.run.paths(root, "multidirection")
        result = json.loads(decisions[step].read_text())["result"]
        receipt = json.loads(execution.read_text())
        history = [action for row in receipt["rows"][:step] for action in row["actions"]]
        nominal = np.asarray(result["response"]["nominal"])
        jacobian = np.asarray(result["response"]["jacobian"])
        columns = []
        for direction in np.eye(10).reshape(10, 5, 2):
            plus = source.trace_prediction(result["prediction_trace"], nominal + .05 * direction)[0].astype(np.float32)
            minus = source.trace_prediction(result["prediction_trace"], nominal - .05 * direction)[0].astype(np.float32)
            columns.append((plus - minus) / .1)
        max_j_error = max(max_j_error, float(np.max(np.abs(np.asarray(columns).T - jacobian))))
        labels = [("nominal", nominal.tolist(), rows[0]["nominal_trace"])]
        labels.extend((f"candidate{i}", item["actions"], item["candidate_trace"]) for i, item in enumerate(rows))
        for name, actions, trace in labels:
            requests.append({"start": list(source.run.old.STARTS[root]), "policy": f"{root}-{step}-{name}",
                             "actions": source.run.old.WARMUP.tolist() + history + actions})
            expected.append(trace)
        for item in rows:
            model_delta = np.asarray(item["model_position"]) - item["model_center"]
            physical_delta = np.asarray(item["physical_position"]) - item["physical_nominal_position"]
            np.testing.assert_allclose(item["model_delta"], model_delta, rtol=0, atol=0)
            np.testing.assert_allclose(item["physical_delta"], physical_delta, rtol=0, atol=0)
            if item["linear_vs_model"] != source.compare(item["model_linear_delta"], model_delta):
                raise ValueError("linear metrics mismatch")
            if item["model_vs_physical"] != source.compare(model_delta, physical_delta):
                raise ValueError("physical metrics mismatch")
    if max_j_error != 0.0 or len(requests) != data["physical_requests"]:
        raise ValueError("response or request count mismatch")
    with tempfile.TemporaryDirectory(prefix="pc1_rgb_candidate_audit_") as directory:
        temp = Path(directory)
        req, xml, out = temp / "request.json", temp / "scene.xml", temp / "out.json"
        req.write_text(json.dumps({"rows": requests}, allow_nan=False))
        xml.write_text(source.official_xml())
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
                   "physical_replays": len(requests), "candidate_rows": len(data["rows"]),
                   "max_jacobian_reconstruction_error": max_j_error}, stream, allow_nan=False, indent=2)
    print(f"Replayed {len(requests)} physical paths and reconstructed 12 response Jacobians.", flush=True)


if __name__ == "__main__":
    main()
