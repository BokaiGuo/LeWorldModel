"""Compare sealed RGB candidate response to fixed-history physical replay."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_multidirection_rgb_fixed_terminal_dev_20261103_codex as policy
from experiments import pc_multidirection_rgb_fixed_terminal_replay_20261103_codex as replay
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

run, backend = policy.runner, replay.backend
PROTOCOL = run.ROOT.parent / "docs/PC1_RGB_MULTIDIRECTION_CANDIDATE_RESPONSE_PROTOCOL_20261103_CODEX.md"
OUTPUT = policy.PREFIX.with_name(policy.PREFIX.name + "_candidate_response_physical_diag.json")
EPS = 1e-8


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    error = float(np.linalg.norm(a - b))
    return {"absolute_error": error, "reference_norm": nb, "candidate_norm": na,
            "relative_error": error / nb if nb > EPS else None,
            "cosine": float(np.dot(a, b) / (na * nb)) if na > EPS and nb > EPS else None}


def trace_prediction(trace, actions):
    wanted = np.asarray(actions, dtype=float)
    matches = []
    for batch in trace:
        requested = np.asarray(batch["requested_actions"])
        if requested.shape[1:] != wanted.shape:
            continue
        for i in np.flatnonzero(np.all(requested == wanted, axis=(1, 2))):
            matches.append((np.asarray(batch["positions"][i]), float(batch["scores"][i])))
    if not matches:
        raise ValueError("candidate was not scored")
    spread = {"count": len(matches),
              "max_position_difference": max(float(np.linalg.norm(x[0] - matches[0][0])) for x in matches),
              "max_score_difference": max(abs(x[1] - matches[0][1]) for x in matches)}
    return *matches[0], spread


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    run.PREFIX, run.ARMS = policy.PREFIX, policy.ARMS
    saved = json.loads(replay.OUTPUT.read_text())
    sealed = {row["root"]: row for row in saved["rows"] if row["arm"] == "multidirection"}
    paths = [Path(__file__), PROTOCOL, replay.OUTPUT, backend.WORKER]
    bindings, requests = [], []
    for root in range(4):
        decisions, execution, images = run.paths(root, "multidirection")
        receipt = json.loads(execution.read_text())
        if sealed[root]["execution_sha256"] != sha(execution) or sealed[root]["images_sha256"] != sha(images):
            raise ValueError("execution seal mismatch")
        paths.extend([execution, images, *decisions])
        history = []
        for step, path in enumerate(decisions):
            decision = json.loads(path.read_text())
            if sealed[root]["decision_sha256"][step] != sha(path):
                raise ValueError("decision seal mismatch")
            result = decision["result"]
            response, trace = result["response"], result["prediction_trace"]
            nominal = np.asarray(response["nominal"], dtype=float)
            jacobian = np.asarray(response["jacobian"], dtype=float)
            if nominal.shape != (5, 2) or jacobian.shape != (2, 10):
                raise ValueError("invalid response geometry")
            center, center_score, center_spread = trace_prediction(trace, nominal)
            np.testing.assert_array_equal(center, response["center"])
            candidates = []
            for index, item in enumerate(result["first_bank"]):
                actions = np.asarray(item["prefix"], dtype=float)
                position, score, spread = trace_prediction(trace, actions)
                if score != result["prefix_scores"][index]:
                    raise ValueError("candidate score mismatch")
                candidates.append({"name": item["name"], "actions": actions.tolist(),
                                   "model_position": position.tolist(), "model_score": score, "repeat_spread": spread,
                                   "model_linear_delta": (jacobian @ (actions - nominal).reshape(10)).tolist()})
            entries = [("nominal", nominal.tolist())] + [(f"candidate{i}", c["actions"]) for i, c in enumerate(candidates)]
            for name, actions in entries:
                requests.append({"start": list(run.old.STARTS[root]), "policy": f"{root}-{step}-{name}",
                                 "actions": run.old.WARMUP.tolist() + history + actions})
            bindings.append({"root": root, "step": step, "decision_sha256": sha(path),
                             "prefix_length": 15 + len(history), "nominal": nominal.tolist(),
                             "model_center": center.tolist(), "model_center_score": center_score, "center_spread": center_spread,
                             "jacobian": jacobian.tolist(), "candidates": candidates})
            history.extend(receipt["rows"][step]["actions"])
    hashes = {str(p): sha(p) for p in paths}
    with tempfile.TemporaryDirectory(prefix="pc1_rgb_multidirection_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp / "request.json", tmp / "scene.xml", tmp / "out.json"
        req.write_text(json.dumps({"rows": requests}, allow_nan=False))
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
        raise ValueError("backend or replay count mismatch")
    goal, offset, rows = np.asarray([1.8, 2.8]), 0, []
    for binding in bindings:
        n = len(binding["candidates"])
        batch = physical["rows"][offset:offset + 1 + n]
        offset += 1 + n
        root, step, prefix = binding["root"], binding["step"], binding["prefix_length"]
        for index, item in enumerate(batch):
            label = "nominal" if index == 0 else f"candidate{index - 1}"
            if item["policy"] != f"{root}-{step}-{label}" or len(item["trace"]) != prefix + 6:
                raise ValueError("physical action clock mismatch")
            for observed, original in zip(item["trace"][:prefix + 1], sealed[root]["trace"][:prefix + 1], strict=True):
                if observed["state"] != original["state"] or observed["time"] != original["time"]:
                    raise ValueError("physical history mismatch")
        base = np.asarray(batch[0]["trace"][-1]["state"][:2])
        base_distance = float(np.linalg.norm(base - goal))
        for index, candidate in enumerate(binding["candidates"]):
            physical_pos = np.asarray(batch[index + 1]["trace"][-1]["state"][:2])
            model_delta = np.asarray(candidate["model_position"]) - binding["model_center"]
            physical_delta = physical_pos - base
            physical_distance = float(np.linalg.norm(physical_pos - goal))
            model_cost_delta = candidate["model_score"] - binding["model_center_score"]
            physical_cost_delta = physical_distance - base_distance
            rows.append({"root": root, "step": step, "decision_sha256": binding["decision_sha256"],
                         **candidate, "nominal": binding["nominal"], "model_center": binding["model_center"],
                         "model_delta": model_delta.tolist(), "center_spread": binding["center_spread"],
                         "physical_nominal_position": base.tolist(),
                         "physical_position": physical_pos.tolist(), "physical_delta": physical_delta.tolist(),
                         "linear_vs_model": compare(candidate["model_linear_delta"], model_delta),
                         "model_vs_physical": compare(model_delta, physical_delta),
                         "model_score_delta": model_cost_delta, "physical_distance_delta": physical_cost_delta,
                         "physical_nominal_distance": base_distance, "physical_distance": physical_distance,
                         "model_rank_sign": int(np.sign(model_cost_delta)),
                         "physical_rank_sign": int(np.sign(physical_cost_delta)),
                         "nominal_trace": batch[0]["trace"], "candidate_trace": batch[index + 1]["trace"]})
    if len(bindings) != 12 or offset != len(requests) or hashes != {str(p): sha(p) for p in paths}:
        raise ValueError("panel or dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": hashes, "physical_requests": len(requests), "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} candidate response comparisons across 12 contexts.", flush=True)


if __name__ == "__main__":
    main()
