"""Check successor-conditioned RGB candidate predictions against physical replay."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc1_rgb_multidirection_candidate_response_20261103_codex as prior

run, policy, backend = prior.run, prior.policy, prior.backend
PROTOCOL = run.ROOT.parent / "docs/PC1_RGB_MULTIDIRECTION_SUCCESSOR_RESPONSE_PROTOCOL_20261103_CODEX.md"
OUTPUT = policy.PREFIX.with_name(policy.PREFIX.name + "_successor_response_physical_diag.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    previous = json.loads(prior.OUTPUT.read_text())
    if previous["dependencies"] != {path: sha(Path(path)) for path in previous["dependencies"]}:
        raise ValueError("first-response evidence drift")
    parent_traces = {(x["root"], x["name"]): x["candidate_trace"]
                     for x in previous["rows"] if x["step"] == 0}
    run.PREFIX, run.ARMS = policy.PREFIX, policy.ARMS
    paths, requests, bindings = [Path(__file__), PROTOCOL, prior.OUTPUT, backend.WORKER], [], []
    for root in range(4):
        decisions, execution, images = run.paths(root, "multidirection")
        decision = json.loads(decisions[0].read_text())
        result = decision["result"]
        paths.extend((decisions[0], execution, images))
        for successor in result["successor_response"]:
            parent_index = successor["parent_index"]
            parent_item = result["first_bank"][parent_index]
            parent = np.asarray(parent_item["prefix"])
            response = successor["response"]
            nominal, jacobian = np.asarray(response["nominal"]), np.asarray(response["jacobian"])
            if parent.shape != (5, 2) or nominal.shape != (5, 2) or jacobian.shape != (2, 10):
                raise ValueError("invalid response geometry")
            center, center_score, center_repeat = prior.trace_prediction(
                result["prediction_trace"], np.concatenate((parent, nominal)))
            np.testing.assert_array_equal(center, response["center"])
            candidates = []
            for index, item in enumerate(successor["bank"]):
                tail = np.asarray(item["prefix"])
                position, score, repeat = prior.trace_prediction(result["prediction_trace"], np.concatenate((parent, tail)))
                np.testing.assert_array_equal(position, successor["positions"][index])
                if score != successor["scores"][index]:
                    raise ValueError("successor score mismatch")
                candidates.append({"name": item["name"], "tail": tail.tolist(), "model_position": position.tolist(),
                                   "model_score": score, "model_linear_delta": (jacobian @ (tail - nominal).reshape(10)).tolist(),
                                   "repeat_spread": repeat})
            branch = len(bindings)
            choices = [("nominal", nominal.tolist())] + [(f"candidate{i}", item["tail"]) for i, item in enumerate(candidates)]
            for label, tail in choices:
                requests.append({"start": list(run.old.STARTS[root]), "policy": f"{branch}-{label}",
                                 "actions": run.old.WARMUP.tolist() + parent.tolist() + tail})
            bindings.append({"root": root, "branch": branch, "parent_index": parent_index,
                             "parent_name": parent_item["name"], "parent": parent.tolist(), "nominal": nominal.tolist(),
                             "center": center.tolist(), "center_score": center_score, "center_repeat": center_repeat,
                             "jacobian": jacobian.tolist(), "retained": successor["retained"], "candidates": candidates})
    hashes = {str(p): sha(p) for p in paths}
    with tempfile.TemporaryDirectory(prefix="pc1_rgb_successor_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp / "request.json", tmp / "scene.xml", tmp / "out.json"
        req.write_text(json.dumps({"rows": requests}, allow_nan=False))
        xml.write_text(prior.official_xml())
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
    rows, offset, goal = [], 0, np.asarray([1.8, 2.8])
    for binding in bindings:
        n = len(binding["candidates"])
        batch = physical["rows"][offset:offset + 1 + n]
        offset += n + 1
        for index, item in enumerate(batch):
            label = "nominal" if index == 0 else f"candidate{index - 1}"
            if item["policy"] != f'{binding["branch"]}-{label}' or len(item["trace"]) != 26:
                raise ValueError("physical clock or label mismatch")
            original = parent_traces[(binding["root"], binding["parent_name"])]
            if item["trace"][:21] != original:
                raise ValueError("parent trajectory mismatch")
        base = np.asarray(batch[0]["trace"][-1]["state"][:2])
        base_dist = float(np.linalg.norm(base - goal))
        candidates = []
        for index, candidate in enumerate(binding["candidates"]):
            pos = np.asarray(batch[index + 1]["trace"][-1]["state"][:2])
            physical_dist = float(np.linalg.norm(pos - goal))
            model_delta = np.asarray(candidate["model_position"]) - binding["center"]
            physical_delta = pos - base
            model_cost_delta = candidate["model_score"] - binding["center_score"]
            physical_cost_delta = physical_dist - base_dist
            candidates.append({**candidate, "model_delta": model_delta.tolist(), "physical_position": pos.tolist(),
                               "physical_delta": physical_delta.tolist(), "physical_distance": physical_dist,
                               "linear_vs_model": prior.compare(candidate["model_linear_delta"], model_delta),
                               "model_vs_physical": prior.compare(model_delta, physical_delta),
                               "model_score_delta": model_cost_delta, "physical_distance_delta": physical_cost_delta,
                               "trace": batch[index + 1]["trace"]})
        model_best = int(np.argmin([c["model_score"] for c in candidates]))
        physical_order = np.argsort([c["physical_distance"] for c in candidates], kind="stable")
        physical_best = int(physical_order[0])
        if model_best not in binding["retained"]:
            raise ValueError("model best missing from retained set")
        rounding = max(float(np.max(np.abs(np.asarray(x) - np.asarray(x, dtype=np.float32))))
                       for x in [binding["parent"], binding["nominal"], *[c["tail"] for c in candidates]])
        rows.append({**binding, "physical_nominal_position": base.tolist(), "physical_nominal_distance": base_dist,
                     "nominal_trace": batch[0]["trace"], "candidates": candidates, "model_best": model_best,
                     "physical_order": physical_order.tolist(), "physical_best": physical_best,
                     "physical_best_retained": physical_best in binding["retained"],
                     "model_best_physical_rank": int(np.flatnonzero(physical_order == model_best)[0]) + 1,
                     "max_float32_rounding": rounding})
    if len(rows) != 8 or offset != len(requests) or hashes != {str(p): sha(p) for p in paths}:
        raise ValueError("panel or dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": hashes, "physical_requests": len(requests), "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} successor branches and {sum(len(r['candidates']) for r in rows)} candidates.", flush=True)


if __name__ == "__main__":
    main()
