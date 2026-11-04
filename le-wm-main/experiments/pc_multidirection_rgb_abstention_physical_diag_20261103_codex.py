"""Post-hoc physical replay of sealed RGB precomparison versus zero-prefix chains."""
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
PROTOCOL = run.ROOT.parent / "docs/PC_MULTIDIRECTION_RGB_ABSTENTION_PHYSICAL_DIAG_PROTOCOL_20261103_CODEX.md"
OUTPUT = policy.PREFIX.with_name(policy.PREFIX.name + "_abstention_physical_diag.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    run.PREFIX, run.ARMS = policy.PREFIX, policy.ARMS
    physical = json.loads(replay.OUTPUT.read_text())
    physical_by_root = {r["root"]: r for r in physical["rows"] if r["arm"] == "multidirection"}
    dependencies = {str(p): sha(p) for p in (Path(__file__), PROTOCOL, replay.OUTPUT, backend.WORKER)}
    requests, bindings = [], []
    for root in range(4):
        decisions, execution, images = run.paths(root, "multidirection")
        receipt = json.loads(execution.read_text())
        saved = physical_by_root[root]
        if saved["execution_sha256"] != sha(execution) or saved["images_sha256"] != sha(images):
            raise ValueError("physical receipt mismatch")
        history = []
        for step, path in enumerate(decisions):
            decision = json.loads(path.read_text())
            result = decision["result"]
            if saved["decision_sha256"][step] != sha(path):
                raise ValueError("decision seal mismatch")
            before = np.asarray(result["precomparison"]["chain"], dtype=float)
            zero = np.asarray(result["comparison_chain"], dtype=float)
            expected = before.copy()
            expected[:5] = 0
            if before.shape != (15 - 5 * step, 2) or not np.array_equal(expected, zero):
                raise ValueError("zero-prefix comparison mismatch")
            if result["abstained"] != (result["comparison_score"] <= result["precomparison"]["terminal_score"]):
                raise ValueError("abstention decision mismatch")
            for label, chain in (("before", before), ("zero", zero)):
                requests.append({"start": list(run.old.STARTS[root]), "policy": f"{root}-{step}-{label}",
                                 "actions": run.old.WARMUP.tolist() + history + chain.tolist()})
            bindings.append({"root": root, "step": step, "decision_sha256": sha(path),
                             "model_before": result["precomparison"]["terminal_score"],
                             "model_zero": result["comparison_score"], "abstained": result["abstained"],
                             "prefix_length": 15 + len(history)})
            history.extend(receipt["rows"][step]["actions"])
    with tempfile.TemporaryDirectory(prefix="pc_rgb_counterfactual_") as directory:
        temp = Path(directory)
        req, xml, out = temp / "request.json", temp / "scene.xml", temp / "out.json"
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
        counterfactual = json.loads(out.read_text())
    if counterfactual["binding"] != "2.1.2.14" or len(counterfactual["rows"]) != 24:
        raise ValueError("unexpected replay backend or panel")
    rows = []
    goal = np.asarray([1.8, 2.8])
    for index, binding in enumerate(bindings):
        before, zero = counterfactual["rows"][2 * index:2 * index + 2]
        root, step, prefix = binding["root"], binding["step"], binding["prefix_length"]
        saved_trace = physical_by_root[root]["trace"]
        for name, item in (("before", before), ("zero", zero)):
            if item["policy"] != f"{root}-{step}-{name}" or len(item["trace"]) != 31:
                raise ValueError("counterfactual clock mismatch")
            for observed, original in zip(item["trace"][:prefix + 1], saved_trace[:prefix + 1], strict=True):
                if observed["state"] != original["state"] or observed["time"] != original["time"]:
                    raise ValueError("prior execution differs from sealed replay")
        distance = lambda trace, t: float(np.linalg.norm(np.asarray(trace[t]["state"][:2]) - goal))
        d_before, d_zero = distance(before["trace"], -1), distance(zero["trace"], -1)
        model_difference = binding["model_zero"] - binding["model_before"]
        physical_difference = d_zero - d_before
        selected_trace = zero["trace"] if binding["abstained"] else before["trace"]
        rows.append({**binding, "physical_before": d_before, "physical_zero": d_zero,
                     "model_zero_minus_before": model_difference,
                     "physical_zero_minus_before": physical_difference,
                     "model_sign": int(np.sign(model_difference)), "physical_sign": int(np.sign(physical_difference)),
                     "actual_prefix_distance_change": distance(selected_trace, prefix + 5) - distance(selected_trace, prefix),
                     "before_trace": before["trace"], "zero_trace": zero["trace"]})
    if dependencies != {str(p): sha(p) for p in (Path(__file__), PROTOCOL, replay.OUTPUT, backend.WORKER)}:
        raise ValueError("dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows}, stream, allow_nan=False, separators=(",", ":"))
    print("Sealed 12 post-hoc zero-prefix physical comparisons.", flush=True)


if __name__ == "__main__":
    main()
