"""Prepare and check privileged exact-dynamics fixed-prefix diagnostic."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_pointmaze_cp12_rgb_history_planning_20261103_codex as prior
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import LEGACY, RUNTIME, _scene

ROOT, h = prior.ROOT, prior.h
SOURCE = ROOT / "results/pc_spectral_prefix_same_action_physical_20261103_CODEX/result.json"
AUDIT = SOURCE.with_name("audit.json")
SEALED = ROOT / "results/pc_spectral_prefix_common_suffix_dev_20261103_CODEX.json"
PROTOCOL = ROOT.parent / "docs/PC_SPECTRAL_PREFIX_EXACT_DYNAMICS_CEILING_PROTOCOL_20261103_CODEX.md"
WORKER = ROOT / "experiments/spectral_reachability/pc_spectral_prefix_exact_worker_20261103_codex.py"
OUTPUT = ROOT / "results/pc_spectral_prefix_exact_ceiling_20261103_CODEX.json"
ROSTER = ((0, "r8343_a0_p2"), (8, "r8345_a0_p2"))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit, source, sealed = (json.loads(path.read_text()) for path in (AUDIT, SOURCE, SEALED))
    if audit["result_sha256"] != prior.sha(SOURCE) or source["source_sha256"] != prior.sha(SEALED):
        raise ValueError("input receipt mismatch")
    _, archive, names = prior.files()
    starts, prefixes, full_actions = [], [], []
    with np.load(prior.SOURCE.with_suffix(".npz"), allow_pickle=False) as inp:
        for (idx, name), row in zip(ROSTER, source["rows"], strict=True):
            if names[idx] != name or row["root"] != name or len(row["branches"]) != 8:
                raise ValueError("roster mismatch")
            for branch in row["branches"]:
                path = Path(branch["trajectory_path"])
                if prior.sha(path) != branch["trajectory_sha256"]:
                    raise ValueError("trajectory hash mismatch")
                with np.load(path, allow_pickle=False) as data:
                    prefixes.append(data["actions"][:17].tolist())
                    full_actions.append(data["actions"].copy())
                starts.append(inp["starts"][idx].tolist())
    endpoints = h.physical.replay_position_endpoints(np.asarray(starts), np.asarray(full_actions))
    for i, endpoint in enumerate(endpoints):
        reference = source["rows"][i//8]["branches"][i%8]["physical_endpoints"]["22"]
        np.testing.assert_allclose(endpoint, reference, rtol=0, atol=1e-10)
    request = {"starts": starts, "prefixes": prefixes, "goal": list(map(float, h.GOAL[:2])),
               "action_mean": archive["provenance"]["action_mean"],
               "action_std": archive["provenance"]["action_std"]}
    print("Offline exact-dynamics equal-suffix search; privileged state never enters RGB selection.", flush=True)
    with tempfile.TemporaryDirectory(prefix="pc_exact_ceiling_") as temp:
        folder = Path(temp)
        request_path, xml_path, result_path = folder / "request.json", folder / "scene.xml", folder / "result.json"
        request_path.write_text(json.dumps(request, allow_nan=False))
        xml_path.write_bytes(_scene())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                            "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(LEGACY / "bin/python"), str(WORKER), str(request_path), str(xml_path), str(result_path)],
                       env=env, check=True, timeout=120)
        result = json.loads(result_path.read_text())
    for i, row in enumerate(result["rows"]):
        reference = source["rows"][i//8]["branches"][i%8]
        for tick, value in ((12, row["prefix_distances"][0]), (17, row["prefix_distances"][-1])):
            np.testing.assert_allclose(value, reference["physical_distances"][str(tick)], rtol=0, atol=1e-10)
        row["root"] = source["rows"][i//8]["root"]
        row["name"] = reference["name"]
        row["original_model_suffix_distance"] = reference["physical_distances"]["22"]
        row["non_immediate_progress"] = row["prefix_distances"][-1] >= row["prefix_distances"][0]
        controls = result["rows"][8*(i//8):8*(i//8)+2]
        row["beats_zero_and_greedy"] = row["optimized_distance"] < min(c["optimized_distance"] for c in controls)
    result.update({"schema": "pc_spectral_prefix_exact_ceiling.v1", "protocol_sha256": prior.sha(PROTOCOL),
                   "source_sha256": prior.sha(SOURCE), "audit_sha256": prior.sha(AUDIT),
                   "worker_sha256": prior.sha(WORKER), "runner_sha256": prior.sha(Path(__file__)),
                   "privileged_evaluation_only": True, "independent_confirmation": False})
    with OUTPUT.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
