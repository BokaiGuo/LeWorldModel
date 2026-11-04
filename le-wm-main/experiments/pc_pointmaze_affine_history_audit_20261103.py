"""Recompute saved RGB histories in a separate legacy process, without planning."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.pc_pointmaze_affine_feedback_cem_20261103 import OUTPUT as RANDOM
from experiments.pc_pointmaze_affine_spectral_cem_20261103 import OUTPUT as SPECTRAL
from experiments.pointmaze_native_rgb_bridge_20261102 import LEGACY, RUNTIME
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import ROOT, DEFAULT_DATASET, sha256_file

WORKER = ROOT / "experiments/pc_pointmaze_history_replay_worker_20261103.py"
OUTPUT = ROOT / "results/pc_pointmaze_affine_history_audit_20261103.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    reports = [(path, json.loads(path.read_text())) for path in (RANDOM, SPECTRAL)]
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).numpy()
    starts, sequences, histories, terminal_images, keys = [], [], [], [], []
    for path, report in reports:
        if sha256_file(path.with_suffix(".npz")) != report["arrays_sha256"]:
            raise ValueError("saved arrays drift")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
            for i, row in enumerate(report["rows"]):
                root = row["root"]
                starts.append(states[root, 0])
                sequences.append(arrays["sequences"][i])
                histories.append(arrays[f"r{root}_history"])
                terminal_images.append(arrays[f"r{root}_rgb10"])
                keys.append({"root": root, "method": path.name})
    with tempfile.TemporaryDirectory(prefix="pc_history_replay_") as tmp:
        request, xml, response = (Path(tmp) / n for n in ("request.npz", "scene.xml", "response.npz"))
        np.savez_compressed(request, sources=np.asarray(starts), sequences=np.asarray(sequences))
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        proc = subprocess.run([str(LEGACY / "bin/python"), str(WORKER), str(request), str(xml), str(response)],
                              capture_output=True, text=True, env=env, timeout=300)
        if proc.returncode or not response.is_file():
            raise RuntimeError(f"legacy replay failed: {proc.stderr[-2000:]}")
        with np.load(response, allow_pickle=False) as data:
            actual = data["histories"].copy()
    expected, terminal = np.asarray(histories), np.asarray(terminal_images)
    if actual.shape != (len(keys), 11, 224, 224, 3) or expected.shape != actual[:, :6].shape:
        raise ValueError("unexpected replay history shape")
    rows = [{**key, "frame0_to_5_exact": [bool(np.array_equal(actual[i, j], expected[i, j]))
                                            for j in range(6)],
             "frame10_exact": bool(np.array_equal(actual[i, 10], terminal[i]))}
            for i, key in enumerate(keys)]
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, replay_histories=actual)
    result = {"schema": "pc_pointmaze_affine_history_audit.v1",
              "source_sha256": {p.name: sha256_file(p) for p, _ in reports},
              "states_sha256": sha256_file(DEFAULT_DATASET / "states.pth"),
              "worker_sha256": sha256_file(WORKER), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "rows": rows,
              "all_match": all(all(r["frame0_to_5_exact"]) and r["frame10_exact"] for r in rows)}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Saved {OUTPUT}: all_match={result['all_match']}", flush=True)
    if not result["all_match"]:
        raise RuntimeError("saved feedback frame replay mismatch")


if __name__ == "__main__":
    main()
