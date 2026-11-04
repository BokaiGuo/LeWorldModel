"""Post-outcome legacy-physics replay of sealed history-CEM trajectories."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a1_pointmaze_rgb_history_cem_feedback_20261103 import (
    NOISE_SOURCE, NOISE_SHA, OUTPUT as SOURCE, SOURCE as PARENT, SOURCE_SHA as PARENT_SHA,
)
from experiments.pointmaze_native_rgb_bridge_20261102 import LEGACY, RUNTIME
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import DEFAULT_DATASET, ROOT, sha256_file

SOURCE_SHA = "a5cf40bad026652440dff1c123a489ae9f80deeb5102c8d6a8b79f29ad3d3932"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_HISTORY_CEM_NATIVE_ENDPOINT_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_history_cem_native_endpoint_20261103_DEV01.json"
BACKEND = ROOT / "experiments/a0_pointmaze_feedback_endpoint_audit_20261102.py"
WRAPPER = ROOT / "external/jepa-wms/evals/simu_env_planning/envs/pointmaze_gym_wrap.py"


def evaluate(goal, actual):
    goal, actual = np.asarray(goal, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if goal.shape != (4,) or actual.shape != (4,) or not np.isfinite(goal).all() or not np.isfinite(actual).all():
        raise ValueError("expected finite goal and actual state")
    distance = float(np.linalg.norm(goal[:2] - actual[:2]))
    return {"success": bool(distance < .5), "position_distance": distance,
            "state_distance": float(np.linalg.norm(goal - actual))}


def main():
    if OUTPUT.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (PARENT, PARENT_SHA), (NOISE_SOURCE, NOISE_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    source, parent, noise = (json.loads(p.read_text()) for p in (SOURCE, PARENT, NOISE_SOURCE))
    if sha256_file(DEFAULT_DATASET / "states.pth") != noise["states_sha256"]:
        raise RuntimeError("state archive changed")
    if sha256_file(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]:
        raise RuntimeError("source image arrays changed")
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    starts, sequences, roster = [], [], []
    for row, upstream in zip(source["rows"], parent["rows"], strict=True):
        root = row["root"]
        if root != upstream["root"]:
            raise RuntimeError("root changed")
        for branch in row["branches"]:
            prefix, tail = (np.asarray(branch[key], dtype=np.float32) for key in ("prefix", "tail"))
            if prefix.shape != (2,) or tail.shape != (1, 2) or not np.isfinite(prefix).all() or not np.isfinite(tail).all():
                raise ValueError("invalid recorded action")
            if np.max(np.abs(prefix)) > 1 or np.max(np.abs(tail)) > 1:
                raise ValueError("action exceeds bounds")
            starts.append(states[root, 0])
            sequences.append(np.concatenate((np.repeat(prefix[None], 5, axis=0), np.repeat(tail, 5, axis=0))))
            roster.append((root, branch))
    starts, sequences = np.asarray(starts, np.float64), np.asarray(sequences, np.float64)
    if starts.shape != (16, 4) or sequences.shape != (16, 10, 2):
        raise RuntimeError("unexpected sealed panel")
    xml = official_xml()
    with tempfile.TemporaryDirectory(prefix="pc_history_endpoint_") as tmp:
        folder = Path(tmp)
        request, scene, response = (folder / name for name in ("request.npz", "scene.xml", "response.npz"))
        np.savez_compressed(request, sources=starts, sequences=sequences)
        scene.write_text(xml)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        proc = subprocess.run([str(LEGACY / "bin/python"), str(BACKEND), "--backend",
                               str(request), str(scene), str(response)],
                              capture_output=True, text=True, env=env, timeout=120)
        if proc.returncode or not response.is_file():
            raise RuntimeError(f"legacy replay failed: {proc.stderr[-1500:]}")
        with np.load(response, allow_pickle=False) as data:
            legacy, rgb5, rgb10 = (data[k].copy() for k in ("endpoints", "rgb5", "rgb10"))
    standard = physical.replay_position_endpoints(starts, sequences)
    rows = []
    for i, (root, branch) in enumerate(roster):
        if str(rgb5[i]) != branch["rgb5_sha256"] or str(rgb10[i]) != branch["rgb10_sha256"]:
            raise RuntimeError(f"native RGB replay mismatch: {root}/{branch['prefix_method']}/{branch['mode']}")
        if not np.array_equal(standard[i], np.asarray(branch["endpoint_eval_only"], dtype=np.float64)):
            raise RuntimeError("stored position endpoint mismatch")
        goal = states[root, 10].astype(np.float64)
        rows.append({"root": root, "prefix_method": branch["prefix_method"], "mode": branch["mode"],
                     "goal_eval_only": goal.tolist(), "legacy_state_eval_only": legacy[i].tolist(),
                     "position_state_eval_only": standard[i].tolist(),
                     "legacy_minus_position_l2": float(np.linalg.norm(legacy[i, :2] - standard[i, :2])),
                     "velocity_delta_l2": float(np.linalg.norm(legacy[i, 2:] - standard[i, 2:])),
                     "rgb5_matches": True, "rgb10_matches": True, "saved_position_matches": True,
                     "wrapper_eval": evaluate(goal, legacy[i])})
    report = {"schema": "a1_pointmaze_history_cem_native_endpoint.v1", "source_sha256": SOURCE_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "legacy_backend_sha256": sha256_file(BACKEND), "wrapper_sha256": sha256_file(WRAPPER),
              "legacy_runtime_sha256": sha256_file(RUNTIME / "lib/libmujoco.so.2.1.2"),
              "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(), "states_sha256": noise["states_sha256"],
              "position_backend": physical.position_backend_receipt(),
              "scope": "post-outcome deterministic legacy replay, no early stopping or independent P1", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest(),
                      "successes": sum(r["wrapper_eval"]["success"] for r in rows),
                      "max_position_delta": max(r["legacy_minus_position_l2"] for r in rows)}))


if __name__ == "__main__":
    main()
