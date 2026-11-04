"""Replay every sealed CP5 feedback action chain using both physics backends."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from experiments.a0_pointmaze_feedback_endpoint_audit_20261102 import digest
from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
from experiments.pointmaze_native_rgb_bridge_20261102 import LEGACY, RUNTIME
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/a1_pointmaze_spectral_cp5_replan_20261102_DEV01.json"
SOURCE_SHA = "8f0bba93fa1f093cae4550da317d9b20a6cc018756e37402419f384eabdf6446"
OUTPUT = ROOT / "results/a1_pointmaze_cp5_replan_endpoint_audit_20261102_DEV01.json"
BACKEND = ROOT / "experiments/a0_pointmaze_feedback_endpoint_audit_20261102.py"


def make_sequences(report, states):
    starts, actions, roster = [], [], []
    if [r["root"] for r in report["rows"]] != [1880, 1881, 1882, 1883]:
        raise ValueError("root roster changed")
    for row in report["rows"]:
        root = int(row["root"])
        if [b["method"] for b in row["branches"]] != ["spectral", "random", "axis", "zero"]:
            raise ValueError("branch roster changed")
        for branch in row["branches"]:
            prefix = np.asarray(branch["prefix_action"], dtype=np.float32)
            tail = np.asarray(branch["tail_action"], dtype=np.float32)
            if (prefix.shape != (2,) or tail.shape != (2,) or
                    not np.isfinite(prefix).all() or not np.isfinite(tail).all() or
                    np.any(np.abs(prefix) > 1) or np.any(np.abs(tail) > 1)):
                raise ValueError("invalid action")
            starts.append(states[root, 0])
            actions.append(np.concatenate((np.repeat(prefix[None], 5, axis=0),
                                           np.repeat(tail[None], 5, axis=0))))
            roster.append((root, branch))
    return np.asarray(starts, dtype=np.float64), np.asarray(actions, dtype=np.float64), roster


def main():
    if OUTPUT.exists() or digest(SOURCE) != SOURCE_SHA:
        raise RuntimeError("output exists or sealed source changed")
    report = json.loads(SOURCE.read_text())
    original_path = ROOT / "results/a1_pointmaze_native_spectral_20261102_DEV01.json"
    if digest(original_path) != report["source_sha256"]:
        raise RuntimeError("original prefix source changed")
    if digest(DATA / "states.pth") != json.loads(original_path.read_text())["states_sha256"]:
        raise RuntimeError("reset states changed")
    states = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    starts, actions, roster = make_sequences(report, states)
    if starts.shape != (16, 4) or actions.shape != (16, 10, 2):
        raise RuntimeError("unexpected action panel")
    with tempfile.TemporaryDirectory(prefix="pc_cp5_endpoint_") as tmp:
        directory = Path(tmp)
        request, xml, response = (directory / item for item in ("request.npz", "scene.xml", "response.npz"))
        np.savez_compressed(request, sources=starts, sequences=actions)
        xml_text = official_xml()
        xml.write_text(xml_text)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        proc = subprocess.run([str(LEGACY / "bin/python"), str(BACKEND), "--backend",
                               str(request), str(xml), str(response)],
                              capture_output=True, text=True, env=env, timeout=120)
        if proc.returncode or not response.is_file():
            raise RuntimeError(f"legacy replay failed: {proc.stderr[-1500:]}")
        with np.load(response, allow_pickle=False) as payload:
            legacy, rgb5, rgb10 = (payload[key].copy() for key in ("endpoints", "rgb5", "rgb10"))
    standard = physical.replay_position_endpoints(starts, actions)
    rows = []
    for i, (root, branch) in enumerate(roster):
        saved = np.asarray(branch["endpoint_eval_only"], dtype=np.float64)
        rows.append({
            "root": root, "method": branch["method"],
            "rgb5_matches": str(rgb5[i]) == branch["rgb5_sha256"],
            "rgb10_matches": str(rgb10[i]) == branch["rgb10_sha256"],
            "saved_position_backend_matches": bool(np.array_equal(standard[i], saved)),
            "legacy_state": legacy[i].tolist(), "position_backend_state": standard[i].tolist(),
            "position_delta_l2": float(np.linalg.norm(legacy[i, :2] - standard[i, :2])),
            "velocity_delta_l2": float(np.linalg.norm(legacy[i, 2:] - standard[i, 2:])),
        })
    result = {
        "schema": "a1_pointmaze_cp5_replan_endpoint_audit.v1",
        "scope": "post-outcome deterministic replay of fixed exposed-root panel; no independent P1",
        "source_sha256": SOURCE_SHA, "states_sha256": digest(DATA / "states.pth"),
        "reset_dtype": "archive converted to float32 as in dataset, then float64",
        "legacy_backend_code_sha256": digest(BACKEND), "code_sha256": digest(Path(__file__)),
        "xml_sha256": hashlib.sha256(xml_text.encode()).hexdigest(),
        "legacy_runtime_sha256": digest(RUNTIME / "lib/libmujoco.so.2.1.2"),
        "position_backend": physical.position_backend_receipt(),
        "tolerance_preregistered": None, "rows": rows,
    }
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest(),
                      "rgb5": sum(r["rgb5_matches"] for r in rows),
                      "rgb10": sum(r["rgb10_matches"] for r in rows),
                      "saved": sum(r["saved_position_backend_matches"] for r in rows),
                      "max_position_delta": max(r["position_delta_l2"] for r in rows),
                      "max_velocity_delta": max(r["velocity_delta_l2"] for r in rows)}))


if __name__ == "__main__":
    main()
