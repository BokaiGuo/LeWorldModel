"""Audit fixed native-feedback actions against independent backend replays."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/a0_pointmaze_native_feedback_time_20261102_DEV01.json"
SOURCE_SHA = "0a94cb02813e916cdb5190e9a578fc0314a7e3e7a808479d5a4f0b92bc96a2d8"
OUTPUT = ROOT / "results/a0_pointmaze_feedback_endpoint_audit_20261102_DEV01.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backend(request: Path, xml: Path, response: Path) -> None:
    import mujoco_py

    with np.load(request, allow_pickle=False) as data:
        sources, sequences = data["sources"], data["sequences"]
    model = mujoco_py.load_model_from_xml(xml.read_text())
    ends, hashes5, hashes10 = [], [], []
    for state, actions in zip(sources, sequences):
        sim = mujoco_py.MjSim(model)
        sim.render(224, 224)
        camera = sim.render_contexts[0].cam
        camera.azimuth, camera.elevation = 90, -90
        camera.lookat[0], camera.lookat[1] = 3.0, 3.0
        sim.data.qpos[:], sim.data.qvel[:] = state[:2], state[2:]
        sim.forward()
        image_hashes = []
        for index, action in enumerate(actions):
            sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
            sim.forward()
            sim.data.ctrl[:] = action
            for _ in range(5):
                sim.step()
            sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
            if index in (4, 9):
                image_hashes.append(hashlib.sha256(sim.render(224, 224).tobytes()).hexdigest())
        ends.append(np.r_[sim.data.qpos[:2], sim.data.qvel[:2]].copy())
        hashes5.append(image_hashes[0])
        hashes10.append(image_hashes[1])
    np.savez_compressed(response, endpoints=np.stack(ends),
                        rgb5=np.asarray(hashes5), rgb10=np.asarray(hashes10))


def main() -> None:
    import torch
    from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
    from experiments.pointmaze_native_rgb_bridge_20261102 import LEGACY, RUNTIME
    from experiments.spectral_reachability import pointmaze_physical_evaluator as physical

    if OUTPUT.exists() or digest(SOURCE) != SOURCE_SHA:
        raise RuntimeError("output exists or sealed source changed")
    source = json.loads(SOURCE.read_text())
    if source["states_sha256"] != digest(DATA / "states.pth"):
        raise RuntimeError("reset state source differs from feedback run")
    # Original PointMazeDataset explicitly applies .float() before reset/goal.
    states = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    bank = np.asarray(source["bank"], dtype=np.float32)
    if bank.shape != (8, 2) or not np.isfinite(bank).all() or np.any(np.abs(bank) > 1):
        raise ValueError("invalid candidate bank")
    starts, sequences, roster = [], [], []
    for row in source["rows"]:
        root = int(row["root"])
        for branch in row["branches"]:
            first, tail = int(branch["first"]), int(branch["chosen_tail"])
            if not (0 <= first < 8 and 0 <= tail < 8):
                raise ValueError("invalid selected action index")
            starts.append(states[root, 0])
            sequences.append(np.concatenate([np.repeat(bank[first][None], 5, axis=0),
                                             np.repeat(bank[tail][None], 5, axis=0)]))
            roster.append((root, branch))
    starts, sequences = np.asarray(starts, dtype=np.float64), np.asarray(sequences, dtype=np.float64)
    if starts.shape != (8, 4) or sequences.shape != (8, 10, 2):
        raise RuntimeError("unexpected fixed panel")
    with tempfile.TemporaryDirectory(prefix="pc_feedback_endpoint_") as tmp:
        folder = Path(tmp)
        request, xml, response = folder / "request.npz", folder / "scene.xml", folder / "output.npz"
        np.savez_compressed(request, sources=starts, sequences=sequences)
        xml_text = official_xml()
        xml.write_text(xml_text)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        proc = subprocess.run([str(LEGACY / "bin/python"), str(Path(__file__).resolve()), "--backend",
                               str(request), str(xml), str(response)], capture_output=True, text=True,
                              env=env, timeout=90)
        if proc.returncode or not response.is_file():
            raise RuntimeError(f"legacy replay failed: {proc.stderr[-3000:]}")
        with np.load(response, allow_pickle=False) as payload:
            legacy, rgb5, rgb10 = (payload[key].copy() for key in ("endpoints", "rgb5", "rgb10"))
    standard = physical.replay_position_endpoints(starts, sequences)
    rows = []
    for i, (root, branch) in enumerate(roster):
        goal = states[root, 10, :2].astype(np.float64)
        original_position = np.asarray(branch["physical_position_eval_only"], dtype=np.float64)
        if not np.array_equal(standard[i, :2], original_position):
            raise RuntimeError("position backend no longer reproduces saved selected endpoint")
        rows.append({
            "root": root, "method": branch["method"], "first": branch["first"], "tail": branch["chosen_tail"],
            "rgb5_matches_source": str(rgb5[i]) == branch["rgb5_sha256"],
            "rgb10_matches_source": str(rgb10[i]) == branch["rgb10_sha256"],
            "legacy_endpoint_eval_only": legacy[i].tolist(),
            "position_backend_endpoint_eval_only": standard[i].tolist(),
            "position_delta_l2": float(np.linalg.norm(legacy[i, :2] - standard[i, :2])),
            "velocity_delta_l2": float(np.linalg.norm(legacy[i, 2:] - standard[i, 2:])),
            "legacy_goal_distance_eval_only": float(np.linalg.norm(legacy[i, :2] - goal)),
            "position_backend_goal_distance_eval_only": float(np.linalg.norm(standard[i, :2] - goal)),
        })
    report = {
        "schema": "a0_pointmaze_feedback_endpoint_audit.v1",
        "scope": "post-outcome replay of eight fixed development action chains; not independent P1",
        "reset_and_goal_dtype": "source float64 cast to float32 exactly as PointMazeDataset, then evaluated as float64",
        "feedback_result_sha256": SOURCE_SHA, "states_sha256": digest(DATA / "states.pth"),
        "xml_sha256": hashlib.sha256(xml_text.encode()).hexdigest(),
        "runtime_sha256": digest(RUNTIME / "lib/libmujoco.so.2.1.2"),
        "code_sha256": digest(Path(__file__)),
        "physical_backend": physical.position_backend_receipt(),
        "position_tolerance_predeclared": None, "rows": rows,
    }
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest(),
                      "matched_rgb5": sum(row["rgb5_matches_source"] for row in rows),
                      "matched_rgb10": sum(row["rgb10_matches_source"] for row in rows),
                      "max_position_delta": max(row["position_delta_l2"] for row in rows)}))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--backend":
        backend(*(Path(value) for value in sys.argv[2:]))
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("unexpected arguments")
