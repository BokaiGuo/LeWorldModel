"""Replay sealed state-assisted action witness through legacy MuJoCo."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SELECT = ROOT / "results/pc_fixed_prefix_state_smoke_20261107_codex.json"
OUT = ROOT / "results/pc_fixed_prefix_state_physical_20261107_codex.json"


def backend(request, xml, response):
    import mujoco_py

    with np.load(request, allow_pickle=False) as data:
        state, actions = data["state"], data["actions"]
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    if abs(float(model.opt.timestep) - .01) > 1e-12:
        raise ValueError("unexpected MuJoCo time step")
    sim = mujoco_py.MjSim(model)
    traces = []
    for chain in actions:
        sim.reset()
        sim.data.qpos[:] = state[:2]
        sim.data.qvel[:] = state[2:]
        sim.forward()
        trace = [np.r_[sim.data.qpos, sim.data.qvel, sim.data.time].copy()]
        for action in chain:
            sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
            sim.forward()
            sim.data.ctrl[:] = action
            for _ in range(5):
                sim.step()
            trace.append(np.r_[sim.data.qpos, sim.data.qvel, sim.data.time].copy())
        traces.append(trace)
    with open(response, "xb") as f:
        np.savez_compressed(f, trace=np.asarray(traces), binding=np.asarray(mujoco_py.__version__),
                            runtime=np.asarray(mujoco_py.get_version()))


def main():
    import hashlib
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    from experiments import pc_pointmaze_multimodal_objective_20261107_codex as source
    from experiments.pc_input_permissions import STATE, file_hash

    if OUT.exists():
        raise FileExistsError(OUT)
    saved = json.loads(SELECT.read_text())
    if saved["track"] != STATE or saved["input_sha256"] != file_hash(source.PUBLIC):
        raise ValueError("track or public input drift")
    for field, path in (("shared_core_sha256", ROOT / "experiments/pc_fixed_prefix_shared_20261107_codex.py"),
                        ("adapter_sha256", ROOT / "experiments/pc_fixed_prefix_state_adapter_20261107_codex.py"),
                        ("config_sha256", source.CONFIG)):
        if saved[field] != file_hash(path):
            raise ValueError(f"selection dependency mismatch {field}")
    actions = np.asarray(saved["actions"], dtype=np.float32)
    if actions.shape != (25, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
        raise ValueError("invalid sealed action chain")
    with np.load(source.PUBLIC, allow_pickle=False) as inputs:
        state = inputs["proprio"][0, 0].copy()
        goal = inputs["goal_proprio"][0, 0].copy()
    if state.shape != (4,) or goal.shape != (4,) or not np.isfinite(state).all() or not np.isfinite(goal).all():
        raise ValueError("invalid explicit state inputs")
    chains = np.stack([actions, np.zeros_like(actions)])
    with tempfile.TemporaryDirectory(prefix="pc_state_chain_replay_") as temporary:
        folder = Path(temporary)
        request, xml_path, response = folder / "request.npz", folder / "scene.xml", folder / "response.npz"
        with request.open("xb") as f:
            np.savez_compressed(f, state=state, actions=chains)
        xml = official_xml()
        xml_path.write_text(xml)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(LEGACY / "bin/python"), str(Path(__file__).resolve()),
                        "--backend", str(request), str(xml_path), str(response)],
                       env=env, check=True, timeout=120)
        with np.load(response, allow_pickle=False) as data:
            trace = data["trace"].copy()
            versions = {"binding": str(data["binding"]), "runtime": str(data["runtime"])}
    if trace.shape != (2, 26, 5) or not np.isfinite(trace).all():
        raise ValueError("invalid replay output")
    np.testing.assert_allclose(trace[:, 0, :4], np.broadcast_to(state, (2, 4)), atol=1e-7, rtol=0)
    np.testing.assert_allclose(trace[:, :, 4], np.broadcast_to(np.arange(26) * .05, (2, 26)), atol=1e-10, rtol=0)
    rows = {}
    for name, t in zip(("selected", "zero_control"), trace, strict=True):
        distance = np.linalg.norm(t[:, :2] - goal[:2], axis=1)
        rows[name] = {"initial_distance": float(distance[0]), "final_distance": float(distance[-1]),
                      "ever_success": bool(np.any(distance < .5)), "final_success": bool(distance[-1] < .5),
                      "trace": t.tolist()}
    record = {"track": STATE, "scope": "exposed one-root open-loop engineering replay, not RGB-only",
              "selection_sha256": file_hash(SELECT), "runner_sha256": file_hash(__file__),
              "public_sha256": file_hash(source.PUBLIC), "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
              "versions": versions, "time_contract": "25 environment steps, 5 physics steps at 0.01 s each",
              "goal_metric": "position norm < 0.5, including initial state",
              "model_candidates": saved["logical_queries"], "goal_encoding_calls": saved["goal_encoding_calls"],
              "rows": rows}
    with OUT.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)
    print(json.dumps({k: {x: v[x] for x in ("initial_distance", "final_distance", "ever_success")}
                      for k, v in rows.items()}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--backend":
        backend(*sys.argv[2:])
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("unexpected arguments")
