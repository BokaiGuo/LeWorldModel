"""Development-only same-action cross-backend state check; no rendering."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/a0_pointmaze_physical_cem_mpc_20261103_DEV01.json"
OUTPUT = ROOT / "results/a0_pointmaze_cpu_crossbackend_contract_20261103_SIDE.json"
LEGACY = ROOT / ".legacy_mujoco210_py310"
RUNTIME = LEGACY / "vendor/mujoco-2.1.2"


def worker():
    import mujoco_py

    req = json.load(sys.stdin)
    model = mujoco_py.load_model_from_xml(req["xml"])
    results = []
    for case in req["cases"]:
        sim = mujoco_py.MjSim(model)
        sim.reset()
        state = np.asarray(case["initial"], dtype=np.float64)
        sim.data.qpos[:], sim.data.qvel[:] = state[:2], state[2:]
        sim.forward()
        checkpoints = []
        for index, action in enumerate(case["actions"], 1):
            sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
            sim.forward()
            sim.data.ctrl[:] = action
            for _ in range(5):
                sim.step()
            if index in (5, 10):
                checkpoints.append(np.r_[sim.data.qpos[:2], sim.data.qvel[:2]].tolist())
        results.append(checkpoints)
    print(json.dumps(results, allow_nan=False))


def main():
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    from experiments.spectral_reachability import pointmaze_physical_evaluator as physical

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    cases = []
    for row in source["rows"]:
        plan = np.asarray(row["initial_plan"], dtype=np.float32)
        tail = np.asarray(row["tail_plan"], dtype=np.float32)
        actions = np.repeat(np.concatenate((plan[:1], tail), axis=0), 5, axis=0)
        if actions.shape != (10, 2) or np.any(np.abs(actions) > 1):
            raise ValueError("unexpected action contract")
        cases.append({"root": row["root"], "initial": row["initial_state_privileged"],
                      "actions": actions.tolist()})
    xml = official_xml()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME)
    env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                        "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
    proc = subprocess.run([str(LEGACY / "bin/python"), str(Path(__file__)), "--worker"],
                          input=json.dumps({"xml": xml, "cases": cases}), text=True,
                          capture_output=True, env=env, check=True)
    native = np.asarray(json.loads(proc.stdout), dtype=np.float64)
    initial = np.asarray([c["initial"] for c in cases], dtype=np.float64)
    actions = np.asarray([c["actions"] for c in cases], dtype=np.float64)
    cp5 = physical.replay_position_endpoints(initial, actions[:, :5])
    cp10 = physical.replay_position_endpoints(initial, actions)
    physical_states = np.stack((cp5, cp10), axis=1)
    saved = np.asarray([[row["cp5_state_privileged"], row["full_endpoint"]]
                        for row in source["rows"]])
    rows = []
    for i, case in enumerate(cases):
        delta = native[i] - physical_states[i]
        rows.append({"root": case["root"], "native_state_cp5_cp10": native[i].tolist(),
                     "physical_state_cp5_cp10": physical_states[i].tolist(),
                     "absolute_difference_cp5_cp10": np.abs(delta).tolist(),
                     "max_position_difference": float(np.max(np.abs(delta[:, :2]))),
                     "max_velocity_difference": float(np.max(np.abs(delta[:, 2:]))),
                     "max_saved_physical_difference": float(np.max(np.abs(physical_states[i] - saved[i])))})
    report = {"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "official_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
              "source_backend_receipt": physical.position_backend_receipt(),
              "native_backend": "mujoco-py 2.1.2; official RGB bridge XML and stepping; no render",
              "clock": "two packets of five archive actions; five simulator ticks per archive action",
              "scope": "four selected development chains; not candidate-wide parity, native success, or perfect dynamics",
              "rows": rows}
    with OUTPUT.open("x") as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    print(json.dumps({"output": str(OUTPUT), "rows": rows}, indent=2))


if __name__ == "__main__":
    worker() if sys.argv[1:] == ["--worker"] else main()
