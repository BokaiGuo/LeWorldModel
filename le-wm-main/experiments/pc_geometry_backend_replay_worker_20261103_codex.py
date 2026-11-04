"""Isolated evaluator for fixed-action backend comparisons; no planner access."""
import json
import sys
from pathlib import Path

import numpy as np


def run(engine, request_path, xml_path, output_path):
    request = json.loads(Path(request_path).read_text())
    if engine == "native":
        import mujoco_py
        model = mujoco_py.load_model_from_xml(Path(xml_path).read_text())
        version = mujoco_py.__version__
        runtime = mujoco_py.get_version()
    elif engine == "modern":
        import mujoco
        model = mujoco.MjModel.from_xml_string(Path(xml_path).read_text())
        version = runtime = mujoco.__version__
    else:
        raise ValueError("unknown engine")
    rows = []
    for row in request["rows"]:
        if engine == "native":
            sim = mujoco_py.MjSim(model)
            data = sim.data
            forward, step = sim.forward, sim.step
            name = lambda idx: model.geom_id2name(idx) or f"geom#{idx}"
        else:
            data = mujoco.MjData(model)
            forward = lambda: mujoco.mj_forward(model, data)
            step = lambda: mujoco.mj_step(model, data)
            name = lambda idx: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, idx) or f"geom#{idx}"
        data.qpos[:] = row["start"]
        data.qvel[:] = 0
        forward()
        trace = []

        def capture():
            pairs = [sorted([name(int(data.contact[i].geom1)), name(int(data.contact[i].geom2))])
                     for i in range(data.ncon)]
            trace.append({"state": np.r_[data.qpos[:2], data.qvel[:2]].tolist(),
                          "time": float(data.time), "contacts": sorted(pairs)})
        capture()
        for action in row["actions"]:
            data.qvel[:] = np.clip(data.qvel, -5, 5)
            forward()
            data.ctrl[:] = np.clip(action, -1, 1)
            for _ in range(5):
                step()
            capture()
        rows.append({"start": row["start"], "policy": row["policy"], "trace": trace})
    with Path(output_path).open("x") as handle:
        json.dump({"engine": engine, "binding": version, "runtime": runtime,
                   "timestep": float(model.opt.timestep), "rows": rows}, handle, allow_nan=False)


if __name__ == "__main__":
    run(*sys.argv[1:])
