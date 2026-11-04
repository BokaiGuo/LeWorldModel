"""Contact-by-tick contrast for frame-25 archive versus continuous state."""
import argparse
import json

import mujoco
import numpy as np

import a0_pointmaze_frame25_backend_20261102_codex_exclusive as base


def inspect(data):
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        contacts.append({"geom1": int(contact.geom1), "geom2": int(contact.geom2),
                         "distance": float(contact.dist)})
    return {"state": base.vector(data).tolist(), "time": float(data.time),
            "qacc_warmstart": data.qacc_warmstart.tolist(), "ncon": int(data.ncon),
            "contacts": contacts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml)
    with np.load(args.input, allow_pickle=False) as raw:
        states, actions = raw["states"], raw["actions"]
    continuous = base.set_qv(model, states[0])
    for frame in range(25):
        base.step(model, continuous, actions[frame])
    archived = base.set_qv(model, states[25])
    branches = {"continuous": continuous, "archive": archived}
    traces = {}
    for name, data in branches.items():
        trace = [inspect(data)]
        data.qvel[:] = np.clip(data.qvel, -5.0, 5.0)
        mujoco.mj_forward(model, data)
        data.ctrl[:] = np.clip(actions[25], -1.0, 1.0)
        trace.append(inspect(data))
        for _ in range(5):
            mujoco.mj_step(model, data)
            trace.append(inspect(data))
        traces[name] = trace
    result = {"mujoco": mujoco.__version__, "traces": traces,
              "geom_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                             for i in range(model.ngeom)]}
    with open(args.output, "x") as out:
        json.dump(result, out, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
