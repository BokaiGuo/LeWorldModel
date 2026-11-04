"""Cross-version MuJoCo state/clock differential, executed in each runtime."""
import argparse
import copy
import json
import sys

import mujoco
import numpy as np


def set_qv(model, vector):
    data = mujoco.MjData(model)
    data.qpos[:] = vector[:2]
    data.qvel[:] = vector[2:]
    mujoco.mj_forward(model, data)
    return data


def step(model, data, action):
    data.qvel[:] = np.clip(data.qvel, -5, 5)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = np.clip(action, -1, 1)
    for _ in range(5):
        mujoco.mj_step(model, data)


def vector(data):
    return np.r_[data.qpos[:2], data.qvel[:2]].copy()


def fields(data):
    names = ("time", "qpos", "qvel", "act", "ctrl", "qacc_warmstart",
             "qacc", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "userdata")
    return {n: np.asarray(getattr(data, n)).tolist() for n in names}


def compare(predicted, target):
    delta = predicted - target
    return {"state": predicted.tolist(), "target": target.tolist(), "delta": delta.tolist(),
            "state_l2": float(np.linalg.norm(delta)), "position_l2": float(np.linalg.norm(delta[:2])),
            "velocity_l2": float(np.linalg.norm(delta[2:]))}


def analyze(model, states, actions):
    results = []
    for start in (0, 20, 23, 24, 25):
        data = set_qv(model, states[start])
        for frame in range(start, 25):
            step(model, data, actions[frame])
        continuous = copy.copy(data)
        full_clone = copy.copy(data)
        qv_reset = set_qv(model, vector(data))
        archive_reset = set_qv(model, states[25])
        inputs = {"continuous_from_start": fields(continuous), "full_clone": fields(full_clone),
                  "qv_reset_from_continuous": fields(qv_reset), "qv_reset_from_archive": fields(archive_reset)}
        at25 = compare(vector(continuous), states[25])
        alternatives = {"continuous": continuous, "full_clone": full_clone,
                        "qv_from_continuous": qv_reset, "qv_from_archive": archive_reset}
        predictions = {}
        for name, value in alternatives.items():
            predictions[name] = {}
            for frame in (25, 26):
                step(model, value, actions[frame])
                predictions[name][str(frame + 1)] = compare(vector(value), states[frame + 1])
        results.append({"start_frame": start, "pre_step": inputs, "continuous_at25_vs_archive": at25,
                        "predictions": predictions})
    # Explicit precision perturbation, diagnostic only; not a proposed correction.
    variants = {}
    for variant in ("original_float64", "state_float32", "action_float32", "both_float32"):
        source = states[25].astype(np.float32).astype(np.float64) if variant in ("state_float32", "both_float32") else states[25]
        action = actions[25].astype(np.float32).astype(np.float64) if variant in ("action_float32", "both_float32") else actions[25]
        data = set_qv(model, source)
        step(model, data, action)
        variants[variant] = compare(vector(data), states[26])
    return {"runtime": {"mujoco": mujoco.__version__, "numpy": np.__version__,
                        "python": sys.version, "mjtNum_bytes": int(np.dtype(mujoco.MjData(model).qpos.dtype).itemsize)},
            "model": {"nq": model.nq, "nv": model.nv, "na": model.na, "nu": model.nu,
                      "nmocap": model.nmocap, "nuserdata": model.nuserdata,
                      "nbody": model.nbody, "timestep": model.opt.timestep,
                      "solver_iterations": model.opt.iterations},
            "archive_states_23_27": states[23:28].tolist(),
            "archive_actions_23_26": actions[23:27].tolist(),
            "runs": results, "precision_variants": variants}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        model = mujoco.MjModel.from_xml_path(args.xml)
        result = analyze(model, data["states"], data["actions"])
    with open(args.output, "x") as out:
        json.dump(result, out, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
