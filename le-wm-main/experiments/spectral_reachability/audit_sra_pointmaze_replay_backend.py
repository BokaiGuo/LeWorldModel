"""Backend-only half of the versioned PointMaze raw-transition audit.

This module intentionally depends only on NumPy and one explicitly selected
MuJoCo binding.  Its CUDA-gated parent exports a fixed raw subset and invokes
it from an isolated legacy interpreter, allowing simulator version fidelity to
be measured without moving a learned-model run off the host CUDA environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FRAME_SKIP, VELOCITY_LIMIT = 5, 5.0


def replay(model: object, state: np.ndarray, actions: np.ndarray, *, engine: str = "mujoco") -> np.ndarray:
    """Replay one high-level sequence under the selected, reported binding."""
    if engine == "mujoco":
        import mujoco

        data = mujoco.MjData(model)
        forward, step = lambda: mujoco.mj_forward(model, data), lambda: mujoco.mj_step(model, data)
    elif engine == "mujoco_py":
        import mujoco_py

        sim = mujoco_py.MjSim(model)
        data = sim.data
        forward, step = sim.forward, sim.step
    else:  # Defensive even though argparse constrains the option.
        raise ValueError(f"unsupported replay engine: {engine}")
    data.qpos[:] = state[:2]
    data.qvel[:] = state[2:]
    forward()
    for action in actions:
        data.qvel[:] = np.clip(data.qvel, -VELOCITY_LIMIT, VELOCITY_LIMIT)
        forward()
        data.ctrl[:] = np.clip(action, -1.0, 1.0)
        for _ in range(FRAME_SKIP):
            step()
    return np.r_[np.asarray(data.qpos[:2]), np.asarray(data.qvel[:2])]


def load_model(xml: str, *, engine: str) -> tuple[object, dict[str, str]]:
    """Load the XML and return immutable binding/runtime provenance."""
    if engine == "mujoco":
        import mujoco

        return mujoco.MjModel.from_xml_string(xml), {"engine": engine, "mujoco": mujoco.__version__}
    if engine == "mujoco_py":
        import mujoco_py

        return mujoco_py.load_model_from_xml(xml), {
            "engine": engine,
            "mujoco_py": mujoco_py.__version__,
            "mujoco_runtime": mujoco_py.get_version(),
        }
    raise ValueError(f"unsupported replay engine: {engine}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--state-tolerance", type=float, required=True)
    parser.add_argument("--engine", choices=("mujoco", "mujoco_py"), default="mujoco")
    args = parser.parse_args()
    if args.report.exists() or not args.subset.is_file() or not args.xml.is_file():
        raise FileExistsError("backend report must be new and subset/XML must exist")
    payload = np.load(args.subset, allow_pickle=False)
    states, actions = payload["states"], payload["actions"]
    roots, frames, horizons = payload["roots"], payload["frames"], payload["horizons"]
    model, backend = load_model(args.xml.read_text(encoding="utf-8"), engine=args.engine)
    rows = []
    for root in roots:
        for frame in frames:
            for horizon in horizons:
                predicted = replay(model, states[root, frame], actions[root, frame:frame + horizon], engine=args.engine)
                target = states[root, frame + horizon]
                error = predicted - target
                rows.append({"root": int(root), "frame": int(frame), "horizon": int(horizon), "state_l2_error": float(np.linalg.norm(error)), "position_l2_error": float(np.linalg.norm(error[:2])), "velocity_l2_error": float(np.linalg.norm(error[2:]))})
    values = np.asarray([row["state_l2_error"] for row in rows])
    positions = np.asarray([row["position_l2_error"] for row in rows])
    report = {"schema": "sra_pointmaze_replay_backend.v2", "backend": backend, "status": "pass" if float(values.max()) <= args.state_tolerance else "fail", "summary": {"rows": len(rows), "state_tolerance": args.state_tolerance, "max_state_l2_error": float(values.max()), "mean_state_l2_error": float(values.mean()), "max_position_l2_error": float(positions.max()), "mean_position_l2_error": float(positions.mean())}, "rows": rows}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"backend": report["backend"], "status": report["status"], "summary": report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
