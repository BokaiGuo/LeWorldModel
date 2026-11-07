"""Post-hoc reward contract and position/velocity decomposition for saved chains."""
import inspect
import json
from pathlib import Path

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_geometric_contact_equivalence_dev_20261106_codex as panel
from experiments import pc_rgb_cem_incumbent_selection_20261107_codex as source

OUTPUT = source.ROOT / "results/pc_rgb_cem_reward_decomposition_20261107_codex.json"


def decompose(snapshot, goal):
    snapshot, goal = np.asarray(snapshot, float), np.asarray(goal, float)
    if snapshot.shape != (11,) or goal.shape != (7,) or not np.isfinite(np.r_[snapshot, goal]).all():
        raise ValueError("expected finite physical snapshot and seven-dimensional goal")
    state = np.r_[snapshot[:2], snapshot[5:7], snapshot[9] % (2 * np.pi), snapshot[2:4]]
    delta = state - goal
    wrapped_angle = float(abs(np.arctan2(np.sin(delta[4]), np.cos(delta[4]))))
    raw_angle = abs(float(delta[4]))
    # Exact local eval_state uses minimum(raw, 2*pi-raw), not atan2 wrap.
    official_angle = min(raw_angle, 2 * np.pi - raw_angle)
    combined_position = float(np.linalg.norm(delta[:4]))
    return {"state": state.tolist(), "agent_position": float(np.linalg.norm(delta[:2])),
            "object_position": float(np.linalg.norm(delta[2:4])),
            "agent_velocity": float(np.linalg.norm(delta[5:])),
            "raw_angle": raw_angle, "wrapped_angle": wrapped_angle,
            "combined_position": combined_position,
            "object_pose_cost": panel.pose_cost(state, goal),
            "reward": -float(np.linalg.norm(delta)),
            "success": bool(combined_position < 20 and official_angle < np.pi / 9)}


def run():
    from stable_worldmodel.envs.pusht.env import PushT
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    scores = json.loads(source.SCORES.read_text())
    evaluation = json.loads(source.EVAL.read_text())
    physical = json.loads(panel.PHYSICAL.read_text())
    if (evaluation["scores_sha256"] != file_hash(source.SCORES)
            or evaluation["physical_sha256"] != file_hash(panel.PHYSICAL)):
        raise ValueError("source drift")
    env_file = Path(inspect.getfile(PushT))
    rows = []
    for scored, evaluated in zip(scores["rows"], evaluation["rows"], strict=True):
        root = evaluated["root"]
        if root != scored["root"]:
            raise ValueError("root mismatch")
        goal = physical["rows"][root]["goal"]
        values = {}
        for arm in ("old", "best"):
            trace = evaluated["arms"][arm]
            values[arm] = decompose(trace[-1]["snapshot"], goal)
            if abs(values[arm]["reward"] - trace[-1]["reward"]) > 1e-9:
                raise ValueError("reward does not match native state contract")
            if values[arm]["success"] != trace[-1]["terminated"]:
                raise ValueError("termination does not match native state contract")
        rows.append({"root": root, "goal": goal, "arms": values})
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "physical_sha256": file_hash(panel.PHYSICAL),
                   "eval_sha256": file_hash(source.EVAL), "env_path": str(env_file),
                   "env_sha256": file_hash(env_file), "runner_sha256": file_hash(__file__),
                   "scope": "post-hoc diagnostic; hidden state is not planner input"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    run()
