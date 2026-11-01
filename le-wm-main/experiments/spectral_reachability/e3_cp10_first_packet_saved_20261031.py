"""Independent saved-vector arithmetic for the E3 cp10 first-packet run."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import e3_cp10_first_packet_cuda_20261031 as experiment
from . import e3_initial_goal_qualification_20261031 as qualification

RESULT = experiment.ROOT / "results/e3_cp10_first_packet_cuda_20261031_RUN01.json"
RESULT_SHA = "d9d099c9e496973c389bd4d66eb83dd1e1e5f6470da1617c9d9a55460ca8ec06"


def verify(data):
    if data["schema"] != experiment.NAME or data["status"] != "complete" or \
            data["E3_admission"] is not False or data["independent_confirmation"] is not False:
        raise ValueError("contract")
    expected = [(root, arm) for root in (0, 1) for arm in experiment.ARMS]
    if [(r["root"], r["arm"]) for r in data["rows"]] != expected:
        raise ValueError("roster")
    rows = []
    for row in data["rows"]:
        g, p, y = (np.asarray(row[name], dtype=np.float64) for name in ("goal", "prediction", "actual"))
        if any(v.shape != (192,) or not np.isfinite(v).all() for v in (g, p, y)):
            raise ValueError("vectors")
        scores = {"prediction_goal_mse": np.mean((p-g)**2),
                  "actual_goal_mse": np.mean((y-g)**2),
                  "prediction_actual_mse": np.mean((p-y)**2)}
        if any(not np.isclose(value, row[key], rtol=1e-5, atol=1e-7) for key, value in scores.items()):
            raise ValueError("score")
        goal, state = np.asarray(row["goal_state"]), np.asarray(row["actual_state"])
        native = qualification.native(goal, state)
        rows.append({"root": row["root"], "arm": row["arm"],
                     **{key: float(value) for key, value in scores.items()},
                     "object_position_error": float(np.linalg.norm(goal[2:4]-state[2:4])),
                     "pusher_position_error": float(np.linalg.norm(goal[:2]-state[:2])),
                     "native_success": native["initial_success"],
                     "joint_xy_error": native["joint_xy_error"]})
    return rows


def main():
    if hashlib.sha256(RESULT.read_bytes()).hexdigest() != RESULT_SHA:
        raise ValueError("result SHA")
    print(json.dumps({"status": "verified", "rows": verify(json.loads(RESULT.read_bytes()))}))


if __name__ == "__main__":
    main()
