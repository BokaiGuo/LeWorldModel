"""Qualify initial success in the sealed E3 matrix without changing its tasks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_worldmodel.envs.pusht.env import PushT

ROOT = Path(__file__).resolve().parents[2]
NAME = "e3_initial_goal_qualification_20261031"
INPUT = ROOT / "results/agent_e3_frozen_model_task_resolver_20261029_v1_prepare_MAIN01.json"
INPUT_SHA = "b44699e6a10bb4c257ed8872dc86b99a37508d7022850f8b4fcd257bcd232b26"
ENV_SOURCE = ROOT / ".venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py"
ENV_SHA = "c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2"


def require(value, message):
    if not value:
        raise ValueError(message)


def state(value):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == (7,) and np.isfinite(result).all(), "native state")
    return result


def native(goal, initial):
    # Pure environment method: no simulator construction, reset, neural inference or action.
    success, distance = PushT.eval_state(None, state(goal), state(initial))
    require(np.isfinite(distance), "native distance")
    return {"initial_success": bool(success), "native7_distance": float(distance),
            "joint_xy_error": float(np.linalg.norm(state(goal)[:4] - state(initial)[:4]))}


def analyze(data):
    require(data["schema"] == "agent_e3_frozen_model_task_resolver_20261029_v1.prepare"
            and len(data["cells"]) == 16 and len(data["root_provenance"]) == 2,
            "fixed matrix")
    roots = []
    for index, item in enumerate(data["root_provenance"]):
        require(item["root"] == index and item["status"] == "qualified_saved_prefix", "root identity")
        branches = item["provenance"]["root_native_branches"]
        require(len(branches) == 2, "bank roster")
        for key in ("c0", "c5"):
            require(branches[0][key]["state"] == branches[1][key]["state"]
                    and branches[0][key]["pixel_sha256"] == branches[1][key]["pixel_sha256"],
                    "bank state and pixel identity")
        roots.append(branches[0])
    rows = []
    for index, cell in enumerate(data["cells"]):
        require(cell["index"] == index and cell["root"] in (0, 1)
                and cell["goal_index"] in (0, 1) and cell["status"] == "input_ready_NOT_run_authorized",
                "cell identity")
        root = roots[cell["root"]]
        goal = cell["evaluator"]["native_goal"]
        if cell["goal_index"] == 1:
            require(cell["goal_evidence"]["kind"] == "G1_previously_observed_control0"
                    and cell["goal_evidence"]["original_control_step"] == 0
                    and goal == root["c0"]["state"], "G1 actual root identity")
        rows.append({"index": index, "cell_id": cell["cell_id"], "root": cell["root"],
                     "goal_index": cell["goal_index"], **native(goal, root["c0"]["state"])})
    candidates = []
    for index, root in enumerate(roots):
        candidates.append({"root": index, "candidate_kind": "previously_observed_cp5",
                           "pixel_sha256": root["c5"]["pixel_sha256"],
                           "goal_state": root["c5"]["state"],
                           **native(root["c5"]["state"], root["c0"]["state"]),
                           "historical_action_witness": True})
    return {"schema": NAME, "rows": rows, "exploratory_cp5_candidates": candidates,
            "scientific_admission": False, "independent_confirmation": False,
            "production_authorized": False,
            "limitations": "Saved known roots and observed targets; initial success only, "
                           "not an online evaluation or new independent task."}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    require(out.is_absolute() and out.parent == ROOT / "results"
            and out.name.startswith(NAME + "_") and not out.exists()
            and not out.is_symlink(), "exclusive output")
    require(sha(INPUT) == INPUT_SHA and sha(ENV_SOURCE) == ENV_SHA, "input hashes")
    result = analyze(json.loads(INPUT.read_bytes()))
    result["sources"] = {str(INPUT): INPUT_SHA, str(ENV_SOURCE): ENV_SHA}
    result["code_sha256"] = sha(Path(__file__).resolve())
    require(sha(INPUT) == INPUT_SHA and sha(ENV_SOURCE) == ENV_SHA, "source drift")
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with out.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"initially_successful": sum(r["initial_success"] for r in result["rows"]),
                      "candidate_initial_success": [r["initial_success"] for r in result["exploratory_cp5_candidates"]],
                      "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
