"""Independently audit saved action/score/selection bindings; no model calls."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.pc_pointmaze_affine_nonprogress_lookahead_20261103 import choose
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def audit(report, arrays):
    rows = report["rows"]
    if [r["root"] for r in rows] != list(ROOTS):
        raise ValueError("root roster mismatch")
    starts, actions = arrays["starts"], arrays["sequences"]
    cp2, cp7, cp12 = (arrays[f"cp{i}_eval_only"] for i in (2, 7, 12))
    if starts.shape != (36, 4) or actions.shape != (36, 12, 2):
        raise ValueError("invalid action array")
    for k, row in enumerate(rows):
        root = row["root"]
        first = arrays[f"r{root}_first_scores"]
        terminal = arrays[f"r{root}_terminal_scores"]
        tails = arrays[f"r{root}_old_tails"]
        bank = arrays[f"r{root}_bank"]
        decision = choose(first, row["current_score"], terminal)
        if decision != (row["selected"], row["greedy"], row["eligible"]):
            raise ValueError("saved model selection mismatch")
        for i, branch in enumerate(row["branches"]):
            if branch["bank_index"] != i:
                raise ValueError("branch order mismatch")
            idx = 9*k + i
            recorded = arrays[f"r{root}_p{i}_actions"]
            history = arrays[f"r{root}_p{i}_history"]
            if recorded.shape != (12, 2) or history.shape != (13, 224, 224, 3):
                raise ValueError("branch shape mismatch")
            if (not np.array_equal(recorded, actions[idx])
                    or not np.array_equal(recorded[:2], np.zeros((2, 2)))
                    or not np.array_equal(recorded[2:7], bank[i])):
                raise ValueError("prefix/sequence binding mismatch")
            if i and not np.array_equal(history[:3], arrays[f"r{root}_p0_history"][:3]):
                raise ValueError("RGB prelude binding mismatch")
            proposals = arrays[f"r{root}_p{i}_lookahead3_proposals"]
            scores = arrays[f"r{root}_p{i}_lookahead3_scores"]
            if proposals.shape != (1, 2, 1, 10) or scores.shape != (2,):
                raise ValueError("invalid final lookahead batch")
            if not np.array_equal(scores, terminal[i]):
                raise ValueError("lookahead score mismatch")
            winner = int(scores[1] < scores[0])
            if not np.array_equal(proposals[0, winner, 0], tails[i]):
                raise ValueError("lookahead tail mismatch")
            feedback_scores = arrays[f"r{root}_p{i}_feedback3_scores"]
            expected = [branch["feedback_prior_score"], branch["feedback_refined_score"]]
            if not np.array_equal(feedback_scores, expected):
                raise ValueError("feedback score mismatch")
            if branch["feedback_refined"] != bool(feedback_scores[1] < feedback_scores[0]):
                raise ValueError("feedback selection mismatch")
            for label, position in (("cp2", cp2[idx]), ("cp7", cp7[idx]), ("cp12", cp12[idx])):
                if not np.isclose(np.linalg.norm(position[:2] - GOAL[:2]),
                                  branch[f"{label}_distance"], rtol=0, atol=1e-12):
                    raise ValueError("physical distance mismatch")
            if branch["physical_nonprogress_cp7"] != (branch["cp7_distance"] >= branch["cp2_distance"]):
                raise ValueError("physical nonprogress mismatch")
    return {"roots": len(rows), "branches_checked": len(starts), "status": "pass"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    data_path = args.report.with_suffix(".npz")
    if sha256_file(data_path) != report["arrays_sha256"]:
        raise ValueError("array hash mismatch")
    with np.load(data_path, allow_pickle=False) as data:
        result = audit(report, data)
    result.update({"scope": "saved data binding only, no independent model or physical run",
                   "report_sha256": sha256_file(args.report), "arrays_sha256": sha256_file(data_path),
                   "script_sha256": sha256_file(Path(__file__))})
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
