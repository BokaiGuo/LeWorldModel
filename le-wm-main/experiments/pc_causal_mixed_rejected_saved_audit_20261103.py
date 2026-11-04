"""Audit sealed rejected-prefix trajectories, selection and saved outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.pc_causal_mixed_rejected_physical_20261103 import rank_counts
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def audit(report, data, source, bank):
    if [r["root"] for r in report["rows"]] != list(ROOTS):
        raise ValueError("root roster mismatch")
    if data["starts"].shape != (32, 4) or data["sequences"].shape != (32, 17, 2):
        raise ValueError("invalid trajectory shape")
    for k, row in enumerate(report["rows"]):
        root = row["root"]
        reference = {b["name"]: b for b in source["rows"][k]["branches"]}
        if len(row["branches"]) != 8:
            raise ValueError("invalid branch count")
        for i, branch in enumerate(row["branches"]):
            arm, proposal = divmod(i, 4)
            candidate = proposal + 2
            if branch["arm"] != ("raw", "spectral")[arm] or branch["candidate"] != candidate:
                raise ValueError("candidate order mismatch")
            name = f"r{root}_a{arm}_p{candidate}"
            sequence, history = data[f"{name}_actions"], data[f"{name}_history"]
            if (not np.array_equal(sequence, data["sequences"][8*k+i])
                    or not np.array_equal(sequence[:2], np.zeros((2, 2)))
                    or not np.array_equal(sequence[2:7], bank[f"r{root}_banks"][arm, candidate])
                    or history.shape != (18, 224, 224, 3)
                    or not np.array_equal(data["starts"][8*k+i], bank["starts"][4*k])):
                raise ValueError("physical prefix binding mismatch")
            if branch["predicted_cp7_score"] != float(bank[f"r{root}_first_scores"][arm, candidate]):
                raise ValueError("predicted score binding mismatch")
            scores = data[f"{name}_feedback3_scores"]
            if not np.array_equal(scores, [branch["feedback_prior_score"], branch["feedback_refined_score"]]):
                raise ValueError("feedback score mismatch")
            if branch["feedback_refined"] != bool(scores[1] < scores[0]):
                raise ValueError("feedback selector mismatch")
            for tag in ("cp2", "cp7", "cp17"):
                pos = data[f"{tag}_eval_only"][8*k+i]
                if not np.isclose(np.linalg.norm(pos[:2] - GOAL[:2]), branch[f"{tag}_distance"],
                                  atol=1e-12, rtol=0):
                    raise ValueError("physical distance mismatch")
            if branch["physical_nonprogress_cp7"] != (branch["cp7_distance"] >= branch["cp2_distance"]):
                raise ValueError("nonprogress mismatch")
            beats = (branch["cp17_distance"] < reference["zero"]["cp17_distance"] and
                     branch["cp17_distance"] < reference["greedy"]["cp17_distance"])
            if branch["outcome_exposed_beats_zero_and_greedy"] != beats:
                raise ValueError("outcome comparison mismatch")
        pred = [b["predicted_cp7_score"] for b in row["branches"]]
        for key, vals in (("predicted_vs_actual_rgb", [b["actual_cp7_rgb_score"] for b in row["branches"]]),
                          ("predicted_vs_physical", [b["cp7_distance"] for b in row["branches"]])):
            if row["cp7_ranking"][key] != rank_counts(pred, vals):
                raise ValueError("ranking mismatch")
    return {"roots_checked": 4, "branches_checked": 32, "status": "pass"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report, source = (json.loads(p.read_text()) for p in (args.report, args.source))
    if sha256_file(args.source) != report["source_sha256"]:
        raise ValueError("source hash mismatch")
    paths = [p.with_suffix(".npz") for p in (args.report, args.source)]
    if any(sha256_file(p) != doc["arrays_sha256"] for p, doc in zip(paths, (report, source))):
        raise ValueError("array hash mismatch")
    with np.load(paths[0], allow_pickle=False) as data, np.load(paths[1], allow_pickle=False) as bank:
        result = audit(report, data, source, bank)
    result.update({"report_sha256": sha256_file(args.report), "source_sha256": sha256_file(args.source),
                   "script_sha256": sha256_file(Path(__file__)),
                   "scope": "saved bindings only, not model recalculation or independent physical replay"})
    with args.output.open("x") as out:
        json.dump(result, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
