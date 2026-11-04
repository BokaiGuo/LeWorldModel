"""Audit saved candidate, score, selector, physical and historical bindings."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.pc_causal_mixed_prefix_search_20261103 import candidates, select
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def audit(report, arrays, source, h10):
    if [r["root"] for r in report["rows"]] != list(ROOTS):
        raise ValueError("root roster mismatch")
    if arrays["starts"].shape != (16, 4) or arrays["sequences"].shape != (16, 17, 2):
        raise ValueError("branch shape mismatch")
    for k, row in enumerate(report["rows"]):
        root = row["root"]
        old_first = source[f"r{root}_first_scores"]
        greedy = int(np.argmin(old_first))
        if row["source_greedy"] != greedy:
            raise ValueError("greedy source mismatch")
        proposal, _ = candidates(source[f"r{root}_jacobian"], 2026110390 + root)
        banks = arrays[f"r{root}_banks"]
        if not np.array_equal(banks[:, 2:], proposal):
            raise ValueError("candidate generation mismatch")
        expected_shared = np.stack((np.zeros((5, 2), np.float32), source[f"r{root}_bank"][greedy]))
        if not np.array_equal(banks[:, :2], np.repeat(expected_shared[None], 2, axis=0)):
            raise ValueError("shared candidate mismatch")
        first, terminal = arrays[f"r{root}_first_scores"], arrays[f"r{root}_terminal_scores"]
        for arm in range(2):
            decision = select(first[arm], terminal[arm], row["current_score"])
            saved = row["decisions"][arm]
            if decision != (saved["winner"], saved["qualified_nonprogress"], saved["eligible"]):
                raise ValueError("selector mismatch")
            for i in range(6):
                score = arrays[f"r{root}_a{arm}_p{i}_lookahead3_scores"]
                if score.shape != (2,) or terminal[arm, i] != min(score):
                    raise ValueError("terminal scoring mismatch")
        selected = [banks[0, 0], banks[0, 1], banks[0, row["decisions"][0]["winner"]],
                    banks[1, row["decisions"][1]["winner"]]]
        for i, branch in enumerate(row["branches"]):
            name = branch["name"]
            seq = arrays[f"r{root}_{name}_actions"]
            history = arrays[f"r{root}_{name}_history"]
            if (history.shape != (18, 224, 224, 3) or
                    not np.array_equal(seq, arrays["sequences"][4*k+i]) or
                    not np.array_equal(seq[:2], np.zeros((2, 2))) or
                    not np.array_equal(seq[2:7], selected[i]) or
                    not np.array_equal(history[:3], source[f"r{root}_p0_history"][:3])):
                raise ValueError("physical action/history binding mismatch")
            feedback = arrays[f"r{root}_{name}_feedback3_scores"]
            if not np.array_equal(feedback, [branch["feedback_prior_score"], branch["feedback_refined_score"]]):
                raise ValueError("feedback score mismatch")
            if branch["feedback_refined"] != bool(feedback[1] < feedback[0]):
                raise ValueError("feedback selection mismatch")
            for tag in ("cp2", "cp7", "cp17"):
                pos = arrays[f"{tag}_eval_only"][4*k+i]
                if not np.isclose(np.linalg.norm(pos[:2] - GOAL[:2]), branch[f"{tag}_distance"],
                                  atol=1e-12, rtol=0):
                    raise ValueError("physical distance mismatch")
            if branch["physical_nonprogress_cp7"] != (branch["cp7_distance"] >= branch["cp2_distance"]):
                raise ValueError("physical nonprogress mismatch")
        for label, idx in (("zero", 0), ("greedy", greedy)):
            if not np.array_equal(arrays[f"r{root}_{label}_actions"], h10[f"r{root}_p{idx}_actions"]):
                raise ValueError("historical fixed branch actions differ")
            if not np.array_equal(arrays[f"r{root}_{label}_history"], h10[f"r{root}_p{idx}_history"]):
                raise ValueError("historical fixed branch RGB differs")
    return {"roots_checked": len(report["rows"]), "branches_checked": 16, "status": "pass"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("h10", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report, source, h10 = (json.loads(p.read_text()) for p in (args.report, args.source, args.h10))
    if sha256_file(args.source) != report["source_sha256"] or h10["source_sha256"] != report["source_sha256"]:
        raise ValueError("source hash mismatch")
    paths = [p.with_suffix(".npz") for p in (args.report, args.source, args.h10)]
    for path, doc in zip(paths, (report, source, h10)):
        if sha256_file(path) != doc["arrays_sha256"]:
            raise ValueError("array hash mismatch")
    with np.load(paths[0], allow_pickle=False) as arrays, np.load(paths[1], allow_pickle=False) as old, \
            np.load(paths[2], allow_pickle=False) as prior:
        result = audit(report, arrays, old, prior)
    result.update({"report_sha256": sha256_file(args.report),
                   "script_sha256": sha256_file(Path(__file__)),
                   "scope": "saved bindings and historical replay; no model recomputation or provenance certification"})
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
