"""Audit saved H10 candidate, selector, tail decoding and physical bindings."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.pc_pointmaze_affine_nonprogress_lookahead_20261103 import choose
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file

SOURCE_SHA = "d04c8b56024dbe3cbb923bdc5a76f930f9af1e5438b5d5e1c87d72ae41fa316a"
NORMALIZER_SHA = "a4c1c2a7083195e4951eb26ed559d90af583d6b414eb06125d628c0e9ea17f71"


def audit(report, arrays, old, mean, std):
    if [r["root"] for r in report["rows"]] != list(ROOTS):
        raise ValueError("invalid root roster")
    starts, actions = arrays["starts"], arrays["sequences"]
    if starts.shape != (36, 4) or actions.shape != (36, 17, 2):
        raise ValueError("invalid action or reset shape")
    for k, row in enumerate(report["rows"]):
        root = row["root"]
        first, scores = arrays[f"r{root}_first_scores"], arrays[f"r{root}_terminal_scores"]
        tails, bank = arrays[f"r{root}_old_tails"], arrays[f"r{root}_bank"]
        if choose(first, row["current_score"], scores) != (row["selected"], row["greedy"], row["eligible"]):
            raise ValueError("selector mismatch")
        if not np.array_equal(bank, old[f"r{root}_bank"]):
            raise ValueError("source bank mismatch")
        for i, branch in enumerate(row["branches"]):
            if branch["bank_index"] != i:
                raise ValueError("branch ordering mismatch")
            name, idx = f"r{root}_p{i}", k*9 + i
            history, sequence = arrays[f"{name}_history"], arrays[f"{name}_actions"]
            if history.shape != (18, 224, 224, 3) or sequence.shape != (17, 2):
                raise ValueError("history shape mismatch")
            if (not np.array_equal(sequence, actions[idx]) or not np.array_equal(sequence[:2], np.zeros((2, 2)))
                    or not np.array_equal(sequence[2:7], bank[i])
                    or not np.array_equal(history[:8], old[f"{name}_history"][:8])):
                raise ValueError("prefix binding mismatch")
            proposals, score = arrays[f"{name}_lookahead3_proposals"], arrays[f"{name}_lookahead3_scores"]
            if proposals.shape != (1, 2, 1, 20) or score.shape != (2,) or not np.array_equal(score, scores[i]):
                raise ValueError("lookahead final score mismatch")
            if not np.array_equal(proposals[0, int(score[1] < score[0]), 0], tails[i]):
                raise ValueError("lookahead chosen tail mismatch")
            feedback, cost = arrays[f"{name}_feedback3_proposals"], arrays[f"{name}_feedback3_scores"]
            if feedback.shape != (1, 2, 1, 20) or cost.shape != (2,):
                raise ValueError("feedback final batch mismatch")
            if not np.array_equal(cost, [branch["feedback_prior_score"], branch["feedback_refined_score"]]):
                raise ValueError("feedback reported score mismatch")
            winner = int(cost[1] < cost[0])
            if branch["feedback_refined"] != bool(winner):
                raise ValueError("feedback selection mismatch")
            decoded = np.clip(feedback[0, winner, 0].reshape(10, 2)*std + mean, -1, 1)
            if not np.allclose(decoded, sequence[7:], rtol=0, atol=1e-7):
                raise ValueError("feedback physical tail decode mismatch")
            for tag in ("cp2", "cp7", "cp17"):
                pos = arrays[f"{tag}_eval_only"][idx]
                if not np.isclose(np.linalg.norm(pos[:2] - GOAL[:2]), branch[f"{tag}_distance"],
                                  rtol=0, atol=1e-12):
                    raise ValueError("physical score mismatch")
            if branch["physical_nonprogress_cp7"] != (branch["cp7_distance"] >= branch["cp2_distance"]):
                raise ValueError("physical nonprogress mismatch")
    return {"branches_checked": len(starts), "status": "pass"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("normalizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.source) != SOURCE_SHA or sha256_file(args.normalizer) != NORMALIZER_SHA:
        raise ValueError("source or normalizer changed")
    report, source, norm = (json.loads(p.read_text()) for p in (args.report, args.source, args.normalizer))
    data_path, old_path = args.report.with_suffix(".npz"), args.source.with_suffix(".npz")
    if sha256_file(data_path) != report["arrays_sha256"] or sha256_file(old_path) != source["arrays_sha256"]:
        raise ValueError("array hash mismatch")
    mean, std = np.asarray(norm["action_mean"], np.float32), np.asarray(norm["action_std"], np.float32)
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or not (std > 0).all():
        raise ValueError("invalid historical normalizer")
    with np.load(data_path, allow_pickle=False) as data, np.load(old_path, allow_pickle=False) as old:
        result = audit(report, data, old, mean, std)
    result.update({"report_sha256": sha256_file(args.report), "arrays_sha256": sha256_file(data_path),
                   "normalizer_sha256": NORMALIZER_SHA, "source_sha256": SOURCE_SHA,
                   "script_sha256": sha256_file(Path(__file__)),
                   "scope": "saved bindings only; historical normalizer is not complete provenance proof"})
    with args.output.open("x") as out:
        json.dump(result, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
