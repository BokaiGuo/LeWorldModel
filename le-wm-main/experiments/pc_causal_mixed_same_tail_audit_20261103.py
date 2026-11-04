"""Check planned/executed tail binding and retrospective per-root rankings."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.pc_causal_mixed_rejected_physical_20261103 import rank_counts
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file

NORM_SHA = "a4c1c2a7083195e4951eb26ed559d90af583d6b414eb06125d628c0e9ea17f71"


def analyze(report, arrays, mean, std):
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or not np.all(std > 0):
        raise ValueError("invalid normalizer")
    rows = []
    for row in report["rows"]:
        root, branches = row["root"], row["branches"]
        if len(branches) != 8:
            raise ValueError("expected eight proposals")
        for branch in branches:
            arm = ("raw", "spectral").index(branch["arm"])
            name = f"r{root}_a{arm}_p{branch['candidate']}"
            scores = arrays[f"{name}_feedback3_scores"]
            proposals = arrays[f"{name}_feedback3_candidates"]
            if scores.shape != (2,) or proposals.shape != (1, 2, 1, 20):
                raise ValueError("invalid final feedback candidates")
            winner = int(scores[1] < scores[0])
            if bool(winner) != branch["feedback_refined"]:
                raise ValueError("feedback choice mismatch")
            expected = np.clip(proposals[0, winner, 0].reshape(10, 2) * std + mean, -1, 1)
            actual = arrays[f"{name}_actions"][7:]
            if not np.allclose(expected, actual, rtol=0, atol=1e-7):
                raise ValueError("predicted and executed physical tail differ")
        predicted = [min(b["feedback_prior_score"], b["feedback_refined_score"]) for b in branches]
        rgb = [b["actual_cp17_rgb_score"] for b in branches]
        physical = [b["cp17_distance"] for b in branches]
        rows.append({"root": root, "predicted_vs_actual_rgb": rank_counts(predicted, rgb),
                     "predicted_vs_physical": rank_counts(predicted, physical),
                     "actual_rgb_vs_physical": rank_counts(rgb, physical)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("normalizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.normalizer) != NORM_SHA:
        raise ValueError("historical normalizer receipt changed")
    report, norm = (json.loads(p.read_text()) for p in (args.report, args.normalizer))
    path = args.report.with_suffix(".npz")
    if sha256_file(path) != report["arrays_sha256"]:
        raise ValueError("array hash mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        rows = analyze(report, arrays, np.asarray(norm["action_mean"], np.float32),
                       np.asarray(norm["action_std"], np.float32))
    result = {"scope": "retrospective exposed-root fixed feedback actions; no model rerun",
              "report_sha256": sha256_file(args.report), "normalizer_sha256": NORM_SHA,
              "script_sha256": sha256_file(Path(__file__)), "rows": rows}
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
