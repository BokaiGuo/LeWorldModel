"""Saved binding audit for imagined versus true-RGB cp12 interventions."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.pc_causal_mixed_cp12_replan_saved_audit_20261103 import NORM_SHA, verify_tail
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def audit(report, arrays, sources, source_arrays, reals, real_arrays, mean, std):
    if [row["root"] for row in report["rows"]] != list(ROOTS):
        raise ValueError("invalid root roster")
    if arrays["sequences"].shape != (40, 17, 2):
        raise ValueError("invalid sequence shape")
    summary = []
    for k, row in enumerate(report["rows"]):
        root = row["root"]
        if len(row["branches"]) != 10:
            raise ValueError("invalid branch count")
        identical = 0
        for i, branch in enumerate(row["branches"]):
            group, index = (0, i) if i < 8 else (1, i-8)
            if group == 0:
                arm, candidate = divmod(index, 4)
                if (branch["arm"], branch["candidate"]) != (("raw", "spectral")[arm], candidate+2):
                    raise ValueError("candidate order mismatch")
                name = f"r{root}_a{arm}_p{candidate+2}"
            else:
                label = ("zero", "greedy")[index]
                if branch["name"] != label:
                    raise ValueError("control order mismatch")
                name = f"r{root}_{label}"
            seq, history = arrays[f"{name}_actions"], arrays[f"{name}_history"]
            old_seq, old_history = source_arrays[group][f"{name}_actions"], source_arrays[group][f"{name}_history"]
            if (not np.array_equal(seq, arrays["sequences"][10*k+i])
                    or not np.array_equal(seq[:12], old_seq[:12])
                    or not np.array_equal(history[:13], old_history[:13])):
                raise ValueError("physical prefix binding mismatch")
            scores = arrays[f"{name}_feedback3_scores"]
            if not np.array_equal(scores, [branch["prior_score"], branch["refined_score"]]):
                raise ValueError("feedback score mismatch")
            verify_tail(arrays[f"{name}_feedback3_candidates"], scores, branch["refined"],
                        seq[12:], mean, std)
            if branch["same_tail"] != bool(np.array_equal(seq[12:], old_seq[12:])):
                raise ValueError("tail identity mismatch")
            if not np.isfinite(arrays[f"{name}_imagined_proprio"]).all():
                raise ValueError("invalid imagined state")
            old_branch, real_branch = sources[group]["rows"][k]["branches"][index], reals[group]["rows"][k]["branches"][index]
            if (branch["original_cp17_distance"] != old_branch["cp17_distance"] or
                    branch["real_refresh_cp17_distance"] != real_branch["cp17_distance"]):
                raise ValueError("comparison reference mismatch")
            distance = np.linalg.norm(arrays["cp17_eval_only"][10*k+i, :2] - GOAL[:2])
            if not np.isclose(distance, branch["imagined_cp17_distance"], atol=1e-12, rtol=0):
                raise ValueError("physical distance mismatch")
            if np.array_equal(seq[12:], real_arrays[group][f"{name}_actions"][12:]):
                identical += 1
        summary.append({"root": root, "identical_real_and_imagined_tails": identical})
    return {"roots_checked": 4, "branches_checked": 40, "status": "pass", "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("candidate_source", type=Path)
    parser.add_argument("control_source", type=Path)
    parser.add_argument("candidate_real", type=Path)
    parser.add_argument("control_real", type=Path)
    parser.add_argument("normalizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [args.report, args.candidate_source, args.control_source, args.candidate_real, args.control_real]
    report, *docs = (json.loads(p.read_text()) for p in paths)
    if ([sha256_file(p) for p in paths[1:3]] != report["source_sha256"] or
            [sha256_file(p) for p in paths[3:5]] != report["real_sha256"] or
            sha256_file(args.normalizer) != NORM_SHA):
        raise ValueError("source hash mismatch")
    arrays_paths = [p.with_suffix(".npz") for p in paths]
    if any(sha256_file(path) != doc["arrays_sha256"] for path, doc in zip(arrays_paths, [report, *docs])):
        raise ValueError("array hash mismatch")
    norm = json.loads(args.normalizer.read_text())
    mean, std = np.asarray(norm["action_mean"], np.float32), np.asarray(norm["action_std"], np.float32)
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or not (std > 0).all():
        raise ValueError("invalid normalizer")
    with np.load(arrays_paths[0], allow_pickle=False) as out, \
            np.load(arrays_paths[1], allow_pickle=False) as src1, \
            np.load(arrays_paths[2], allow_pickle=False) as src2, \
            np.load(arrays_paths[3], allow_pickle=False) as real1, \
            np.load(arrays_paths[4], allow_pickle=False) as real2:
        result = audit(report, out, docs[:2], [src1, src2], docs[2:], [real1, real2], mean, std)
    result.update({"report_sha256": sha256_file(args.report), "script_sha256": sha256_file(Path(__file__)),
                   "scope": "saved bindings only; no model re-inference or provenance certification"})
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
