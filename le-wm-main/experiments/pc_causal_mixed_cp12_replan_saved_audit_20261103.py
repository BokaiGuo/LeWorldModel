"""Verify cp12 replan action, feedback score and physical trajectory bindings."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file

NORM_SHA = "a4c1c2a7083195e4951eb26ed559d90af583d6b414eb06125d628c0e9ea17f71"


def verify_tail(proposals, scores, refined, action, mean, std):
    if proposals.shape != (1, 2, 1, 10) or scores.shape != (2,):
        raise ValueError("invalid final CEM batch")
    winner = int(scores[1] < scores[0])
    if bool(winner) != refined:
        raise ValueError("feedback selection mismatch")
    decoded = np.clip(proposals[0, winner, 0].reshape(5, 2) * std + mean, -1, 1)
    if not np.allclose(decoded, action, rtol=0, atol=1e-7):
        raise ValueError("normalized action decoding mismatch")


def audit(report, arrays, source, old_arrays, mean, std, controls):
    if [r["root"] for r in report["rows"]] != list(ROOTS):
        raise ValueError("invalid root roster")
    per_root = 2 if controls else 8
    if arrays["sequences"].shape != (4 * per_root, 17, 2):
        raise ValueError("sequence shape mismatch")
    for k, row in enumerate(report["rows"]):
        if len(row["branches"]) != per_root or source["rows"][k]["root"] != row["root"]:
            raise ValueError("branch mismatch")
        for i, branch in enumerate(row["branches"]):
            root = row["root"]
            if controls:
                label = ("zero", "greedy")[i]
                if branch["name"] != label:
                    raise ValueError("control name mismatch")
                name = f"r{root}_{label}"
            else:
                arm, candidate = divmod(i, 4)
                if (branch["arm"], branch["candidate"]) != (("raw", "spectral")[arm], candidate + 2):
                    raise ValueError("candidate mismatch")
                name = f"r{root}_a{arm}_p{candidate + 2}"
            seq, hist = arrays[f"{name}_actions"], arrays[f"{name}_history"]
            old_seq, old_hist = old_arrays[f"{name}_actions"], old_arrays[f"{name}_history"]
            if (not np.array_equal(seq, arrays["sequences"][per_root*k+i])
                    or not np.array_equal(seq[:12], old_seq[:12])
                    or not np.array_equal(hist[:13], old_hist[:13])):
                raise ValueError("physical prefix/RGB mismatch")
            scores = arrays[f"{name}_feedback3_scores"]
            if not np.array_equal(scores, [branch["prior_score"], branch["refined_score"]]):
                raise ValueError("feedback score mismatch")
            verify_tail(arrays[f"{name}_feedback3_candidates"], scores, branch["refined"],
                        seq[12:], mean, std)
            if branch["same_tail"] != bool(np.array_equal(seq[12:], old_seq[12:])):
                raise ValueError("tail identity mismatch")
            expected_old = source["rows"][k]["branches"][i]["cp17_distance"]
            if branch["old_cp17_distance"] != expected_old:
                raise ValueError("old physical outcome mismatch")
            pos = arrays["cp17_eval_only"][per_root*k+i]
            dist = np.linalg.norm(pos[:2] - GOAL[:2])
            if not np.isclose(dist, branch["cp17_distance"], atol=1e-12, rtol=0):
                raise ValueError("new physical outcome mismatch")
            if not np.isclose(dist - expected_old, branch["distance_delta_new_minus_old"], atol=1e-12, rtol=0):
                raise ValueError("physical comparison mismatch")
    return {"roots_checked": 4, "branches_checked": 4 * per_root, "status": "pass"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("normalizer", type=Path)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report, source, norm = (json.loads(p.read_text()) for p in (args.report, args.source, args.normalizer))
    if sha256_file(args.source) != report["source_sha256"] or sha256_file(args.normalizer) != NORM_SHA:
        raise ValueError("source/normalizer hash mismatch")
    paths = [args.report.with_suffix(".npz"), args.source.with_suffix(".npz")]
    if any(sha256_file(p) != doc["arrays_sha256"] for p, doc in zip(paths, (report, source))):
        raise ValueError("array hash mismatch")
    mean, std = np.asarray(norm["action_mean"], np.float32), np.asarray(norm["action_std"], np.float32)
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or not (std > 0).all():
        raise ValueError("invalid normalizer")
    with np.load(paths[0], allow_pickle=False) as arrays, np.load(paths[1], allow_pickle=False) as old:
        result = audit(report, arrays, source, old, mean, std, args.controls)
    result.update({"report_sha256": sha256_file(args.report), "source_sha256": sha256_file(args.source),
                   "normalizer_sha256": NORM_SHA, "script_sha256": sha256_file(Path(__file__)),
                   "scope": "saved binding audit only; normalizer provenance not established"})
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
