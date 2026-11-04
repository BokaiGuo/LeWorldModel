"""Audit joint optimizer decisions and common-suffix trajectory bindings."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.pc_causal_joint_prefix_gradient_20261103 import decide
from experiments.pc_causal_joint_qualified_common_suffix_20261103 import build_suffixes
from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import GOAL, ROOTS
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def audit(joint, ja, common, ca):
    if [r["root"] for r in joint["rows"]] != list(ROOTS) or common["root"] != 8349:
        raise ValueError("root roster mismatch")
    for row in joint["rows"]:
        root = row["root"]
        plans, first, terminal = (ja[f"r{root}_{x}"] for x in ("plans", "first", "end"))
        if plans.shape != (13, 10, 15, 2) or not np.isfinite(plans).all() or np.any(np.abs(plans) > 1):
            raise ValueError("invalid joint optimizer trace")
        controls, decisions = decide(first, terminal, row["current_score"])
        if controls != row["control_iterations"] or decisions != row["decisions"]:
            raise ValueError("joint selector mismatch")
        for b in row["branches"]:
            p, t = b["candidate"], b["iteration"]
            seq = ja[f"r{root}_{b['name']}_actions"]
            if not np.array_equal(seq[2:7], plans[t, p, :5]):
                raise ValueError("joint prefix binding mismatch")
            if b["predicted_prefix_score"] != float(first[t, p]) or b["predicted_terminal_score"] != float(terminal[t, p]):
                raise ValueError("joint score binding mismatch")
    row = joint["rows"][list(ROOTS).index(8349)]
    qualified, greedy = row["decisions"][0]["qualified"], [1, row["control_iterations"][1]]
    if qualified is None:
        raise ValueError("qualified model prefix missing")
    suffixes = build_suffixes(ja["r8349_plans"], qualified, greedy)
    if not np.array_equal(suffixes, ca["suffixes"]):
        raise ValueError("suffix bank mismatch")
    names, suffix_names = ("zero", "greedy", "raw_qualified"), ("zero", "qualified_model", "greedy_model", "random0", "random1")
    if len(common["rows"]) != 15 or ca["sequences"].shape != (15, 17, 2):
        raise ValueError("common panel shape mismatch")
    for i, b in enumerate(common["rows"]):
        prefix, suffix = names[i//5], suffix_names[i%5]
        if (b["prefix"], b["suffix"]) != (prefix, suffix):
            raise ValueError("branch roster mismatch")
        name = f"{prefix}_{suffix}"
        seq, hist = ca[f"{name}_actions"], ca[f"{name}_history"]
        if (not np.array_equal(seq, ca["sequences"][i]) or
                not np.array_equal(seq[:7], ja[f"r8349_{prefix}_actions"][:7]) or
                not np.array_equal(seq[7:], suffixes[i%5]) or
                not np.array_equal(hist[:8], ja[f"r8349_{prefix}_history"][:8])):
            raise ValueError("common action/RGB binding mismatch")
        for tag in ("cp2", "cp7", "cp17"):
            dist = float(np.linalg.norm(ca[f"{tag}_eval_only"][i, :2]-GOAL[:2]))
            if not np.isclose(dist, b[f"{tag}_distance"], rtol=0, atol=1e-12):
                raise ValueError("distance mismatch")
        ref = (i//5)*5
        diff = ca["cp17_eval_only"][i, :2] - ca["cp17_eval_only"][ref, :2]
        if not np.allclose(diff, b["cp17_position_minus_own_zero"], rtol=0, atol=1e-12):
            raise ValueError("physical response mismatch")
        if not np.isclose(b["cp17_distance"] - common["rows"][ref]["cp17_distance"],
                          b["cp17_distance_minus_own_zero"], rtol=0, atol=1e-12):
            raise ValueError("relative score mismatch")
        if b["physical_nonprogress_cp7"] != (b["cp7_distance"] >= b["cp2_distance"]):
            raise ValueError("nonprogress mismatch")
    return {"joint_roots_checked": 4, "common_suffix_branches_checked": 15, "status": "pass"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("joint", type=Path)
    parser.add_argument("common", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    joint, common = (json.loads(p.read_text()) for p in (args.joint, args.common))
    if sha256_file(args.joint) != common["source_sha256"]:
        raise ValueError("joint hash mismatch")
    paths = [p.with_suffix(".npz") for p in (args.joint, args.common)]
    if any(sha256_file(p) != d["arrays_sha256"] for p, d in zip(paths, (joint, common))):
        raise ValueError("array hash mismatch")
    with np.load(paths[0], allow_pickle=False) as ja, np.load(paths[1], allow_pickle=False) as ca:
        result = audit(joint, ja, common, ca)
    result.update({"joint_sha256": sha256_file(args.joint), "common_sha256": sha256_file(args.common),
                   "script_sha256": sha256_file(Path(__file__)),
                   "scope": "saved decisions, actions and physical positions; no independent model rerun"})
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
