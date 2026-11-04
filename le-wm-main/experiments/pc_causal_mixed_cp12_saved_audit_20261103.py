"""Independently recompute cp12 refresh embedding errors and saved rankings."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.pc_causal_mixed_rejected_physical_20261103 import rank_counts
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def audit(report, arrays, old):
    if len(report["rows"]) != len(old["rows"]):
        raise ValueError("root count mismatch")
    branches = []
    for row, source in zip(report["rows"], old["rows"]):
        root = row["root"]
        if root != source["root"] or len(row["branches"]) != 8:
            raise ValueError("root/branch mismatch")
        for b, original in zip(row["branches"], source["branches"]):
            if (b["arm"], b["candidate"]) != (original["arm"], original["candidate"]):
                raise ValueError("candidate mismatch")
            name = f"r{root}_a{('raw', 'spectral').index(b['arm'])}_p{b['candidate']}"
            for kind in ("visual", "proprio"):
                actual = arrays[f"{name}_actual_{kind}"].astype(np.float64)
                for mode in ("open", "refresh"):
                    predicted = arrays[f"{name}_{mode}_{kind}"].astype(np.float64)
                    if predicted.shape != actual.shape:
                        raise ValueError("feature shape mismatch")
                    value = float(np.mean((predicted - actual)**2))
                    if not np.isclose(value, b[f"{mode}_{kind}_mse"], rtol=2e-6, atol=1e-10):
                        raise ValueError("embedding MSE mismatch")
            if not np.isclose(b["open_score"], min(original["feedback_prior_score"],
                                                   original["feedback_refined_score"]), rtol=3e-5, atol=1e-6):
                raise ValueError("open score mismatch")
            if not np.isclose(b["actual_score"], original["actual_cp17_rgb_score"], rtol=3e-5, atol=1e-6):
                raise ValueError("actual score mismatch")
            if b["physical_distance"] != original["cp17_distance"]:
                raise ValueError("physical target mismatch")
            branches.append(b)
        for mode in ("open", "refresh"):
            pred = [b[f"{mode}_score"] for b in row["branches"]]
            for name, target in (("rgb", [b["actual_score"] for b in row["branches"]]),
                                 ("physical", [b["physical_distance"] for b in row["branches"]])):
                if row["rankings"][f"{mode}_vs_{name}"] != rank_counts(pred, target):
                    raise ValueError("ranking mismatch")
    return {"branches_checked": len(branches), "status": "pass",
            "summary": {field: {"open_mean": float(np.mean([b[f"open_{field}"] for b in branches])),
                                "refresh_mean": float(np.mean([b[f"refresh_{field}"] for b in branches])),
                                "refresh_lower": sum(b[f"refresh_{field}"] < b[f"open_{field}"] for b in branches)}
                        for field in ("visual_mse", "proprio_mse")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report, source = (json.loads(p.read_text()) for p in (args.report, args.source))
    if sha256_file(args.source) != report["source_sha256"]:
        raise ValueError("source hash mismatch")
    path = args.report.with_suffix(".npz")
    if sha256_file(path) != report["arrays_sha256"]:
        raise ValueError("array hash mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        result = audit(report, arrays, source)
    result.update({"report_sha256": sha256_file(args.report), "source_sha256": sha256_file(args.source),
                   "script_sha256": sha256_file(Path(__file__)),
                   "scope": "saved feature math and ranking only, not independent model rerun"})
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
