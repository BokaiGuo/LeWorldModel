"""Describe saved suffix-gradient geometry; never select or evaluate actions."""
import argparse
import json
import math
import numpy as np
from experiments.spectral_reachability import agent_p1_fixed_prefix_counts_20261029_v1 as counts
from experiments.spectral_reachability import agent_p1_fixed_prefix_refinement_20261029_v1 as refinement


def analyze(report, receipt, boundary_tolerance=1e-6):
    counts.recount(report, receipt)
    if not math.isfinite(boundary_tolerance) or boundary_tolerance < 0:
        raise ValueError("invalid diagnostic tolerance")
    cap = refinement.F.CAP
    total = outward = near = near_outward = 0
    ratios = []
    projected_linear_deltas = []
    tangent_linear_deltas = []
    for row in report["rows"]:
        result = row["result"]
        if len(result["gradients"]) != 8 or len(result["iterates"]) != 9:
            raise ValueError("incomplete gradient roster")
        for candidate, gradient in zip(result["iterates"][:8], result["gradients"], strict=True):
            x = np.asarray(candidate["logits"], dtype=np.float64)
            g_raw = np.asarray(gradient["raw_gradient"], dtype=np.float64).reshape(-1)
            if x.shape != (10,) or g_raw.shape != (10,) or not np.isfinite(x).all() or not np.isfinite(g_raw).all():
                raise ValueError("invalid saved gradient or logits")
            action = refinement.F.chart(x).reshape(-1).astype(np.float64)
            if refinement.F.bits(action.astype(np.float32)) != candidate["raw_bits"]:
                raise ValueError("chart action identity mismatch")
            g = g_raw * (1 - action ** 2)
            norm = float(np.linalg.norm(x))
            if norm > cap + boundary_tolerance:
                raise ValueError("point outside saved constraint")
            at_boundary = norm >= cap - boundary_tolerance
            points_outward = float(x @ g) < 0  # descent direction is -g
            total += 1
            outward += points_outward
            near += at_boundary
            near_outward += at_boundary and points_outward
            gnorm = float(np.linalg.norm(g))
            if at_boundary and norm > 0 and gnorm > 0:
                ratios.append(float(np.linalg.norm(g - x * (x @ g) / (norm * norm)) / gnorm))
                if points_outward:
                    tangent = g - x * (x @ g) / (norm * norm)
                    projected_linear_deltas.append(float(g @ (refinement.F.proposal(x, g) - x)))
                    tangent_linear_deltas.append(float(g @ (refinement.F.proposal(x, tangent) - x)))
    return {"scope": "saved_model_gradient_geometry_only", "steps": total,
            "outward_descent": outward, "near_boundary": near,
            "near_boundary_outward_descent": near_outward, "boundary_tolerance": boundary_tolerance,
            "radius": cap, "tangent_ratio_count": len(ratios),
            "tangent_ratio_min": min(ratios) if ratios else None,
            "tangent_ratio_median": float(np.median(ratios)) if ratios else None,
            "tangent_ratio_max": max(ratios) if ratios else None,
            "linear_comparison_count": len(projected_linear_deltas),
            "tangent_lower_linear_delta": sum(t < p for t, p in zip(tangent_linear_deltas, projected_linear_deltas, strict=True)),
            "projected_linear_delta_mean": float(np.mean(projected_linear_deltas)) if projected_linear_deltas else None,
            "tangent_linear_delta_mean": float(np.mean(tangent_linear_deltas)) if tangent_linear_deltas else None,
            "linear_comparison_scope": "suffix gradient linearization only; tangent candidates not evaluated",
            "physical_release": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "receipt", "receipt-sha"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    report, receipt = counts.load_verified(args.report, args.receipt, args.receipt_sha)
    print(json.dumps(analyze(report, receipt), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
