"""Verify metric-covariant ridge value and demonstrate untransformed metric sensitivity."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v4_paired_retention_quality_20261104_codex as prior
from experiments.spectral_reachability import d2p_local_compiler as local
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_LOCAL_METRIC_COVARIANCE_POSTHOC_PROTOCOL_20261104_CODEX.md"
OUTPUT = ROOT / "results/pc_v4_local_metric_covariance_posthoc_20261104_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    pairs = prior.source()
    deps = seal.capture([Path(__file__), PROTOCOL, Path(local.__file__), *prior.PLANS,
                         *prior.AUDITS, prior.RULES])
    A = np.array([[2., .4], [0., .5]])
    B = np.diag(np.linspace(.5, 2., 10))
    inv_a, inv_b = np.linalg.inv(A), np.linalg.inv(B)
    rows = []
    for row, _ in pairs:
        goal = np.asarray(row["task"]["goal"])
        candidates = []
        for response in row["result"]["successor_responses"]:
            g = np.asarray(response["matrix"])
            residual = goal - np.asarray(response["position"])
            gp, rp = A @ g @ inv_b, A @ residual
            original = local.local_geometry(g, residual, np.eye(2), np.eye(10), .1)
            transformed = local.local_geometry(gp, rp, inv_a, B, .1)
            naive = local.local_geometry(gp, rp, np.eye(2), np.eye(10), .1)
            delta = np.asarray(original["raw_delta"])
            direct = np.linalg.solve(g.T @ g + .1 * np.eye(10), g.T @ residual)
            delta_p = np.asarray(transformed["raw_delta"])
            transformed_direct = B @ direct
            record = {"original": original["linear_reach_value"],
                      "transformed": transformed["linear_reach_value"],
                      "naive": naive["linear_reach_value"],
                      "source_value_error": abs(original["linear_reach_value"] - response["linear_reach_value"]),
                      "value_error": abs(original["linear_reach_value"] - transformed["linear_reach_value"]),
                      "direct_error": float(np.linalg.norm(delta - direct)),
                      "transformed_direct_error": float(np.linalg.norm(delta_p - transformed_direct)),
                      "singular_error": float(np.max(np.abs(np.asarray(original["singular_values"])
                                                             - transformed["singular_values"])))}
            if max(record["source_value_error"], record["value_error"], record["direct_error"],
                   record["transformed_direct_error"], record["singular_error"]) > 1e-10:
                raise ValueError("ridge covariance or source value mismatch")
            candidates.append(record)
        if len(candidates) != 9:
            raise ValueError("unexpected bank size")
        rows.append({"root": row["root"], "candidates": candidates,
                     "original_winner": int(np.argmin([c["original"] for c in candidates])),
                     "covariant_winner": int(np.argmin([c["transformed"] for c in candidates])),
                     "naive_winner": int(np.argmin([c["naive"] for c in candidates]))})
    keys = ("source_value_error", "value_error", "direct_error", "transformed_direct_error", "singular_error")
    maxima = {k: max(c[k] for row in rows for c in row["candidates"]) for k in keys}
    seal.verify(deps)
    prior.write(OUTPUT, {"scope": "post-hoc 24 exposed roots; local unconstrained ridge only",
        "seal": deps, "transform_output": A.tolist(), "transform_action": B.tolist(),
        "damping": .1, "roots": len(rows), "candidates": sum(len(r["candidates"]) for r in rows),
        "max_errors": maxima,
        "covariant_winner_changes": sum(r["original_winner"] != r["covariant_winner"] for r in rows),
        "naive_winner_changes": sum(r["original_winner"] != r["naive_winner"] for r in rows),
        "rows": rows})
    print(json.dumps({"max_errors": maxima,
          "covariant_winner_changes": sum(r["original_winner"] != r["covariant_winner"] for r in rows),
          "naive_winner_changes": sum(r["original_winner"] != r["naive_winner"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
