"""Independently recompute recorded ridge costs from normal equations."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v4_local_metric_covariance_posthoc_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.OUTPUT.with_name(run.OUTPUT.stem + "_audit.json")


def solve(g, residual, q, r):
    delta = np.linalg.solve(g.T @ q @ g + .1 * r, g.T @ q @ residual)
    remaining = residual - g @ delta
    return float((remaining @ q @ remaining + .1 * delta @ r @ delta) / len(residual)), delta


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    data = json.loads(run.OUTPUT.read_text())
    seal.verify(data["seal"])
    pairs = run.prior.source()
    a = np.asarray(data["transform_output"])
    b = np.asarray(data["transform_action"])
    ai, bi = np.linalg.inv(a), np.linalg.inv(b)
    q, r = ai.T @ ai, bi.T @ bi
    max_err = 0.
    changed, robust_changed, covariant_tie_changes = [], [], []
    for record, (source, _) in zip(data["rows"], pairs, strict=True):
        if record["root"] != source["root"] or len(record["candidates"]) != 9:
            raise ValueError("root or candidate mismatch")
        values = []
        for candidate, info in zip(record["candidates"], source["result"]["successor_responses"], strict=True):
            g = np.asarray(info["matrix"])
            residual = np.asarray(source["task"]["goal"]) - np.asarray(info["position"])
            gp, rp = a @ g @ bi, a @ residual
            original, d = solve(g, residual, np.eye(2), np.eye(10))
            covariant, dp = solve(gp, rp, q, r)
            naive, _ = solve(gp, rp, np.eye(2), np.eye(10))
            max_err = max(max_err, abs(original-candidate["original"]),
                          abs(covariant-candidate["transformed"]), abs(naive-candidate["naive"]),
                          float(np.linalg.norm(dp-b @ d)))
            values.append((original, covariant, naive))
        x = np.asarray(values)
        ow, cw, nw = [int(np.argmin(x[:, i])) for i in range(3)]
        if (ow, cw, nw) != (record["original_winner"], record["covariant_winner"], record["naive_winner"]):
            raise ValueError("winner mismatch")
        if ow != cw:
            covariant_tie_changes.append(record["root"])
        if ow != nw:
            changed.append(record["root"])
            if x[nw, 0] - x[ow, 0] > 1e-10 and x[ow, 2] - x[nw, 2] > 1e-10:
                robust_changed.append(record["root"])
    if max_err > 1e-10:
        raise ValueError("normal equation discrepancy")
    seal.verify(data["seal"])
    run.prior.write(OUT, {"source_sha256": run.prior.sha(run.OUTPUT),
        "audit_sha256": run.prior.sha(Path(__file__)), "max_direct_error": max_err,
        "covariant_winner_changes": covariant_tie_changes, "naive_winner_changes": changed,
        "naive_winner_changes_excluding_ties_1e-10": robust_changed,
        "scope": "independent normal equation reconstruction from saved response matrices"})
    print(json.dumps({"max_error": max_err, "naive_changes": changed, "robust_changes": robust_changed}, indent=2))


if __name__ == "__main__":
    main()
