"""Check saved dev305 spectral outputs without rerunning neural inference."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from . import p1_dev305_goalrank_spectral_20261102 as run

CONTRACT_SHA = "f318092c7df0d36aacd3b7a646ed5a3c3aebe0035a19831b9ebff178954e79ed"
REPORT_SHA = "17a8e42dafeffaf2d2eaf2184009470c337f4f191e9c91b26d43c278c6285f18"


def analyze():
    root = run.ROOT / "results"
    contract = run.base.read(root / f"{run.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    run.need(contract == run.protocol(), "contract and physical source binding")
    directory = root / f"{run.NAME}_RUN_DEV01"
    report = run.base.read(directory / "report.json", REPORT_SHA)
    run.need(report["contract_sha256"] == CONTRACT_SHA
             and report["model_before"] == report["model_after"] == contract["model_hash"],
             "contract and frozen model")
    counts = report["counts"]
    run.need(counts["vjp_attempted"] == counts["vjp_returned"] == 384
             and counts["svd_calls"] == 2, "derivative ledger")
    summary = []
    for role, row in zip(contract["roles"], report["rows"], strict=True):
        run.need(role["arm"] == row["arm"] and role["prefix_sha256"] == row["prefix_sha256"]
                 and run.base.read(directory / (row["arm"] + ".json")) == row, "row binding")
        jac = np.asarray(row["jacobian"], float)
        direction = np.asarray(row["fd_direction"], float)
        predicted = np.asarray(row["fd_jacobian"], float)
        observed = np.asarray(row["fd_observed"], float)
        run.need(jac.shape == (192, 20) and np.isfinite(jac).all()
                 and np.allclose(jac @ direction, predicted, rtol=1e-6, atol=1e-7)
                 and np.isclose(np.linalg.norm(observed-predicted)/(np.linalg.norm(predicted)+1e-12),
                                row["fd_relative_error"], rtol=1e-6), "finite difference")
        for candidate in row["candidates"]:
            actions = np.asarray(candidate["actions"], dtype="<f4")
            run.need(actions.shape == (2, 5, 2) and np.isfinite(actions).all()
                     and (np.abs(actions) <= 1).all()
                     and hashlib.sha256(actions.tobytes()).hexdigest() == candidate["actions_sha256"],
                     "bounded action witness")
        scores = [c["score"] for c in row["candidates"]]
        run.need(len(scores) == 9 and np.isfinite(scores).all()
                 and int(np.argmin(scores)) == row["selected_index"], "selector")
        selected = row["candidates"][row["selected_index"]]
        summary.append({"arm": row["arm"], "selected_kind": selected["kind"],
                        "selected_score": selected["score"], "anchor_score": scores[0],
                        "fd_relative_error": row["fd_relative_error"]})
    return {"schema": run.NAME + ".saved.v1", "rows": summary,
            "scope": "saved source, witness, finite-difference arithmetic and selection; no neural recomputation or physical execution",
            "p1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(run.base.write_new(parser.parse_args().output, analyze()))
