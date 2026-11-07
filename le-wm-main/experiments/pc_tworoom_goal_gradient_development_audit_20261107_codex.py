"""Reconstruct gradient candidate identities and compare shared physical parents."""
import json

import numpy as np

from experiments import pc_tworoom_goal_gradient_development_20261107_codex as run
from experiments.pc_input_permissions import file_hash
from experiments.pc_rgb_goal_gradient_candidates_20261107_codex import make_candidates

OUT = run.ROOT / "results/pc_tworoom_goal_gradient_development_audit_20261107_codex.json"


def audit(plan, physical, old_plan, old_physical):
    if (plan["dependencies"] != run.dependencies()
            or physical["plan_sha256"] != file_hash(run.PLAN)
            or old_physical["plan_sha256"] != file_hash(run.prior.PLAN)):
        raise ValueError("source drift")
    summary = []
    for g, p, s, q in zip(plan["rows"], physical["rows"], old_plan["rows"], old_physical["rows"], strict=True):
        if len({g["episode"], p["episode"], s["episode"], q["episode"]}) != 1:
            raise ValueError("task identity mismatch")
        parent = np.asarray(s["arms"]["spectral"]["candidates"][0], np.float32)
        candidates, direction = make_candidates(parent, g["gradient"])
        np.testing.assert_array_equal(candidates, np.asarray(g["candidates"], np.float32))
        np.testing.assert_allclose(direction, g["direction"], rtol=0, atol=1e-12)
        scores = np.asarray(g["scores"])
        if (scores.shape != (5,) or not np.isfinite(scores).all()
                or g["selected_index"] != int(np.argmin(scores))
                or p["selected_index"] != g["selected_index"] or len(p["candidates"]) != 5):
            raise ValueError("selection mismatch")
        for item in p["candidates"]:
            if item["ever_success"] != any(t["success"] for t in item["trace"]):
                raise ValueError("physical success mismatch")
        if p["candidates"][0] != q["arms"]["spectral"]["candidates"][0]:
            raise ValueError("physical parent trace mismatch")
        error = abs(g["finite_difference_directional"] - g["autodiff_directional"])
        if g["gradient_check_pass"] != (error <= .1 + .01 * abs(g["autodiff_directional"])):
            raise ValueError("derivative check mismatch")
        summary.append({"episode": g["episode"], "selected_index": g["selected_index"],
                        "success": p["candidates"][g["selected_index"]]["ever_success"],
                        "candidate_successes": sum(x["ever_success"] for x in p["candidates"]),
                        "initial_success": g["initial_success"], "gradient_check_pass": g["gradient_check_pass"]})
    return summary


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan, physical = json.loads(run.PLAN.read_text()), json.loads(run.PHYSICAL.read_text())
    old_plan, old_physical = json.loads(run.prior.PLAN.read_text()), json.loads(run.prior.PHYSICAL.read_text())
    rows = audit(plan, physical, old_plan, old_physical)
    with OUT.open("x") as stream:
        json.dump({"runner_sha256": file_hash(__file__), "plan_sha256": file_hash(run.PLAN),
                   "physical_sha256": file_hash(run.PHYSICAL), "rows": rows,
                   "scope": "stored-data identity audit, no independent model rerun"},
                  stream, indent=2, allow_nan=False)
    print(f"Audited {len(rows)} paired tasks and shared physical parent traces.")


if __name__ == "__main__":
    main()
