"""Descriptive all-pairs analysis of sealed real RGB embeddings and physical states."""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from . import codex_p1_cp5_terminal_rgb_gap_20261030_main as gap
from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base

NAME = "codex_p1_pusher_latent_pairs_20261030_main"
EMB = base.ROOT / "results/codex_p1_multistart20_terminal_gap_20261030_main_RUN01.json"
EMB_SHA = "3cfa240e51b8765e4d80d254d04d4439555581703940384d0366272704eb0235"
PHYS = base.ROOT / "results/codex_p1_multistart20_physical_20261030_main_SAVED_RUN01.json"
PHYS_SHA = "c7cbdf2109d79cf29642659e2f751c8b20a75991cbb30e0427d2c4cdeea25454"


def build(emb, physical):
    if len(emb["rows"]) != 24 or len(physical["rows"]) != 24:
        raise ValueError("fixed panel size")
    rows, excluded = [], []
    for i, (e, p) in enumerate(zip(emb["rows"], physical["rows"], strict=True)):
        if e["index"] != p["index"] or e["index"] != i:
            raise ValueError("index mismatch")
        if e["status"] != p["status"] or e["returned_suffix"] != p["returned_suffix"]:
            raise ValueError("status mismatch")
        if e["status"] != "probe_completed":
            excluded.append({"index": i, "status": e["status"], "returned_suffix": e["returned_suffix"]})
            continue
        if any(e["identity"][k] != p[k] for k in ("root_slot", "condition", "role", "logical_pair")):
            raise ValueError("identity mismatch")
        state = np.asarray(p["completed_endpoint"], dtype=np.float64)
        z = np.asarray(e["actual_latent"], dtype=np.float64)
        goal = np.asarray(e["goal_latent"], dtype=np.float64)
        if state.shape != (7,) or z.shape != (192,) or goal.shape != (192,):
            raise ValueError("array shape")
        if not np.isfinite(state).all() or not np.isfinite(z).all() or not np.isfinite(goal).all():
            raise ValueError("nonfinite input")
        mse = float(np.mean((z - goal) ** 2))
        if abs(mse - e["actual_goal_mse"]) > 1e-6 or e["physical_metrics"] != p["endpoint_metrics"]:
            raise ValueError("saved metric mismatch")
        rows.append({"index": i, "identity": e["identity"], "state": state, "z": z,
                     "goal": goal, "mse": mse, "metrics": p["endpoint_metrics"]})
    pairs = []
    for a, b in itertools.combinations(rows, 2):
        if any(a["identity"][k] != b["identity"][k] for k in ("root_slot", "condition")):
            continue
        if not np.array_equal(a["goal"], b["goal"]):
            raise ValueError("different goal within group")
        sa, sb = a["state"], b["state"]
        angle = math.atan2(math.sin(sa[4] - sb[4]), math.cos(sa[4] - sb[4]))
        latent_diff = a["mse"] - b["mse"]
        pusher_diff = a["metrics"]["pusher_position_error"] - b["metrics"]["pusher_position_error"]
        pairs.append({"indices": [a["index"], b["index"]], "identity": {
                          k: a["identity"][k] for k in ("root_slot", "condition")},
                      "object_xy_separation": float(np.linalg.norm(sa[2:4] - sb[2:4])),
                      "object_angle_separation": abs(angle),
                      "pusher_xy_separation": float(np.linalg.norm(sa[:2] - sb[:2])),
                      "latent_separation_mse": float(np.mean((a["z"] - b["z"]) ** 2)),
                      "latent_goal_mse_difference": latent_diff,
                      "pusher_goal_error_difference": pusher_diff,
                      "order": "tie" if latent_diff == 0 or pusher_diff == 0 else
                               ("agree" if latent_diff * pusher_diff > 0 else "disagree")})
    return {"schema": NAME, "status": "descriptive", "completed_rows": len(rows), "excluded": excluded, "pairs": pairs,
            "order_counts": {k: sum(x["order"] == k for x in pairs) for k in ("agree", "disagree", "tie")},
            "limitations": "Same-root post-outcome pairs are dependent; object and pusher co-vary; "
                           "this is not a controlled intervention, a planner or an independent confirmation.",
            "P1_admission": False, "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    if (not out.is_absolute() or out.parent != base.ROOT / "results" or
            not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink()):
        raise ValueError("exclusive output")
    emb, _ = gap.read(EMB, EMB_SHA)
    physical, _ = gap.read(PHYS, PHYS_SHA)
    result = build(emb, physical)
    result["sources"] = {str(EMB): EMB_SHA, str(PHYS): PHYS_SHA, str(Path(__file__).resolve()): base.sha(__file__)}
    digest = base.write(out, result)
    print(json.dumps({"sha256": digest, "rows": result["completed_rows"],
                      "pairs": len(result["pairs"]), "order_counts": result["order_counts"]}))


if __name__ == "__main__":
    main()
