"""Audit B0 optimizer/candidate identity without launching new experiments."""
import json
from pathlib import Path

import numpy as np

from . import p2_cp15_pair_integrity_20261102_codexexclusive as prior
from . import p2_shared_capture_launch_20261102_exclusive as current

ROOT = current.ROOT
NEW = ROOT / "results/p2_shared_rgb_pair_wiring_20261102_codexexclusive_RUN_20261102_p2continuation_host01"
REPORT_SHA = "d39f8e4f5f292034331b8a60b3fe0682b04d81206fd904200db2295a38e3ef2d"
PANEL = ROOT / "results/codex_p2_cp15_symmetric_spectral_20261030_v1_PREP_01.json"
need = current.need


def compare(pair, case, old):
    a, b = pair["context"], case["context"]
    need(a["layout"] == b["layout"], "layout mismatch")
    for field in ("latent_history", "action_history"):
        need(np.array_equal(np.asarray(a[field]), np.asarray(b[field])), field + " mismatch")
    need(np.array_equal(np.asarray(pair["goal"]), np.asarray(case["goal"])), "goal mismatch")
    result = []
    for arm, new, previous in zip(("joint_multistart", "prefix_chain"),
                                  pair["branches"], old, strict=True):
        need(new["arm"] == previous["arm"] == arm, "arm mismatch")
        need(np.array_equal(new["root_jacobian"], previous["root_jacobian"]), "root Jacobian mismatch")
        need(len(new["candidates"]) == len(previous["candidates"]) == 16, "candidate denominator")
        fields = (["index", "raw_float32_hex", "cost"] if arm == "joint_multistart"
                  else ["index", "first_raw_float32_hex", "second_raw_float32_hex", "cost"])
        need(all(all(x[k] == y[k] for k in fields) for x, y in
                 zip(new["candidates"], previous["candidates"], strict=True)), "candidate mismatch")
        need(new["winner"]["index"] == previous["winner"]["index"], "winner mismatch")
        result.append({"arm": arm, "candidate_action_and_cost_matches": 16,
                       "winner_index": new["winner"]["index"], "root_jacobian_exact": True})
    return {"rows": result, "context_values_exact": True, "goal_values_exact": True,
            "new_context_keys": sorted(a), "prior_context_keys": sorted(b),
            "new_context_sha256": current.wire.kernel.old.digest(a),
            "prior_context_sha256": current.wire.kernel.old.digest(b)}


def audit():
    current.audit(NEW, REPORT_SHA)
    historical = prior.audit_saved()
    parent = json.loads(prior.PREP.read_bytes())["parent"]
    need(Path(parent["panel_path"]) == PANEL and current.wire.checkpoint.sha(PANEL) == parent["panel_sha256"],
         "pinned old panel")
    panel = json.loads(PANEL.read_bytes())
    pair = json.loads((NEW / "pair.json").read_bytes())
    old = [json.loads((prior.reuse.OLD_RUN / f"branch_{i}.json").read_bytes()) for i in range(2)]
    result = compare(pair, panel["cases"][0], old)
    result.update({"historical_fixed_slots": len(historical["pairs"]),
                   "distinct_known_roots": historical["distinct_known_roots"],
                   "duplicate_bases": historical["duplicate_bases"],
                   "historical_model_receipt_sha256": prior.NEW_SAVED_SHA,
                   "historical_physical_reuse_sha256": prior.REUSE_SAVED_SHA,
                   "model_report_sha256": REPORT_SHA,
                   "new_physical_or_model_calls": 0, "P1_admission": False,
                   "P2_admission": False, "independent_confirmation": False})
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, allow_nan=False))
