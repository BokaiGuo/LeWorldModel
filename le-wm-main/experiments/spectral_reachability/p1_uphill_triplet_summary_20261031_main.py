"""Join sealed prefix measurements and three physical suffix strategies by task."""
import argparse
import json
from pathlib import Path

from . import p1_uphill_packet_audit_20261031_main as audit
from . import p1_uphill_multistart_pairing_20261030_main as base

ROOT = base.ROOT
NAME = "p1_uphill_triplet_summary_20261031_main"
PAIR = ROOT / "results/p1_uphill_multistart_pairing_20261030_main_RUN01.json"
PAIR_SHA = "b69b9d38b7f926c31648dc00d0445e710a1014bdff2dff216a979c43ac397ff4"
TRUST_OPEN = ROOT / "results/p1_uphill_trust_compare_20261031_main_open_RUN01.json"
TRUST_OPEN_SHA = "578f22ecf6d4fe92716ccb00c7a40fdf2d7438c57348a4116c0e5e8d77949d9b"
TRUST_REAL = ROOT / "results/p1_uphill_trust_compare_20261031_main_real_RUN01.json"
TRUST_REAL_SHA = "56945707fed6cac3c13d218b8e0776465c35a9a3bdfc663f9cca1a939b0e24f8"


def join(pair, real, trust_open, trust_real):
    base.need(all(len(x["rows"]) == 24 for x in (pair, real, trust_open, trust_real)), "roster")
    joined = []
    for i, (p, r, o, t) in enumerate(zip(
            pair["rows"], real["rows"], trust_open["rows"], trust_real["rows"], strict=True)):
        base.need(all(x["index"] == i for x in (p, r, o, t))
                  and all(all(x[k] == p[k] for k in ("root_slot", "condition", "arm", "role", "logical_pair"))
                          for x in (r, o, t)), "identity")
        base.need(all(x["first5_exact"] for x in (r, o, t))
                  and p["last_native_success"] == r["open_last_success"] == o["open_last_success"]
                  and r["feedback_last_success"] == t["open_last_success"]
                  and o["feedback_last_success"] == t["feedback_last_success"]
                  and p["status"] == r["open_status"] == o["open_status"]
                  and r["feedback_status"] == t["open_status"]
                  and o["feedback_status"] == t["feedback_status"], "outcome binding")
        joined.append({"index": i, "role": p["role"], "prefix_minus_root": p["prefix_minus_root"],
                       "prefix": p["prefix"], "model_best_score": p["model_best_score"],
                       "model_score_minus_zero_prefix": p["model_score_minus_zero_prefix"],
                       "open": {"success": p["last_native_success"], "status": p["status"],
                                "returned": p["returned_suffix"], "endpoint": p["endpoint"]},
                       "real": {"success": r["feedback_last_success"], "status": r["feedback_status"],
                                "returned": r["feedback_returned_suffix"], "endpoint": r["feedback_endpoint"]},
                       "trust": {"success": o["feedback_last_success"], "status": o["feedback_status"],
                                 "returned": o["feedback_returned_suffix"], "endpoint": o["feedback_endpoint"]}})
    groups = []
    for i in range(8):
        original = pair["rows"][3*i:3*i+3]
        chunk = joined[3*i:3*i+3]
        base.need([x["role"] for x in chunk] == list(base.ROLES)
                  and all(x["logical_pair"] == i and x["task_id"] == original[0]["task_id"]
                          and x["physical_root_sha256"] == original[0]["physical_root_sha256"]
                          for x in original), "task identity")
        groups.append({"logical_pair": i, "root_slot": original[0]["root_slot"],
                       "condition": original[0]["condition"], "arm": original[0]["arm"],
                       "task_id": original[0]["task_id"], "root": original[0]["root"],
                       "physical_root_sha256": original[0]["physical_root_sha256"], "prefixes": chunk})
    return {"schema": NAME, "groups": groups, "P1_admission": False,
            "independent_confirmation": False, "equal_budget_certified": False,
            "limitations": "Known roots, shared logical arms and post-outcome analysis. Model scores are "
                           "not physical metrics; independent suffix planning is not a common suffix intervention. "
                           "Early-stop fixed-horizon endpoints are null."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    base.need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
              and not out.exists() and not out.is_symlink(), "output")
    specs = [(PAIR, PAIR_SHA), (audit.OPEN_COMPARE, audit.OPEN_COMPARE_SHA),
             (TRUST_OPEN, TRUST_OPEN_SHA), (TRUST_REAL, TRUST_REAL_SHA)]
    result = join(*(base.read(*spec) for spec in specs))
    result["sources"] = {str(p): h for p, h in specs}
    result["code_sha256"] = base.physical.sha(Path(__file__).resolve())
    base.need(all(base.physical.sha(p) == h for p, h in specs), "source drift")
    digest = base.physical.physical.design.run._write_new(out, result)
    print(json.dumps({"groups": len(result["groups"]), "sha256": digest}))


if __name__ == "__main__":
    main()
