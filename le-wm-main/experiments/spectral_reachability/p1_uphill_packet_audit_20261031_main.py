"""Audit executed packets and score changes in the fixed midpoint experiments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_uphill_midpoint_source_compare_20261031_main as matched
from . import p1_uphill_midpoint_feedback_physical_20261031_main as real
from . import p1_uphill_imagined_midpoint_physical_20261031_main as imagined

prior = matched.prior
ROOT = matched.ROOT
NAME = "p1_uphill_packet_audit_20261031_main"
OPEN_COMPARE = ROOT / "results/p1_uphill_feedback_compare_20261031_main_RUN01.json"
OPEN_COMPARE_SHA = "04fae5c52e4e320ed305e904d4595612fa7083dd52e6fdfe525601d1f595300a"
SOURCE_COMPARE = ROOT / "results/p1_uphill_midpoint_source_compare_20261031_main_RUN01.json"
SOURCE_COMPARE_SHA = "d13f9a8c43a2f3eaa9fc3f3ce19d60855f79275af31b2c7db309c87b1824a6e4"


def verify_actions(actions, events, returned):
    prior.need(actions.shape == (10, 2) and np.isfinite(actions).all(), "action shape or finite")
    prior.need(len(events) == returned and 0 <= returned <= 10, "returned count")
    for i, event in enumerate(events):
        prior.need(event["step"] == i + 1 and event["actual_bits"] == actions[i].tobytes().hex(),
                   "executed action mismatch")


def analyze(original, real_plan, imagined_plan, open_cmp, source_cmp, reports, get_events):
    prior.need(all(len(x["rows"]) == 24 for x in (original, real_plan, imagined_plan, open_cmp, source_cmp))
               and all(len(x["shards"]) == 24 for x in reports), "roster")
    rows = []
    for i, (o, r, m, oc, sc) in enumerate(zip(
            original["rows"], real_plan["rows"], imagined_plan["rows"],
            open_cmp["rows"], source_cmp["rows"], strict=True)):
        prior.need(all(x["index"] == i for x in (o, r, m, oc, sc))
                   and all(x["role"] == o["role"] for x in (r, m, oc, sc)), "identity")
        raw = np.asarray(o["selected_actions"], dtype="<f4").reshape(10, 2)
        prior.need(hashlib.sha256(raw.tobytes()).hexdigest() == o["selected_action_sha256"], "original hash")
        arms = {}
        for name, plan, success, status, returned in (
            ("real", r, sc["real_last_success"], sc["real_status"], sc["real_returned_suffix"]),
            ("imagined", m, sc["imagined_last_success"], sc["imagined_status"], sc["imagined_returned_suffix"]),
        ):
            first, second = bytes.fromhex(plan["first_packet_hex"]), bytes.fromhex(plan["second_packet_hex"])
            prior.need(len(first) == len(second) == 40 and first == raw[:5].tobytes(), "packet binding")
            data = np.frombuffer(first + second, dtype="<f4").reshape(10, 2)
            prior.need(plan["chosen"] == ("optimized" if plan["optimized_score"] < plan["baseline_score"]
                                         else "original"), "choice")
            if plan["chosen"] == "original":
                prior.need(second == raw[5:].tobytes(), "original fallback")
            verify_actions(data, get_events(name, i, reports), returned)
            arms[name] = {
                "chosen": plan["chosen"], "score_delta": plan["optimized_score"] - plan["baseline_score"],
                "second_packet_l2_change": float(np.linalg.norm(data[5:].astype(np.float64) - raw[5:])),
                "second_packet_max_abs": float(np.abs(data[5:]).max()),
                "second_packet_clamped_coordinates": int(np.sum(np.abs(data[5:]) == 1)),
                "success": success, "status": status, "returned": returned,
            }
        verify_actions(raw, get_events("open", i, reports), oc["open_returned_suffix"])
        prior.need(oc["first5_exact"] and sc["first5_exact"]
                   and oc["feedback_last_success"] == arms["real"]["success"], "paired source")
        rows.append({"index": i, "root_slot": oc["root_slot"], "condition": oc["condition"],
                     "arm": oc["arm"], "role": oc["role"], "open_success": oc["open_last_success"],
                     "open_status": oc["open_status"], "open_returned": oc["open_returned_suffix"], **arms})
    return {"schema": NAME, "rows": rows, "P1_admission": False, "independent_confirmation": False,
            "limitations": "Post-outcome fixed known roots and repeated arms; execution checks do not prove "
                           "observation or model calibration correctness. Censored endpoints not imputed."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    out = Path(p.parse_args().output)
    prior.need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
               and not out.exists() and not out.is_symlink(), "output")
    specs = [(prior.open_run.paired.physical.PLAN, prior.open_run.paired.physical.PLAN_SHA),
             (real.PLAN, real.PLAN_SHA), (imagined.PLAN, imagined.PLAN_SHA),
             (OPEN_COMPARE, OPEN_COMPARE_SHA), (SOURCE_COMPARE, SOURCE_COMPARE_SHA),
             (prior.open_run.RUN / "run_report.json", prior.open_run.REPORT_SHA),
             (prior.FB_DIR / "run_report.json", prior.FB_REPORT),
             (matched.IM_DIR / "run_report.json", matched.IM_REPORT)]
    data = [prior.open_run.paired.read(*s) for s in specs]
    directories = {"open": prior.open_run.RUN, "real": prior.FB_DIR, "imagined": matched.IM_DIR}
    report_index = {"open": 0, "real": 1, "imagined": 2}

    def events(name, i, reports):
        shard = reports[report_index[name]]["shards"][i]
        prior.need(shard["index"] == i, "shard index")
        return prior.traces.transitions(directories[name] / f"row_{i:02d}", shard["report_sha256"])

    result = analyze(*data[:5], data[5:], events)
    result["sources"] = {str(path): digest for path, digest in specs}
    result["code_sha256"] = prior.feedback.sha(Path(__file__).resolve())
    prior.need(all(prior.feedback.sha(path) == digest for path, digest in specs), "source drift")
    print(json.dumps({"sha256": prior.feedback.original.physical.design.run._write_new(out, result),
                      "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
