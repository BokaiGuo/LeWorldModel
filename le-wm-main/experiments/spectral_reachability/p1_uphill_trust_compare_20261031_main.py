"""Compare sealed trust-region physical outcomes with both fixed reference arms."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_uphill_feedback_compare_20261031_main as prior
from . import p1_uphill_packet_audit_20261031_main as audit
from . import p1_uphill_midpoint_trust_physical_20261031_main as trust

ROOT = prior.ROOT
NAME = "p1_uphill_trust_compare_20261031_main"
RUN = ROOT / "results/p1_uphill_midpoint_trust_physical_20261031_main_run_RUN01"
SAVED = ROOT / "results/p1_uphill_midpoint_trust_physical_20261031_main_consume_RUN01.json"


def compare(anchors, reference, actual, reference_report, actual_report, transitions, plans, actions):
    result = prior.compare(anchors, reference, actual, reference_report, actual_report, transitions)
    prior.need(len(plans["rows"]) == len(result["rows"]) == 24, "plan roster")
    for i, (row, plan) in enumerate(zip(result["rows"], plans["rows"], strict=True)):
        prior.need(row["index"] == plan["index"] == i and row["role"] == plan["role"], "plan identity")
        packet = bytes.fromhex(plan["first_packet_hex"] + plan["second_packet_hex"])
        prior.need(len(packet) == 80, "plan packet")
        audit.verify_actions(np.frombuffer(packet, dtype="<f4").reshape(10, 2), actions(i),
                             row["feedback_returned_suffix"])
    result["schema"] = NAME
    result["limitations"] = ("Known roots and repeated arms; post-outcome radius and method; "
                             "early-stopped endpoints are null; not P1 admission or independent confirmation.")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", choices=("open", "real"), required=True)
    p.add_argument("--report-sha", required=True)
    p.add_argument("--saved-sha", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    prior.need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
               and not out.exists() and not out.is_symlink(), "output")
    if args.reference == "open":
        reference, reference_sha = prior.open_run.paired.CONSUME, prior.open_run.paired.CONSUME_SHA
        ref_dir, ref_report_sha = prior.open_run.RUN, prior.open_run.REPORT_SHA
    else:
        reference, reference_sha = prior.FB_SAVED, prior.FB_SAVED_SHA
        ref_dir, ref_report_sha = prior.FB_DIR, prior.FB_REPORT
    specs = [(prior.open_run.paired.ANCHORS, prior.open_run.paired.ANCHORS_SHA),
             (reference, reference_sha), (SAVED, args.saved_sha),
             (ref_dir / "run_report.json", ref_report_sha), (RUN / "run_report.json", args.report_sha),
             (trust.PLAN, trust.PLAN_SHA)]
    anchors, ref, actual, ref_report, report, plans = [prior.open_run.paired.read(*x) for x in specs]

    def transitions(i, a, b):
        return (prior.traces.transitions(ref_dir / f"row_{i:02d}", a["report_sha256"]),
                prior.traces.transitions(RUN / f"row_{i:02d}", b["report_sha256"]))

    def actions(i):
        return prior.traces.transitions(RUN / f"row_{i:02d}", report["shards"][i]["report_sha256"])

    result = compare(anchors, ref, actual, ref_report, report, transitions, plans, actions)
    result["reference"] = args.reference
    result["sources"] = {str(path): digest for path, digest in specs}
    result["code_sha256"] = prior.feedback.sha(Path(__file__).resolve())
    prior.need(all(prior.feedback.sha(path) == digest for path, digest in specs), "source drift")
    digest = prior.feedback.original.physical.design.run._write_new(out, result)
    print(json.dumps({"sha256": digest, "rows": len(result["rows"]), "reference": args.reference}))


if __name__ == "__main__":
    main()
