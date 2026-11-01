"""Verify and pair the sealed midpoint multistart replay with the open baseline."""
import json
from pathlib import Path

from . import p1_uphill_trust_compare_20261031_main as comparison
from . import p1_uphill_midpoint_multistart_physical_20261031 as multistart

prior = comparison.prior
ROOT = prior.ROOT
RUN = ROOT / "results/p1_uphill_midpoint_multistart_physical_20261031_run_RUN01"
SAVED = ROOT / "results/p1_uphill_midpoint_multistart_physical_20261031_consume_RUN01.json"
OUT = ROOT / "results/p1_uphill_midpoint_multistart_compare_20261031_RUN01.json"


def main():
    specs = [
        (prior.open_run.paired.ANCHORS, prior.open_run.paired.ANCHORS_SHA),
        (prior.open_run.paired.CONSUME, prior.open_run.paired.CONSUME_SHA),
        (SAVED, "bac2688439b9076f7b883171c1dd4c0fd0bb03e4cd9f48ff73bc07b0e55dda48"),
        (prior.open_run.RUN / "run_report.json", prior.open_run.REPORT_SHA),
        (RUN / "run_report.json", "c48f780c2d81faf9327414cd561cf55ebbecb4ac2dce214a24be4fea87dbc8f0"),
        (multistart.PLAN, multistart.PLAN_SHA),
    ]
    anchors, ref, actual, ref_report, report, plans = [
        prior.open_run.paired.read(*spec) for spec in specs
    ]

    def transitions(i, a, b):
        return (
            prior.traces.transitions(prior.open_run.RUN / f"row_{i:02d}", a["report_sha256"]),
            prior.traces.transitions(RUN / f"row_{i:02d}", b["report_sha256"]),
        )

    def actions(i):
        return prior.traces.transitions(RUN / f"row_{i:02d}", report["shards"][i]["report_sha256"])

    result = comparison.compare(anchors, ref, actual, ref_report, report, transitions, plans, actions)
    result["schema"] = "p1_uphill_midpoint_multistart_compare_20261031"
    result["limitations"] = (
        "Known roots and repeated arms; midpoint feedback has extra optimization cost; "
        "early-stopped endpoints are null; not equal-budget or independent confirmation."
    )
    result["sources"] = {str(path): digest for path, digest in specs}
    result["code_sha256"] = prior.feedback.sha(Path(__file__).resolve())
    prior.need(all(prior.feedback.sha(path) == digest for path, digest in specs), "source drift")
    digest = prior.feedback.original.physical.design.run._write_new(OUT, result)
    print(json.dumps({"sha256": digest, "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
