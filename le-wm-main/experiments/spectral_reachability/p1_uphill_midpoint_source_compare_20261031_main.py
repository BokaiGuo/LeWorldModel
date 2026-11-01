"""Compare matched real versus imagined midpoint replanning on sealed physical traces."""
import argparse
import json
from pathlib import Path

from . import p1_uphill_feedback_compare_20261031_main as prior

ROOT = prior.ROOT
NAME = "p1_uphill_midpoint_source_compare_20261031_main"
IM_DIR = ROOT / "results/p1_uphill_imagined_midpoint_physical_20261031_main_run_RUN01"
IM_REPORT = "3b4a12aa2828c6210753791e5853b41ebc4ee73733e168e0ce90405cbd308432"
IM_SAVED = ROOT / "results/p1_uphill_imagined_midpoint_physical_20261031_main_consume_RUN01.json"
IM_SAVED_SHA = "dc505c53613a4ec53b59f90ae97fc7c1f2994202642dcfa72d6ceec49155351e"


def compare(anchors, real, imagined, real_report, imagined_report, transitions):
    result = prior.compare(anchors, real, imagined, real_report, imagined_report, transitions)
    for row in result["rows"]:
        for old, new in (
            ("open_status", "real_status"), ("feedback_status", "imagined_status"),
            ("open_returned_suffix", "real_returned_suffix"),
            ("feedback_returned_suffix", "imagined_returned_suffix"),
            ("open_last_success", "real_last_success"),
            ("feedback_last_success", "imagined_last_success"),
            ("open_endpoint", "real_endpoint"), ("feedback_endpoint", "imagined_endpoint"),
            ("feedback_minus_open", "imagined_minus_real"),
        ):
            row[new] = row.pop(old)
    result.update(schema=NAME, limitations=(
        "Post-outcome development on known roots; repeated arms are not independent. "
        "Same first five verified; midpoint encoding and prediction costs differ; "
        "early-stopped fixed-horizon endpoints remain null."))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    prior.need(out.is_absolute() and out.parent == ROOT / "results"
               and out.name.startswith(NAME + "_") and not out.exists() and not out.is_symlink(), "output")
    inputs = ((prior.open_run.paired.ANCHORS, prior.open_run.paired.ANCHORS_SHA),
              (prior.FB_SAVED, prior.FB_SAVED_SHA), (IM_SAVED, IM_SAVED_SHA),
              (prior.FB_DIR / "run_report.json", prior.FB_REPORT),
              (IM_DIR / "run_report.json", IM_REPORT))
    anchors, real, imagined, real_report, imagined_report = [
        prior.open_run.paired.read(*entry) for entry in inputs
    ]

    def transitions(i, a, b):
        return (prior.traces.transitions(prior.FB_DIR / f"row_{i:02d}", a["report_sha256"]),
                prior.traces.transitions(IM_DIR / f"row_{i:02d}", b["report_sha256"]))

    result = compare(anchors, real, imagined, real_report, imagined_report, transitions)
    result["sources"] = {str(path): digest for path, digest in inputs}
    result["analysis_sha256"] = prior.feedback.sha(Path(__file__).resolve())
    prior.need(all(prior.feedback.sha(path) == digest for path, digest in result["sources"].items()), "source drift")
    digest = prior.feedback.original.physical.design.run._write_new(out, result)
    print(json.dumps({"sha256": digest, "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
