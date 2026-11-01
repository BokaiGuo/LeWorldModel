"""Recover observed H10 states without imputing trajectories stopped before H10."""
import argparse
import hashlib
import json
from pathlib import Path

from . import p1_uphill_multistart_pairing_20261030_main as paired
from . import p1_uphill_multistart_rgb_20261030_main as rgb

NAME = "p1_uphill_horizon_observation_20261031"
need = paired.need


def extract(rows, shards, anchors, reports):
    need(len(rows) == len(shards) == len(anchors) == len(reports) == 24, "roster")
    result = []
    for i, (row, shard, anchor, report) in enumerate(zip(rows, shards, anchors, reports, strict=True)):
        physical = report["row"]
        need(row["index"] == shard["index"] == anchor["index"] == i, "index")
        need(all(row[k] == anchor[k] for k in ("role", "logical_pair", "root_slot", "condition")),
             "anchor identity")
        need(physical["status"] == row["status"] and physical["condition"] == row["condition"],
             "status binding")
        responses = physical["responses"]
        need(len(responses) == row["returned_suffix"] and
             [r["step"] for r in responses] == list(range(1, len(responses) + 1)),
             "returned steps")
        need(all(not r["terminated"] and not r["truncated"] for r in responses[:-1]),
             "earlier stop")
        if row["status"] == "native_stopped":
            need(bool(responses) and (responses[-1]["terminated"] or responses[-1]["truncated"]),
                 "native stop")
        else:
            need(row["status"] == "probe_completed" and len(responses) == 10
                 and not responses[-1]["terminated"] and not responses[-1]["truncated"],
                 "completed")
        observed = len(responses) == 10
        metrics = None
        if observed:
            state = responses[-1]["state"]
            frames = [f for f in physical["trajectory"] if f["phase"] == "probe"
                      and f["step"] == 10
                      and f["overlay_index"] == (1 if row["condition"] == "on" else 0)]
            need(len(frames) == 1 and frames[0]["state"] == state, "H10 frame identity")
            metrics = paired.metric.physical_metrics(state, anchor["goal"])
            need(metrics["native_success"] == row["last_native_success"], "success metric binding")
            if row["status"] == "probe_completed":
                need(paired.metric.paired_difference(metrics, row["endpoint"]) ==
                     {k: 0.0 for k in paired.metric.paired_difference(metrics, metrics)},
                     "existing endpoint")
        result.append({"index": i, "logical_pair": row["logical_pair"], "role": row["role"],
                       "status": row["status"], "returned_suffix": len(responses),
                       "h10_observed": observed, "h10_metrics": metrics,
                       "native_success": row["last_native_success"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == paired.ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists()
         and not output.is_symlink(), "exclusive output")
    pairs = paired.read(rgb.PAIR, rgb.PAIR_SHA)
    top = paired.read(rgb.RUN / "run_report.json", rgb.REPORT_SHA)
    anchors = paired.read(paired.ANCHORS, paired.ANCHORS_SHA)
    sources = {str(rgb.PAIR): rgb.PAIR_SHA, str(rgb.RUN / "run_report.json"): rgb.REPORT_SHA,
               str(paired.ANCHORS): paired.ANCHORS_SHA}
    reports = []
    for shard in top["shards"]:
        path = rgb.RUN / f"row_{shard['index']:02d}" / "report.json"
        reports.append(paired.read(path, shard["report_sha256"]))
        sources[str(path)] = shard["report_sha256"]
    rows = extract(pairs["rows"], top["shards"], anchors["anchors"], reports)
    need(all(paired.physical.sha(path) == digest for path, digest in sources.items()), "source drift")
    value = {"schema": NAME, "rows": rows, "source_hashes": sources,
             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "limitations": "Post-hoc observation audit; known dependent roots; not an independent "
                            "confirmation or alteration to sealed status-based contracts.",
             "P1_admission": False, "independent_confirmation": False}
    digest = paired.physical.physical.design.run._write_new(output, value)
    print(json.dumps({"rows": len(rows), "observed_h10": sum(r["h10_observed"] for r in rows),
                      "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
