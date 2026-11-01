"""Verify normalized update logs in addition to source/selection/cost checks."""
import argparse
import json
import math
from pathlib import Path

from . import p1_suffix_exchange_saved_20261101_main as prior


def check(directory):
    directory = Path(directory)
    result = prior.check(directory)
    report = json.loads((directory / "report.json").read_bytes())
    summaries = []
    for row in report["rows"]:
        for node in report["nodes"][row["episode_id"]]:
            if node["status"] == "not_started":
                continue
            events = [json.loads(line) for line in Path(node["receipt"]["trace"]).read_bytes().splitlines()]
            headers = [e["payload"] for e in events if e["kind"] == "node_header"]
            records = [e["payload"] for e in events if e["kind"] == "normalized_gradient_updates"]
            prior.prior.saved.need(len(headers) == len(records) == 1
                                   and headers[0]["config"]["gradient_rule"] == "linf_normalized_action_peak",
                                   "normalized optimizer contract/log")
            candidates = records[0]["candidates"]
            prior.prior.saved.need(len(candidates) == 8 and records[0]["physical_labels_used"] is False,
                                   "eight candidate updates")
            peaks = []
            for updates in candidates:
                prior.prior.saved.need(len(updates) == 6, "six updates")
                for i, update in enumerate(updates):
                    prior.prior.saved.need(update["iteration"] == i
                                           and all(math.isfinite(update[k]) for k in
                                                   ("loss", "gradient_linf", "action_step_linf"))
                                           and update["gradient_linf"] >= 0
                                           and 0 <= update["action_step_linf"] <= .0500001,
                                           "finite bounded normalized updates")
                    peaks.append(update["action_step_linf"])
            summaries.append(dict(boundary=node["boundary"], updates=len(peaks),
                                  min_action_step=min(peaks), max_action_step=max(peaks)))
    result["normalized_update_summary"] = summaries
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(nodes=len(result["nodes"]), output=str(args.output))))
