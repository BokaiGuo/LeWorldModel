"""Validate recorded horizon contracts and exact per-boundary neural costs."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from . import p1_horizon_q_20261101_main as host
from . import p1_first_feedback_saved_20261101_main as saved


def check(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_bytes())
    horizons = []
    for row in report["rows"]:
        for node in report["nodes"][row["episode_id"]]:
            if node["status"] == "not_started":
                continue
            packets, _ = host.parent.carry.dimensions(node["boundary"])
            events = [json.loads(x) for x in Path(node["receipt"]["trace"]).read_bytes().splitlines()]
            headers = [e["payload"] for e in events if e["kind"] == "node_header"]
            saved.need(len(headers) == 1 and headers[0]["config"]["steps"] == 6
                       and headers[0]["config"]["planning_horizon_packets"] == packets,
                       "boundary-specific optimizer horizon")
            saved.need(not any(e["kind"] == "winner_refinement" for e in events),
                       "no winner-only refinement")
            horizons.append((row["arm"], packets))
    pending = iter(horizons)

    def expected(arm):
        which, packets = next(pending)
        saved.need(which == arm, "ledger traversal order")
        with patch.dict(host.base.CONFIG, steps=6, planning_horizon_packets=packets):
            return host.cost(arm)

    with patch.object(saved.prior.control, "cost", expected):
        result = saved.check(directory)
    saved.need(next(pending, None) is None, "all node budgets consumed")
    result["horizon_contracts"] = horizons
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
