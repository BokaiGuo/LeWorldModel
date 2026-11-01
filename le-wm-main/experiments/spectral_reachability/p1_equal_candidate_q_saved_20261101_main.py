"""Check equal-six allocation plus existing plan/action/ledger identities."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from . import p1_first_feedback_saved_20261101_main as saved


def check(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_bytes())
    for nodes in report["nodes"].values():
        for node in nodes:
            if node["status"] == "not_started":
                continue
            events = [json.loads(x) for x in Path(node["receipt"]["trace"]).read_bytes().splitlines()]
            headers = [e["payload"] for e in events if e["kind"] == "node_header"]
            saved.need(len(headers) == 1 and headers[0]["config"]["steps"] == 6,
                       "six steps per candidate")
            saved.need(not any(e["kind"] == "winner_refinement" for e in events),
                       "no winner refinement")
    control = saved.prior.control
    with patch.object(control.base, "CONFIG", dict(control.base.CONFIG, steps=6)), \
            patch.object(control, "cost", control.base.cost):
        result = saved.check(directory)
    result["allocation"] = "eight_candidates_six_steps_no_winner_refinement"
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
