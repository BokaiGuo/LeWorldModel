"""Saved-artifact integrity and fixed-roster audit; does not rerun inference."""
import json
import sys
from pathlib import Path

import numpy as np

from . import codex_p1_bounded_joint_20261101_exclusive as run


def verify(prep, directory):
    host = run.blind.host
    report = json.loads((directory / "report.json").read_bytes())
    host.require(host.sha(prep) == report["protocol_sha256"]
                 and json.loads(prep.read_bytes()) == run.protocol(), "protocol/source")
    events_path = directory / "events.jsonl"
    host.require(host.sha(events_path) == report["events_sha256"], "events hash")
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    index = 0
    for slot, row in enumerate(report["rows"]):
        host.require(row["slot"] == slot and len(row["evaluations"]) == 72, "roster")
        for candidate in range(8):
            for iteration in range(9):
                for packet in range(3):
                    for kind in ("action_encoder", "predict"):
                        for event in ("attempt", "return"):
                            expected = {"event": event, "kind": kind, "slot": slot, "candidate": candidate,
                                        "iteration": iteration, "packet": packet}
                            host.require(events[index] == expected, "forward sequence")
                            index += 1
                expected_row = row["evaluations"][candidate * 9 + iteration]
                host.require(events[index] == {"event": "score", "slot": slot, **expected_row}, "score event")
                action = np.asarray(expected_row["actions"])
                host.require(action.shape == (3, 5, 2) and np.isfinite(action).all()
                             and (np.abs(action) <= 1).all(), "actions")
                index += 1
                if iteration < 8:
                    for event in ("attempt", "return"):
                        host.require(events[index] == {"event": event, "kind": "reverse", "slot": slot,
                                                     "candidate": candidate, "iteration": iteration}, "reverse sequence")
                        index += 1
        host.require(row["selected"] == run.choose(row["root_mse"], row["evaluations"]), "selection")
    host.require(len(report["rows"]) == 2 and index == len(events), "extra rows/events")
    host.require(report["attempts"] == report["returns"] ==
                 {"action_encoder": 432, "predict": 432, "reverse": 128}, "counts")
    host.require(report["module_hooks"] == {"action_encoder": 432, "predict": 432}
                 and report["model_before"] == report["model_after"]
                 and not report["physical_execution"], "integrity")
    return {"status": "verified_saved_artifacts_without_inference", "event_count": len(events),
            "report_sha256": host.sha(directory / "report.json")}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]), Path(sys.argv[2])), sort_keys=True))
