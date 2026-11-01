"""Verify diagnostic artifacts and update arithmetic without model inference."""
import json
import sys
from pathlib import Path

import numpy as np

from . import codex_p1_bounded_joint_diag32_20261101_exclusive as diag


def verify(prep, directory):
    host = diag.blind.host
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    host.require(host.sha(prep) == report["protocol_sha256"] and json.loads(prep.read_bytes()) == diag.protocol(),
                 "protocol/source mismatch")
    events_path = directory / "events.jsonl"
    host.require(host.sha(events_path) == report["events_sha256"], "event hash")
    events = [json.loads(x) for x in events_path.read_text().splitlines()]
    pos = 0
    for slot, root in enumerate(report["rows"]):
        host.require(root["slot"] == slot and len(root["evaluations"]) == 264, "roster")
        for candidate in range(8):
            for iteration in range(33):
                loc = {"slot": slot, "candidate": candidate, "iteration": iteration}
                for packet in range(3):
                    for kind in ("action_encoder", "predict"):
                        for event in ("attempt", "return"):
                            host.require(events[pos] == {"event": event, "kind": kind, **loc, "packet": packet},
                                         "forward sequence")
                            pos += 1
                if iteration < 32:
                    for event in ("attempt", "return"):
                        host.require(events[pos] == {"event": event, "kind": "reverse", **loc}, "reverse sequence")
                        pos += 1
                row = root["evaluations"][candidate * 33 + iteration]
                host.require(events[pos] == {"event": "score", "slot": slot, **row}, "score mismatch")
                pos += 1
                action = np.asarray(row["actions"])
                host.require(action.shape == (3, 5, 2) and np.isfinite(action).all() and (np.abs(action) <= 1).all(),
                             "action bounds")
                expected_loss = row["q3"] + 10 * max(root["root_mse"] - row["q1"], 0) ** 2
                host.require(abs(row["loss"] - expected_loss) < 2e-6, "loss")
                host.require(row["saturated_coordinates"] == int((np.abs(action) >= 1).sum()), "saturation")
                if iteration < 32:
                    following = root["evaluations"][candidate * 33 + iteration + 1]
                    distance = np.linalg.norm(np.asarray(following["actions"]) - action)
                    host.require(abs(distance - row["gradient"]["projected_step_l2"]) < 2e-6, "step distance")
                else:
                    host.require(row["gradient"] is None, "final gradient")
        host.require(root["selected"] == diag.choose(root["root_mse"], root["evaluations"]), "selection")
    host.require(len(report["rows"]) == 2 and pos == len(events), "extra events")
    host.require(report["attempts"] == report["returns"] == {"action_encoder": 1584, "predict": 1584, "reverse": 512},
                 "attempts")
    host.require(report["module_hooks"] == {"action_encoder": 1584, "predict": 1584}
                 and report["model_before"] == report["model_after"] and not report["physical_execution"], "integrity")
    return {"status": "saved_artifacts_verified_without_inference", "events": len(events),
            "report_sha256": host.sha(report_path)}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]), Path(sys.argv[2])), sort_keys=True))
