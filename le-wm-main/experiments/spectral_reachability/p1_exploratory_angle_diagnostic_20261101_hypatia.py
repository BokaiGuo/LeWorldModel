"""Offline native-versus-wrapped angle diagnostics; never modifies a gate."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p1_exploratory_rgb_history_v2_20261101_hypatia as capture

NAME = "p1_exploratory_angle_diagnostic_20261101_hypatia"
DIR = capture.OUTPUT
JOURNAL_SHA = "387c51c59f710c9df51aac43fc417c545bd9cdf970aa47bdad3949dc2db0121c"
RECEIPT_SHA = "d4a14f4f000ca345d9cdd233b14e1ac3ba6e0d4e9f366c43334b90f04ce9b832"


def metrics(goal, state):
    diff = abs(float(goal[4]) - float(state[4]))
    native = min(diff, 2 * math.pi - diff)
    wrapped = abs(math.atan2(math.sin(float(state[4]) - float(goal[4])),
                             math.cos(float(state[4]) - float(goal[4]))))
    position = math.sqrt(sum((float(goal[i]) - float(state[i])) ** 2 for i in range(4)))
    return {"absolute_angle_difference": diff, "native_angle_expression": native,
            "strict_wrapped_absolute_angle": wrapped, "position_l2_first4": position,
            "native_formula_success": position < 20 and native < math.pi / 9,
            "wrapped_diagnostic_success": position < 20 and wrapped < math.pi / 9}


def build():
    if capture.sha(DIR / "journal.jsonl") != JOURNAL_SHA or capture.sha(DIR / "receipt.json") != RECEIPT_SHA:
        raise ValueError("saved receipt or journal SHA drift")
    events = [json.loads(line) for line in (DIR / "journal.jsonl").read_bytes().splitlines()]
    previous = None
    for index, event in enumerate(events):
        body = {k: v for k, v in event.items() if k != "sha256"}
        raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        digest = hashlib.sha256(raw).hexdigest()
        if event["index"] != index or event["previous_sha256"] != previous or event["sha256"] != digest:
            raise ValueError("journal chain mismatch")
        previous = digest
    receipt = json.loads((DIR / "receipt.json").read_bytes())
    if receipt["journal_sha256"] != JOURNAL_SHA:
        raise ValueError("receipt journal binding")
    rows = []
    for slot in (0, 1):
        path = capture.RUN / f"source_{slot}.npz"
        if capture.sha(path) != capture.PINNED[path]:
            raise ValueError("source SHA drift")
        with np.load(path, allow_pickle=False) as archive:
            goal = archive["goal"].copy()
        placements = [e for e in events if e["kind"] == "reset_completed" and e["phase"] == "placement"
                      and e["slot"] == slot]
        if len(placements) != 1:
            raise ValueError("missing placement")
        for phase, step, state in [("placement", 0, placements[0]["actual_state"])] + [
                ("step", e["step"], e["state"]) for e in events
                if e["kind"] == "step_completed" and e["slot"] == slot]:
            rows.append({"slot": slot, "phase": phase, "step": step, **metrics(goal, state)})
    return {"schema": NAME + ".v1", "journal_sha256": JOURNAL_SHA, "receipt_sha256": RECEIPT_SHA,
            "values": rows, "observed_slot_statuses": receipt["slots"],
            "interpretation": "offline formula evaluation, not a recorded native termination or changed gate",
            "strict_angle_is_diagnostic_only": True, "censored_steps_retained": True,
            "history_complete": False, "freshness": "unknown", "training_disjoint": "unknown",
            "P1_admission": False}


def main():
    path = capture.ROOT / f"results/{NAME}_LOCAL01.json"
    data = (json.dumps(build(), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}))


if __name__ == "__main__":
    main()
