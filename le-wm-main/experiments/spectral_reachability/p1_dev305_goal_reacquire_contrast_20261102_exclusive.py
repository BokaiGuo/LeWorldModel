"""Fixed post-hoc dev305 RGB acquisition contrast; no physical or neural execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_rgb_goal_reacquire_20261102_exclusive as gate

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "results/p1_dev305_diverse_cem_20261102_CAPTURE_DEV01"
BLIND_SHA = "852a8cf856f61b9118ff8335190b182758e9a36ef098d05c1eb1221af5d348cc"
PROTOCOL = ROOT.parent / "docs/D2P_P1_DEV305_RGB_GOAL_REACQUIRE_PROTOCOL_20261102_EXCLUSIVE.md"
PROTOCOL_SHA = "f8865cf325d8c7d0b342d9f94eec1a89ddbb6ebf644566423a80d2d2ecbba61f"


def read_checked(path, digest):
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != digest:
        raise ValueError(f"hash mismatch: {path}")
    return value


def load_frame(ref):
    name = ref["file"]
    if Path(name).name != name:
        raise ValueError("invalid frame path")
    path = DIR / name
    read_checked(path, ref["file_sha256"])
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
    if gate._sha(pixels.tobytes()) != ref["pixel_sha256"]:
        raise ValueError("pixel hash mismatch")
    return pixels


def analyze():
    read_checked(PROTOCOL, PROTOCOL_SHA)
    if not gate._guard_intact():
        raise ValueError("guard source changed")
    blind = json.loads(read_checked(DIR / "blind.json", BLIND_SHA))
    if blind["seed"] != 301000005 or len(blind["images"]) != 3:
        raise ValueError("unexpected fixed root")
    goal, current = load_frame(blind["goal"]), load_frame(blind["images"][2])
    called = []

    def forbidden_selector(_goal, _current):
        called.append(True)
        raise AssertionError("must not run on blocked goal")

    # The archive has no independent reacquisition or reliable acquisition times.
    decision = gate.choose([{"pixels": goal}], {"pixels": current},
                           now_ns=0, max_age_ns=1, selector=forbidden_selector)
    if called or decision["selector_called"] is not False or decision["action"] is not None:
        raise ValueError("unexpected selector call")
    return {"schema": "p1_dev305_goal_reacquire_contrast_exclusive.v1",
            "protocol_sha256": PROTOCOL_SHA, "blind_sha256": BLIND_SHA,
            "goal_file_sha256": blind["goal"]["file_sha256"],
            "current_file_sha256": blind["images"][2]["file_sha256"],
            "single_frame_guard": gate.guard.inspect(goal),
            "current_frame_guard": gate.guard.inspect(current),
            "reacquisition": {"available_real_goal_captures": 1,
                              "status": "not_recorded_no_duplicate_substitution"},
            "decision": decision, "executed_action": None,
            "outcome": "blocked_before_planning_not_a_full_horizon_result",
            "cost": {"unique_rgb_frames_read": 2, "new_physical_steps": 0, "neural_calls": 0,
                     "upstream_cost_included": False},
            "scope": "opened single development root; no calibration or independent confirmation",
            "p1_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze()
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
