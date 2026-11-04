"""RGB-only wider-view acquisition request; never authorizes robot actions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_rgb_goal_reacquire_20261102_exclusive as guard

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "results/p1_dev305_diverse_cem_20261102_CAPTURE_DEV01"
BLIND_SHA = "852a8cf856f61b9118ff8335190b182758e9a36ef098d05c1eb1221af5d348cc"
PROTOCOL = ROOT.parent / "docs/D2P_PUSHT_WIDEVIEW_REQUEST_PROTOCOL_20261102_EXCLUSIVE.md"
PROTOCOL_SHA = "ea3d6bb68e87ae46f365af59653f0b1b1ae5f0594f213b8093c162df68e0619f"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_frame(ref):
    if Path(ref["file"]).name != ref["file"]:
        raise ValueError("invalid image path")
    path = CAPTURE / ref["file"]
    if digest(path.read_bytes()) != ref["file_sha256"]:
        raise ValueError("file hash mismatch")
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    if digest(pixels.tobytes()) != ref["pixel_sha256"]:
        raise ValueError("pixel hash mismatch")
    return pixels


def inspect_alternate(original_goal, original_current, acquire):
    """Request a single view pair; output is diagnostic only, never an action."""
    report = {"status": "blocked", "action": None, "selector_called": False,
              "original_goal_guard": None, "original_current_guard": None,
              "adapter_calls": 0, "alternate_guard": None}
    try:
        if not guard._guard_intact():
            report["reason"] = "guard_source_mismatch"
            return report
        report["original_goal_guard"] = guard.guard.inspect(original_goal)
        report["original_current_guard"] = guard.guard.inspect(original_current)
        if report["original_goal_guard"]["eligible"]:
            report["reason"] = "original_goal_already_eligible_no_request"
            return report
        report["adapter_calls"] = 1
        pair = acquire({"purpose": "goal_and_current_same_wider_rgb_view",
                        "max_attempts": 1, "move_robot": False, "dispatch_action": False})
        if pair is None:
            report["reason"] = "wider_view_unavailable"
            return report
        goal, current = pair["goal"], pair["current"]
        for key in ("camera_id", "session_id", "target_id", "view_id"):
            a, b = goal[key], current[key]
            if not isinstance(a, str) or not a or a != b:
                report["reason"] = "view_or_task_context_mismatch"
                return report
        if (not isinstance(goal["capture_id"], str) or not goal["capture_id"] or
                not isinstance(current["capture_id"], str) or not current["capture_id"] or
                goal["capture_id"] == current["capture_id"]):
            report["reason"] = "capture_identity_invalid"
            return report
        gs, cs = guard.guard.inspect(goal["pixels"]), guard.guard.inspect(current["pixels"])
        report["alternate_guard"] = {"goal": gs, "current": cs}
        if not gs["eligible"] or not cs["eligible"]:
            report["reason"] = "alternate_guard_failed"
            return report
        report["reason"] = "calibration_and_identity_verification_required"
        return report
    except Exception as exc:
        report["reason"] = "input_or_acquisition_error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        return report


def dev305():
    if digest(PROTOCOL.read_bytes()) != PROTOCOL_SHA:
        raise ValueError("protocol hash mismatch")
    raw = (CAPTURE / "blind.json").read_bytes()
    if digest(raw) != BLIND_SHA:
        raise ValueError("blind hash mismatch")
    data = json.loads(raw)
    if data["seed"] != 301000005 or len(data["images"]) != 3:
        raise ValueError("unexpected development source")
    goal, current = read_frame(data["goal"]), read_frame(data["images"][2])
    requests = []

    def unavailable(request):
        requests.append(request)
        return None

    decision = inspect_alternate(goal, current, unavailable)
    return {"schema": "pusht_wideview_request_exclusive.v1",
            "protocol_sha256": digest(PROTOCOL.read_bytes()), "blind_sha256": BLIND_SHA,
            "goal_pixel_sha256": data["goal"]["pixel_sha256"],
            "current_pixel_sha256": data["images"][2]["pixel_sha256"],
            "requests": requests, "decision": decision, "physical_steps": 0,
            "reset_calls": 0, "neural_calls": 0, "p1_admission": False,
            "scope": "known development root; no alternate camera or physical acquisition"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dev305()
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
