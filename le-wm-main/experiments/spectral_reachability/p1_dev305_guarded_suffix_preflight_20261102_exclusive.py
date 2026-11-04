"""No-action preflight for a fixed 17-by-3 PushT common-suffix panel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_rgb_goal_reacquire_20261102_exclusive as guard
from . import p1_pusht_wideview_request_20261102_exclusive as rgb

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "results/p1_dev305_two_packet_physical_oracle_20261102_PREP_DEV01.json"
SPEC_SHA = "f826ea69ac9a4afb4d207a8980b281eea7262a3e8c20285b00788670c909c921"
PROTOCOL = ROOT.parent / "docs/D2P_PUSHT_FIXEDBANK_GUARDED_SUFFIX_PROTOCOL_20261102_EXCLUSIVE.md"
PROTOCOL_SHA = "a4e7c901db35afa3922b20fb55853501068b01263a3c5f26b6e998d1a271b2ad"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_checked(path, sha):
    value = path.read_bytes()
    if digest(value) != sha:
        raise ValueError(f"hash mismatch: {path.name}")
    return value


def preflight(goal, prefixes, bank, *, environment_factory):
    """No environment is opened, even when the goal passes; separate protocol required."""
    if not guard._guard_intact():
        raise ValueError("guard source mismatch")
    screen = guard.guard.inspect(goal)
    if prefixes != ["candidate_02", "candidate_00", "zero"] or len(bank) != 17:
        raise ValueError("invalid fixed roster")
    actions = np.asarray([item["action"] for item in bank], dtype="<f4")
    if actions.shape != (17, 2) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
        raise ValueError("invalid actions")
    if bank[16]["name"] != "zero" or not np.array_equal(actions[16], [0, 0]):
        raise ValueError("common suffix not zero")
    status = "further_physical_preflight_required" if screen["eligible"] else "blocked_before_environment"
    rows = []
    for prefix in prefixes:
        for first in range(17):
            packet = np.stack([actions[first]] * 5 + [actions[16]] * 5).astype("<f4")
            rows.append({"prefix": prefix, "first_index": first, "second_index": 16,
                         "plan_sha256": digest(packet.tobytes()), "planned_steps": 10,
                         "executed_steps": 0, "status": status, "terminal_reward": None})
    return {"status": status, "goal_guard": screen, "rows": rows,
            "environment_factory_calls": 0, "environment_resets": 0,
            "physical_steps": 0, "neural_calls": 0, "p1_admission": False}


def analyze():
    read_checked(PROTOCOL, PROTOCOL_SHA)
    spec = json.loads(read_checked(SPEC, SPEC_SHA))
    blind = json.loads(read_checked(rgb.CAPTURE / "blind.json", rgb.BLIND_SHA))
    if spec["seed"] != blind["seed"] or spec["packet_length"] != 5:
        raise ValueError("source binding mismatch")
    goal = rgb.read_frame(blind["goal"])
    def forbidden():
        raise AssertionError("preflight must not create environment")
    result = preflight(goal, [x["name"] for x in spec["prefixes"]], spec["bank"],
                       environment_factory=forbidden)
    result.update(schema="p1_dev305_guarded_suffix_preflight_exclusive.v1",
                  protocol_sha256=PROTOCOL_SHA, bank_contract_sha256=SPEC_SHA,
                  goal_pixel_sha256=blind["goal"]["pixel_sha256"],
                  scope="opened development root; no post-preflight physical rollout or independent confirmation")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
