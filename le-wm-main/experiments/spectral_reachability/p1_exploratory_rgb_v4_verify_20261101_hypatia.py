"""Offline verification of fixed v4 observation/source bindings; no promotion."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p1_exploratory_rgb_history_v4_20261101_hypatia as v4

NAME = "p1_exploratory_rgb_v4_verify_20261101_hypatia"
JOURNAL_SHA = "7dca32ff7a435ff4fd86e20e77cc720a6adcaceae5c2ea1c1c8ea0d426cbffba"
RECEIPT_SHA = "66db4014d7df3a2a2e4dcfdc7072f97cddbdc9f76f91c96bbdb6ee797336e840"
PREP_SHA = "9cbab1d0eef2fcf9c067d332e89f12b77178f17c8f3b0c37cce30aa0e39c6dca"
V3 = v4.ROOT / "results/p1_exploratory_rgb_history_v3_20261101_hypatia_RUN01"
V3_FRAME_SHA = ("a550ac393c7c1153364ef9dab2c7bcf13d1abf96e0fa0e976d1ba30729e0566b",
                "1be47c6873d272b8b2eb2f67c84e56297f89ac84f893341c96091b78dd2a1dc5")


def need(ok, text):
    if not ok:
        raise ValueError(text)


def read_journal(path, expected):
    need(v4.sha(path) == expected, "journal SHA")
    lines = path.read_bytes().splitlines()
    previous, events = None, []
    for i, line in enumerate(lines):
        event = json.loads(line)
        body = {k: v for k, v in event.items() if k != "sha256"}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        need(event["index"] == i and event["previous_sha256"] == previous and event["sha256"] == digest,
             "journal chain")
        previous = digest
        events.append(event)
    return events


def verify_frame(event):
    name = event["file"]
    path = v4.OUTPUT / name
    need(path.parent == v4.OUTPUT and path.is_file() and not path.is_symlink(), "frame path")
    need(v4.sha(path) == event["file_sha256"], "frame SHA")
    frame = np.load(path, allow_pickle=False)
    need(frame.shape == (224, 224, 3) and frame.dtype == np.uint8
         and hashlib.sha256(frame.tobytes()).hexdigest() == event["pixel_sha256"], "pixel binding")
    return frame


def build():
    need(v4.sha(v4.PREPARED) == PREP_SHA and v4.sha(v4.OUTPUT / "receipt.json") == RECEIPT_SHA,
         "prep or receipt SHA")
    prep = json.loads(v4.PREPARED.read_bytes())
    for path, digest in prep["files"].items():
        need(v4.sha(path) == digest, "prepared source changed")
    receipt = json.loads((v4.OUTPUT / "receipt.json").read_bytes())
    need(receipt["prepare_sha256"] == PREP_SHA and receipt["journal_sha256"] == JOURNAL_SHA
         and receipt["planner_observation_authorized"] is False and receipt["freshness"] == "unknown",
         "receipt binding")
    events = read_journal(v4.OUTPUT / "journal.jsonl", JOURNAL_SHA)
    old = read_journal(V3 / "journal.jsonl", v4.PINNED[V3 / "journal.jsonl"])
    need(events[0]["kind"] == "run_started" and events[0]["contract"] == prep
         and events[-1]["kind"] == "run_finished" and events[-1]["slots"] == receipt["slots"], "run boundary")
    results = []
    for slot in (0, 1):
        rows = [e for e in events if e.get("slot") == slot]
        oldrows = [e for e in old if e.get("slot") == slot]
        placement = [e for e in rows if e["kind"] == "reset_completed" and e["phase"] == "placement"]
        earlier = [e for e in oldrows if e["kind"] == "reset_completed" and e["phase"] == "placement"]
        need(len(placement) == len(earlier) == 1 and placement[0]["target_visible_flag"] is False,
             "placement or target")
        need(placement[0]["actual_state"] == earlier[0]["actual_state"], "initial state differs from v3")
        steps = [e for e in rows if e["kind"] == "step_completed"]
        starts = [e for e in rows if e["kind"] == "step_started"]
        oldsteps = [e for e in oldrows if e["kind"] == "step_completed"]
        need([e["step"] for e in steps] == list(range(1, 11))
             and [e["step"] for e in starts] == list(range(1, 11)), "step census")
        actions = np.asarray(prep["actions_float32"], dtype=np.float32)
        need(all(a["raw_float32_bits"] == b["raw_float32_bits"] == actions[i].tobytes().hex()
                 for i, (a, b) in enumerate(zip(starts, steps))), "action bits")
        need(not any(e["terminated"] or e["truncated"] for e in steps), "stop must not be hidden")
        renders = [e for e in rows if e["kind"] == "render_completed"]
        need([e["step"] for e in renders] == [0, 5, 10], "RGB census")
        for e in renders:
            need(e["file"] == f"slot{slot}_rgb{e['step']:02}.npy", "frame name")
            verify_frame(e)
        goals = [e for e in rows if e["kind"] == "goal_frame_saved"]
        need(len(goals) == 1 and goals[0]["file"] == f"slot{slot}_goal_rgb.npy", "goal frame")
        verify_frame(goals[0])
        bound = [e for e in rows if e["kind"] == "initial_state_goal_bound"]
        need(len(bound) == 1 and bound[0]["visual_source"] == "new_target_off_not_original_pixel_identity",
             "source binding")
        need(len(oldsteps) == 10, "v3 state comparison unavailable")
        same_states = all(a["state"] == b["state"] for a, b in zip(steps, oldsteps))
        native_disagreement = sum(e["angle_diagnostics"]["native_eval_success"]
                                  != e["angle_diagnostics"]["wrapped_diagnostic_success"] for e in steps)
        need(v4.sha(V3 / f"slot{slot}_rgb00.npy") == V3_FRAME_SHA[slot], "v3 frame SHA")
        oldrgb = np.load(V3 / f"slot{slot}_rgb00.npy", allow_pickle=False)
        newrgb = np.load(v4.OUTPUT / f"slot{slot}_rgb00.npy", allow_pickle=False)
        results.append({"slot": slot, "steps": len(steps), "source_state_exact": True,
                        "states_match_v3_all_steps": same_states,
                        "pixels_different_from_v3_at_0": int(np.any(oldrgb != newrgb, axis=2).sum()),
                        "native_wrapped_diagnostic_disagreements": native_disagreement,
                        "native_terminated": False, "native_truncated": False})
    return {"schema": NAME + ".v1", "prepare_sha256": PREP_SHA, "run_receipt_sha256": RECEIPT_SHA,
            "journal_sha256": JOURNAL_SHA, "slots": results, "with_target": False,
            "visual_source": "new target-off derivative of previously opened development sources",
            "hidden_physics_state_equivalence": "unknown", "historical_coverage": "unknown",
            "training_disjoint": "unknown", "freshness": "unknown",
            "planner_observation_authorized": False, "P1_admission": False}


def main():
    path = v4.ROOT / f"results/{NAME}_LOCAL01.json"
    data = (json.dumps(build(), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}))


if __name__ == "__main__":
    main()
