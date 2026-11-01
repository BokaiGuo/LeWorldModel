import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_rgb_cp10_decision_bridge_20261030_v1 as bridge
from experiments.spectral_reachability import codex_p1_two_stage_journal_feedback_20261030_v1 as source
from experiments.spectral_reachability import p1_backtrack_physical_consume_20261030 as reader

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/codex_p1_cp5_twenty_physical_20261030_v1_run_CODEX01"
PREP = ROOT / "results/codex_p1_cp5_twenty_physical_20261030_v1_prepare_CODEX01.json"


def fixture(index=1):
    item = json.loads(PREP.read_bytes())["roster"][index]
    folder = RUN / f"row_{index:02d}"
    report = json.loads((folder / "report.json").read_bytes())
    events = reader.events(folder / "journal.jsonl", report["journal_chain_tail"])
    start = next(e["index"] for e in events if e["kind"] == "suffix_start_verified")
    last = [e for e in events if e["kind"] == "fresh_observation"
            and e["receipt"]["phase"] == "probe" and e["receipt"]["step"] == 5][-1]["index"]
    events = [e for e in events if e["index"] <= last]
    packet = np.frombuffer(bytes.fromhex(item["raw10_float32_hex"]), dtype="<f4").copy().reshape(10, 2)[:5]
    bound = source.from_journal(events, folder, index, item["condition"], packet, after_index=start)
    goal = bridge.feedback.rgb_pixels(bound["png"])
    return events, folder, item, start, packet, goal


def make(goal_override=None, index=1):
    events, folder, item, start, packet, actual_goal = fixture(index)
    goal = actual_goal if goal_override is None else goal_override
    digest = hashlib.sha256(goal.tobytes()).hexdigest()
    model = np.full((5, 2), .1, dtype="<f4")
    args = (events, folder, index, item["condition"], packet, start, goal, digest, model)
    return bridge.decide(*args), args


def test_real_cp10_and_recomputed_receipt_tamper():
    receipt, args = make()
    assert receipt["decision"] == "model_packet"
    assert bridge.verify(receipt, *args) == receipt
    for key in ("cp10_pixel_sha256", "goal_pixel_sha256", "executed_packet_f32_hex", "decision"):
        bad = copy.deepcopy(receipt)
        bad[key] = "bad"
        with pytest.raises(ValueError):
            bridge.verify(bad, *args)
    badargs = list(args)
    badargs[7] = "0" * 64
    with pytest.raises(ValueError):
        bridge.decide(*badargs)


def test_rejected_rgb_has_no_dispatch():
    receipt, args = make(np.full((224, 224, 3), 255, dtype=np.uint8))
    assert receipt["decision"] == "rgb_rejected"
    assert bridge.verify(receipt, *args) == receipt
    assert bridge.verify_dispatch(receipt, []) == "rgb_rejected"
    with pytest.raises(ValueError):
        bridge.verify_dispatch(receipt, [{}])
    real_rejection, real_args = make(index=0)
    assert real_rejection["decision"] == "rgb_rejected"
    assert bridge.verify(real_rejection, *real_args) == real_rejection


def test_dispatch_requires_exact_action_and_native_stop():
    receipt, _ = make()
    raw = bytes.fromhex(receipt["executed_packet_f32_hex"])
    events = [dict(kind="returned_transition", phase="probe", step=6+i, branch=0,
                   index=receipt["last_render_index"]+i+1,
                   anchor_index=receipt["row"], condition=receipt["condition"], actual_dtype="<f4",
                   requested_bits=raw[8*i:8*i+8].hex(), actual_bits=raw[8*i:8*i+8].hex(),
                   terminated=False, truncated=False) for i in range(5)]
    assert bridge.verify_dispatch(receipt, events) == "probe_completed"
    stopped = copy.deepcopy(events[:2])
    stopped[-1]["terminated"] = True
    assert bridge.verify_dispatch(receipt, stopped) == "native_stopped"
    stopped.append(events[2])
    with pytest.raises(ValueError):
        bridge.verify_dispatch(receipt, stopped)
    with pytest.raises(ValueError):
        bridge.verify_dispatch(receipt, events[:2])
    events[0]["actual_bits"] = "00"*8
    with pytest.raises(ValueError):
        bridge.verify_dispatch(receipt, events)
    events[0]["actual_bits"] = events[0]["requested_bits"]
    events[1]["index"] = events[0]["index"]
    with pytest.raises(ValueError):
        bridge.verify_dispatch(receipt, events)
