"""Regression on prior real journals, never represented as a new feedback run."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_two_stage_journal_feedback_20261030_v1 as bridge
from experiments.spectral_reachability import p1_backtrack_physical_consume_20261030 as reader

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/codex_p1_cp5_twenty_physical_20261030_v1_run_CODEX01"
PREP = ROOT / "results/codex_p1_cp5_twenty_physical_20261030_v1_prepare_CODEX01.json"


def load(index):
    item = json.loads(PREP.read_bytes())["roster"][index]
    folder = RUN / f"row_{index:02d}"
    report = json.loads((folder / "report.json").read_bytes())
    events = reader.events(folder / "journal.jsonl", report["journal_chain_tail"])
    start = next(e["index"] for e in events if e["kind"] == "suffix_start_verified")
    fifth = [e for e in events if e["kind"] == "fresh_observation"
             and e["receipt"]["phase"] == "probe" and e["receipt"]["step"] == 5][-1]["index"]
    prefix = [e for e in events if e["index"] <= fifth]
    packet = np.frombuffer(bytes.fromhex(item["raw10_float32_hex"]), dtype="<f4").copy().reshape(10, 2)[:5]
    return item, folder, start, prefix, packet


@pytest.mark.parametrize("index", range(24))
def test_previous_real_journal_feedback_is_bound(index):
    item, folder, start, events, packet = load(index)
    result = bridge.from_journal(events, folder, index, item["condition"], packet, after_index=start)
    assert len(result["receipts"]) == 5
    assert result["last_render_index"] == events[-1]["index"]


def test_changed_action_stop_state_and_lookahead_rejected():
    item, folder, start, events, packet = load(0)
    transition = next(i for i, e in enumerate(events) if e["kind"] == "returned_transition"
                      and e["phase"] == "probe")
    frame = next(i for i, e in enumerate(events) if e["kind"] == "fresh_observation"
                 and e["receipt"]["phase"] == "probe")
    for mutation in ("action", "stop", "state", "row", "late"):
        bad = copy.deepcopy(events)
        if mutation == "action":
            bad[transition]["actual_bits"] = "00"*8
        elif mutation == "stop":
            bad[transition]["terminated"] = True
        elif mutation == "state":
            bad[frame]["receipt"]["state"][0] += 1
        elif mutation == "row":
            bad[transition]["anchor_index"] = 1
        else:
            bad.append(dict(bad[transition], index=bad[-1]["index"]+1, step=6))
        with pytest.raises(ValueError):
            bridge.from_journal(bad, folder, 0, item["condition"], packet, after_index=start)
