"""Saved-incident and in-memory tampering only; no simulator or GPU."""
import copy

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import p1_matched_canvas_incident_audit_20261030_codex as audit


@pytest.fixture(scope="module")
def data():
    p = audit.old.runner.physical.native.read(audit.PREPARE, cap=32 * 1024**2)
    report = audit.old.runner.physical.native.read(audit.DIRECTORY / "report.json", cap=64 * 1024**2)
    events = audit.saved.events(audit.DIRECTORY / "journal.jsonl", report["journal_chain_tail"])
    row = report["rows"][16]
    with Image.open(audit.DIRECTORY / row["last_observation"]["png"]) as image:
        pixels = np.asarray(image)
    return events, p["anchors"][5], p["probe_sets"][5][16], row, pixels


def test_real_incident_is_not_endpoint_or_authorization():
    value = audit.build()
    assert value["out_of_bounds"] == [dict(coordinate=1, value=528.5548847416848)]
    assert value["action_float32_hex"] == "10c5703e669c4a3f"
    assert value["blue_pixels"] == 0
    assert value["historical_failure_preserved"] and not value["endpoint_accepted"]
    assert not value["continuation_authorized"] and not value["P1_admission"]


@pytest.mark.parametrize("change", ["action", "state", "time", "branch", "snapshot", "pixels", "no_oob"])
def test_in_memory_forgery_rejected(data, change):
    events, anchor, probe, row, pixels = data
    # Restrict copying to the event subsequence used by the incident verifier.
    events = copy.deepcopy([e for e in events if e.get("branch") == 16
                            or e.get("receipt", {}).get("branch") == 16])
    row, probe = copy.deepcopy(row), copy.deepcopy(probe)
    trans = next(e for e in events if e["kind"] == "returned_transition"
                 and e.get("phase") == "probe" and e.get("step") == 5)
    frame = next(e for e in events if e["kind"] == "fresh_observation"
                 and e["receipt"]["phase"] == "probe" and e["receipt"]["step"] == 5)
    if change == "action":
        trans["actual_bits"] = "00" * 8
    elif change == "state":
        trans["state"][1] = 600
    elif change == "time":
        trans["index"] = frame["index"] + 1
    elif change == "branch":
        trans["branch"] = 15
    elif change == "snapshot":
        frame["post"]["snapshot"]["state"][1] = 500
    elif change == "pixels":
        pixels = pixels.copy()
        pixels[0, 0, 0] ^= 1
    else:
        # A self-consistent in-bounds observation is still not censorable.
        state = trans["state"]
        state[1] = 500
        for obj in (trans["snapshot"], frame["receipt"]["snapshot"],
                    frame["pre"]["snapshot"], frame["post"]["snapshot"]):
            obj["state"] = state
        frame["receipt"]["state"] = state
        row["last_observation"] = frame["receipt"]
        import struct
        for obj in (trans["snapshot"], frame["receipt"]["snapshot"],
                    frame["pre"]["snapshot"], frame["post"]["snapshot"]):
            obj["state_float64_bits"] = struct.pack("<7d", *state).hex()
    with pytest.raises(ValueError):
        audit.incident(events, anchor, probe, row, pixels)
