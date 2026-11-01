"""Saved-only audit of the fixed MAIN02 out-of-canvas incident."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p1_backtrack_physical_consume_20261030 as prior
from experiments.spectral_reachability import p1_matched_suffix_scale_20261030 as experiment

ROOT = experiment.ROOT
DIRECTORY = ROOT / "results/p1_matched_suffix_scale_20261030_run_MAIN02/anchor_05"
PREPARED = ROOT / "results/p1_matched_suffix_scale_20261030_prepare_MAIN01.json"
PREP_SHA = "544594f50d20747b96c13afd9adf24926b5f9f80726288eb345438eb1635673d"
REPORT_SHA = "73a05677016081d3feee49ce863f7f88dc17b7a531a77e80d0b30d18aba482de"
JOURNAL_SHA = "e90d38cfad7514041683189f2dcf2c86b93c1d49fa312ff8c668ebf537eb5ed9"
require = experiment.require


def audit(report, events, prepared):
    anchor = prepared["anchors"][5]
    probe = prepared["probe_sets"][5][16]
    row = report["rows"][16]
    require(report["anchor"] == 5 and report["status"] == "partial_or_failed" and
            report["errors"] == [{"index": 16, "message": "fresh RGB/canvas guard",
                                  "stage": "branch", "type": "Rejected"}] and
            row["status"] == "engineering_failed" and row["index"] == probe["index"] == 16 and
            row["probe"] == probe["name"] == "selected_-1.00" and
            (anchor["root_slot"], anchor["condition"], anchor["arm"], anchor["role"]) ==
            (0, "on", "projected_ascent", "model_selected"), "incident identity")
    branch = [e for e in events if e.get("branch") == 16]
    start = [e for e in branch if e["kind"] == "suffix_start_verified"]
    steps = [e for e in branch if e["kind"] == "returned_transition" and e["phase"] == "probe"]
    require(len(start) == 1 and len(steps) == 5 and [e["step"] for e in steps] == list(range(1, 6)),
            "start and probe ordering")
    require(start[0]["index"] < steps[0]["index"] and
            all(left["index"] < right["index"] for left, right in zip(steps, steps[1:])),
            "verified start precedes ordered transitions")
    raw = bytes.fromhex(probe["raw5_float32_hex"])
    require(hashlib.sha256(raw).hexdigest() == probe["raw5_sha256"] and len(raw) == 40,
            "prepared action hash")
    for index, step in enumerate(steps):
        require(step["actual_dtype"] == "<f4" and step["actual_bits"] == step["requested_bits"] ==
                raw[8 * index:8 * (index + 1)].hex() and
                (step["anchor_index"], step["condition"], step["candidate_role"], step["probe"]) ==
                (5, "on", "model_selected", probe["name"]), "action or branch identity")
    last = steps[-1]
    state = last["state"]
    require(len(state) == 7 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in state)
            and state == last["snapshot"]["state"] and
            state[:2] == last["snapshot"]["bodies"]["agent"]["position"] and
            state[2:4] == last["snapshot"]["bodies"]["block"]["position"] and
            any(v < 0 or v > 512 for v in state[:4]), "returned out-of-canvas state")
    images = [e for e in branch if e["kind"] == "fresh_observation" and
              e["receipt"]["phase"] == "probe" and e["receipt"]["step"] == 5]
    require(len(images) == 1 and images[0]["receipt"]["condition"] == "off" and
            images[0]["receipt"]["state"] == state and last["index"] < images[0]["index"],
            "partial incident observation")
    closed = [e for e in branch if e["kind"] == "branch_closed"]
    require(len(closed) == 1 and closed[0]["status"] == "engineering_failed" and
            images[0]["index"] < closed[0]["index"] < events[-1]["index"] and
            events[-1]["kind"] == "run_finished" and events[-1]["errors"] == report["errors"],
            "terminal error ordering")
    return {"state": state, "action_step5_hex": last["actual_bits"],
            "native_terminated": last["terminated"], "native_truncated": last["truncated"],
            "off_observation": images[0]["receipt"], "transition_event_index": last["index"],
            "observation_event_index": images[0]["index"], "branch_close_event_index": closed[0]["index"]}


def main():
    sha = experiment.runner.physical.sha
    read = experiment.runner.physical.native.read
    require(sha(PREPARED) == PREP_SHA and sha(DIRECTORY / "report.json") == REPORT_SHA and
            sha(DIRECTORY / "journal.jsonl") == JOURNAL_SHA, "external source hashes")
    prepared = read(PREPARED, cap=32 * 1024**2)
    require(prepared == experiment.build(), "prepared source closure")
    report = read(DIRECTORY / "report.json", cap=64 * 1024**2)
    require(report["journal_sha256"] == JOURNAL_SHA, "report journal binding")
    events = prior.events(DIRECTORY / "journal.jsonl", report["journal_chain_tail"])
    incident = audit(report, events, prepared)
    frame = incident["off_observation"]
    image_path = experiment.runner.physical.native.paired.flat(DIRECTORY, frame["png"])
    require(sha(image_path) == frame["file_sha256"], "incident PNG hash")
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGB"))
    require(pixels.shape == (224, 224, 3) and
            experiment.runner.physical.native.moving.image_sha(pixels) == frame["pixel_sha256"],
            "incident pixels")
    canonical_agent_pixels = int(np.count_nonzero(np.all(pixels == [65, 105, 225], axis=-1)))
    require(canonical_agent_pixels == 0, "incident agent visibility")
    print(json.dumps({"schema": "p1_canvas_incident_audit_20261030_main",
                      "status": "out_of_canvas_verified_not_endpoint_or_P1_admission",
                      "prepared_sha256": PREP_SHA, "report_sha256": REPORT_SHA,
                      "journal_sha256": JOURNAL_SHA, "canonical_agent_pixels": canonical_agent_pixels,
                      "incident": incident}, sort_keys=True))


if __name__ == "__main__":
    main()
