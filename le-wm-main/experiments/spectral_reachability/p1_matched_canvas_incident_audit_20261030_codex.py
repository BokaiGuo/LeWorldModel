"""Read-only forensic check of the MAIN02 canvas incident, not a new runner.

This checks saved observations and action binding. It cannot prove that an
unrecorded exception did not occur or authorize continuation/physical work.
"""
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p1_backtrack_physical_consume_20261030 as saved
from experiments.spectral_reachability import p1_matched_suffix_scale_20261030 as old

ROOT = old.ROOT
PREPARE = ROOT / "results/p1_matched_suffix_scale_20261030_prepare_MAIN01.json"
PREPARE_SHA = "544594f50d20747b96c13afd9adf24926b5f9f80726288eb345438eb1635673d"
DIRECTORY = ROOT / "results/p1_matched_suffix_scale_20261030_run_MAIN02/anchor_05"
REPORT_SHA = "73a05677016081d3feee49ce863f7f88dc17b7a531a77e80d0b30d18aba482de"
BINDING_SHA = "5d598f854218dd5f2ca6914dbcb786432803c8a1216055a5720160a7261dc873"
need = old.require


def incident(events, anchor, probe, row, image):
    """Validate one saved guard incident without changing its failure status."""
    branch, step = probe["index"], row["last_observation"]["step"]
    need(type(step) is int and 1 <= step <= 5 and branch == row["index"]
         and row["anchor_index"] == anchor["index"] and row["task_id"] == anchor["task_id"]
         and row["probe"] == probe["name"] and row["status"] == "engineering_failed"
         and row["start_verified"] is True and row["start_state"] == anchor["checkpoint"]["state"]
         and row["error"] == dict(stage="branch", type="Rejected", message="fresh RGB/canvas guard")
         and not row["secondary_errors"], "incident row")
    action_bytes = bytes.fromhex(probe["raw5_float32_hex"])
    need(len(action_bytes) == 40 and hashlib.sha256(action_bytes).hexdigest() == probe["raw5_sha256"],
         "probe bytes")
    action = action_bytes[8*(step-1):8*step].hex()
    scope = dict(branch=branch, anchor_index=anchor["index"], condition=anchor["condition"],
                 arm=anchor["arm"], candidate_role=anchor["role"], probe=probe["name"])
    def one(items, label):
        need(len(items) == 1, "unique " + label)
        return items[0]
    def same_scope(obj):
        return all(obj.get(k) == v for k, v in scope.items())
    start = one([e for e in events if e["kind"] == "suffix_start_verified" and same_scope(e)], "start")
    need(all(start["checkpoint"][k] == anchor["checkpoint"][k] for k in
             ("step", "state", "snapshot", "file_sha256", "pixel_sha256")), "saved anchor checkpoint")
    def operations(kind):
        return [e for e in events if e["kind"] == kind and same_scope(e["receipt"])
                and e["receipt"].get("resource") == "public_step"
                and e["receipt"].get("phase") == "probe" and e["receipt"].get("step") == step]
    intent, returned = one(operations("operation_dispatch_intent"), "intent"), one(operations("operation_returned"), "return")
    ri, rr = intent["receipt"], returned["receipt"]
    need(ri["status"] == "pending" and rr["status"] == "returned"
         and dict(ri, status="returned") == rr and rr["raw_bits"] == action, "operation identity")
    trans = one([e for e in events if e["kind"] == "returned_transition" and same_scope(e)
                 and e.get("phase") == "probe" and e.get("step") == step], "transition")
    frame = one([e for e in events if e["kind"] == "fresh_observation" and same_scope(e)
                 and e["receipt"].get("phase") == "probe" and e["receipt"].get("step") == step
                 and e["receipt"].get("overlay_index") == 0], "guard frame")
    need(start["index"] < intent["index"] < returned["index"] < trans["index"] < frame["index"]
         and trans["requested_bits"] == trans["actual_bits"] == action
         and trans["actual_dtype"] == "<f4", "transition order/action")
    receipt = frame["receipt"]
    state = trans["state"]
    need(type(state) is list and len(state) == 7 and all(type(x) in (int, float) and math.isfinite(x) for x in state)
         and trans["snapshot"]["state"] == state == receipt["state"]
         and trans["snapshot"] == receipt["snapshot"] == frame["pre"]["snapshot"] == frame["post"]["snapshot"]
         and frame["pre"] == frame["post"] and receipt == row["last_observation"]
         and receipt["condition"] == "off" and row["own_returned"] == 5 + step
         and len(row["responses"]) == step - 1, "returned state/render identity")
    need(trans["snapshot"]["state_float64_bits"] == struct.pack("<7d", *state).hex(), "state bits")
    out = [dict(coordinate=i, value=state[i]) for i in range(4) if not 0 <= state[i] <= 512]
    need(bool(out), "no proven canvas violation")
    need(type(image) is np.ndarray and image.shape == (224, 224, 3) and image.dtype == np.uint8
         and old.runner.prior.base.moving.image_sha(image) == receipt["pixel_sha256"], "original RGB pixels")
    blue = int(np.count_nonzero(np.all(image == [65, 105, 225], axis=-1)))
    return dict(anchor=anchor["index"], probe=branch, step=step, action_float32_hex=action,
                returned_state=state, out_of_bounds=out, frame_sha256=receipt["file_sha256"],
                blue_pixels=blue, rgb_guard_accepted=False, endpoint_accepted=False,
                historical_failure_preserved=True, continuation_authorized=False,
                unobserved_exception_excluded=False, P1_admission=False)


def build():
    need(old.runner.physical.sha(PREPARE) == PREPARE_SHA, "prepare SHA")
    binding = DIRECTORY.parent / "run_binding.json"
    need(old.runner.physical.sha(binding) == BINDING_SHA, "run binding SHA")
    b = old.runner.physical.native.read(binding, cap=32 * 1024**2)
    need(b["prepare_sha256"] == PREPARE_SHA and b["prepared"] == str(PREPARE), "prepare binding")
    need(old.runner.physical.sha(DIRECTORY / "report.json") == REPORT_SHA, "report SHA")
    report = old.runner.physical.native.read(DIRECTORY / "report.json", cap=64 * 1024**2)
    need(report["status"] == "partial_or_failed" and report["anchor"] == 5
         and report["errors"] == [dict(index=16, stage="branch", type="Rejected", message="fresh RGB/canvas guard")]
         and old.runner.physical.sha(DIRECTORY / "journal.jsonl") == report["journal_sha256"], "failure seal")
    events = saved.events(DIRECTORY / "journal.jsonl", report["journal_chain_tail"])
    prepare = old.runner.physical.native.read(PREPARE, cap=32 * 1024**2)
    anchor, probe, row = prepare["anchors"][5], prepare["probe_sets"][5][16], report["rows"][16]
    frame_path = DIRECTORY / row["last_observation"]["png"]
    need(frame_path.parent == DIRECTORY and frame_path.resolve() == frame_path and frame_path.is_file()
         and not frame_path.is_symlink() and old.runner.physical.sha(frame_path) == row["last_observation"]["file_sha256"],
         "incident image file")
    with Image.open(frame_path) as image:
        pixels = np.asarray(image)
    result = incident(events, anchor, probe, row, pixels)
    result.update(schema="p1_matched_canvas_incident_audit_20261030_codex.v1",
                  prepare_sha256=PREPARE_SHA, run_binding_sha256=BINDING_SHA,
                  report_sha256=REPORT_SHA, journal_sha256=report["journal_sha256"],
                  independent_confirmation=False)
    return result


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True, allow_nan=False, indent=2))
