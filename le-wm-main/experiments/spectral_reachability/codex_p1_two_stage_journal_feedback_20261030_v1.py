"""Bind phase-two RGB to physical transition and fresh-render journal events.

This is a bridge, not a launcher or a proof that a journal was produced by
physical execution. Saved callers must first authenticate the complete journal
hash chain and source/run identity; live callers must use their trusted store.
"""
import hashlib
from pathlib import Path

import numpy as np

from . import codex_p1_two_stage_feedback_20261030_v1 as feedback


def _one(items, message):
    feedback.need(len(items) == 1, message)
    return items[0]


def from_journal(events, folder, row_index, condition, packet, *, after_index):
    """Return verified selected fifth fresh PNG and receipts for an observed phase.

    The supplied event prefix must end at or after the fifth capture and before
    another probe transition. Scope is branch zero, as in the single-row H10
    store. This function does not interpret results or choose rows.
    """
    raw = feedback.packet_bytes(packet)
    folder = Path(folder)
    feedback.need(folder.is_absolute() and folder.is_dir() and not folder.is_symlink()
                  and condition in ("on", "off") and type(after_index) is int, "source scope")
    feedback.need(type(events) is list and all(type(e) is dict and type(e.get("index")) is int
                  for e in events) and all(a["index"] < b["index"] for a, b in zip(events, events[1:])),
                  "ordered journal prefix")
    transitions = [e for e in events if e.get("kind") == "returned_transition"
                   and e.get("phase") == "probe" and e.get("index", -1) > after_index]
    feedback.need(len(transitions) == 5 and [e.get("step") for e in transitions] == list(range(1, 6)),
                  "exact first phase transitions; no later action")
    receipts, chosen_png, previous = [], None, after_index
    for step, trans in enumerate(transitions, 1):
        feedback.need(trans.get("branch") == 0 and trans.get("anchor_index") == row_index
                      and trans.get("condition") == condition and trans["index"] > previous
                      and trans.get("actual_dtype") == "<f4"
                      and trans.get("requested_bits") == trans.get("actual_bits") == raw[(step-1)*8:step*8].hex()
                      and type(trans.get("terminated")) is bool and type(trans.get("truncated")) is bool
                      and not trans["terminated"] and not trans["truncated"], "actual transition identity or stop")
        frame_events = [e for e in events if e.get("kind") == "fresh_observation"
                        and e.get("receipt", {}).get("phase") == "probe"
                        and e["receipt"].get("step") == step and e["index"] > trans["index"]]
        feedback.need(len(frame_events) == 3 and [e["receipt"].get("overlay_index") for e in frame_events]
                      == [0, 1, 2], "fresh off/on/off render sequence")
        next_index = transitions[step]["index"] if step < 5 else None
        feedback.need(next_index is None or frame_events[-1]["index"] < next_index, "capture before next action")
        selected = 1 if condition == "on" else 0
        images = []
        for j, event in enumerate(frame_events):
            rec = event["receipt"]
            feedback.need(event.get("branch") == 0 and event.get("anchor_index") == row_index
                          and event.get("condition") == condition and rec.get("condition") == ("on" if j == 1 else "off")
                          and event.get("pre") == event.get("post") and
                          rec.get("snapshot") == trans.get("snapshot") == event.get("pre", {}).get("snapshot")
                          and rec.get("state") == trans.get("state") == rec["snapshot"].get("state"),
                          "frame versus actual returned state")
            name = rec.get("png")
            feedback.need(type(name) is str and Path(name).name == name, "flat image path")
            path = folder / name
            feedback.need(path.is_file() and not path.is_symlink(), "original image required")
            png = path.read_bytes()
            feedback.need(hashlib.sha256(png).hexdigest() == rec.get("file_sha256"), "image file SHA")
            pixels = feedback.rgb_pixels(png)
            feedback.need(hashlib.sha256(pixels.tobytes()).hexdigest() == rec.get("pixel_sha256"),
                          "pixel SHA")
            state = np.asarray(rec["state"], dtype=np.float64)
            feedback.need(state.shape == (7,) and np.isfinite(state).all()
                          and ((state[:4] >= 0) & (state[:4] <= 512)).all()
                          and np.any(np.all(pixels == [65, 105, 225], axis=-1)), "RGB/canvas guard")
            images.append(pixels)
            if step == 5 and j == selected:
                chosen_png = png
        feedback.need(np.array_equal(images[0], images[2]), "off render consistency")
        rec = frame_events[selected]["receipt"]
        receipts.append(dict(row=row_index, step=step, action_f32_hex=trans["actual_bits"],
                             terminated=trans["terminated"], truncated=trans["truncated"],
                             pixel_sha256=rec["pixel_sha256"]))
        previous = frame_events[-1]["index"]
    feedback.verify_feedback(row_index, packet, receipts, chosen_png)
    return {"png": chosen_png, "receipts": receipts, "last_render_index": previous,
            "png_sha256": hashlib.sha256(chosen_png).hexdigest(),
            "pixel_sha256": receipts[-1]["pixel_sha256"]}
