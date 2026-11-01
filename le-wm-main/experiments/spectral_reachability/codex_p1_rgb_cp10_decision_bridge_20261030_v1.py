"""CP10 RGB decision receipt; not a complete physical runner or saved consumer.

The caller must authenticate the journal chain and the frozen goal separately.
Only the verified fresh pixels, frozen goal pixels and current model packet
reach the candidate. The receipt itself is not evidence of physical execution.
"""
import hashlib
import json

import numpy as np

from . import codex_p1_rgb_feedback_candidate_20261030_v1 as candidate
from . import codex_p1_two_stage_feedback_20261030_v1 as feedback
from . import codex_p1_two_stage_journal_feedback_20261030_v1 as journal


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pixels(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (224, 224, 3):
        raise ValueError("RGB ABI")
    return np.ascontiguousarray(image)


def decide(events, folder, row_index, condition, phase1_packet, after_index,
           goal_rgb, expected_goal_pixel_sha256, model_packet):
    """Construct a receipt only from authenticated CP10 journal input.

    The goal hash must be obtained from an independently authenticated fixed
    task bundle, never from the same untrusted receipt being checked.
    """
    goal = _pixels(goal_rgb)
    if _sha(goal.tobytes()) != expected_goal_pixel_sha256:
        raise ValueError("frozen goal mismatch")
    bound = journal.from_journal(events, folder, row_index, condition, phase1_packet,
                                 after_index=after_index)
    current = _pixels(feedback.rgb_pixels(bound["png"]))
    model_raw = feedback.packet_bytes(model_packet)
    result = candidate.decide(current, goal, model_packet)
    output = result["packet"]
    return {
        "schema": "codex_p1_rgb_cp10_decision_bridge_20261030_v1",
        "row": row_index, "condition": condition,
        "after_index": after_index, "last_render_index": bound["last_render_index"],
        "cp10_png_sha256": bound["png_sha256"],
        "cp10_pixel_sha256": bound["pixel_sha256"],
        "goal_pixel_sha256": expected_goal_pixel_sha256,
        "phase1_packet_sha256": _sha(feedback.packet_bytes(phase1_packet)),
        "model_packet_f32_hex": model_raw.hex(),
        "decision": result["decision"], "reason": result["reason"],
        "features": result["features"],
        "executed_packet_f32_hex": None if output is None else feedback.packet_bytes(output).hex(),
    }


def verify(receipt, events, folder, row_index, condition, phase1_packet, after_index,
           goal_rgb, expected_goal_pixel_sha256, model_packet):
    """Recompute every receipt field; input journal must already be authenticated."""
    expected = decide(events, folder, row_index, condition, phase1_packet, after_index,
                      goal_rgb, expected_goal_pixel_sha256, model_packet)
    if json.dumps(receipt, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("RGB decision receipt mismatch")
    return expected


def verify_dispatch(receipt, probe_transitions):
    """Check phase-two dispatch literal and stop, allowing only a terminal prefix."""
    if type(probe_transitions) is not list:
        raise ValueError("transitions")
    packet_hex = receipt["executed_packet_f32_hex"]
    if packet_hex is None:
        if receipt["decision"] != "rgb_rejected" or probe_transitions:
            raise ValueError("rejection dispatched")
        return "rgb_rejected"
    raw = bytes.fromhex(packet_hex)
    if len(raw) != 40 or len(probe_transitions) > 5:
        raise ValueError("phase-two packet")
    previous_index = receipt["last_render_index"]
    for offset, event in enumerate(probe_transitions):
        if (event.get("kind") != "returned_transition" or event.get("phase") != "probe"
                or type(event.get("index")) is not int or event["index"] <= previous_index
                or event.get("step") != offset + 6 or event.get("branch") != 0
                or event.get("anchor_index") != receipt["row"]
                or event.get("condition") != receipt["condition"]
                or event.get("actual_dtype") != "<f4"
                or event.get("requested_bits") != raw[offset*8:offset*8+8].hex()
                or event.get("actual_bits") != event.get("requested_bits")
                or type(event.get("terminated")) is not bool
                or type(event.get("truncated")) is not bool):
            raise ValueError("phase-two transition mismatch")
        previous_index = event["index"]
        stopped = event["terminated"] or event["truncated"]
        if stopped and offset != len(probe_transitions)-1:
            raise ValueError("action after native stop")
    if probe_transitions and (probe_transitions[-1]["terminated"] or probe_transitions[-1]["truncated"]):
        return "native_stopped"
    if len(probe_transitions) != 5:
        raise ValueError("incomplete phase-two packet")
    return "probe_completed"
