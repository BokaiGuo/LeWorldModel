"""Produce two isolated development-only blind packets from the sealed witness run."""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p1_devroot_witness_producer_20261101_hypatia as src
from experiments.spectral_reachability import p1_devroot_witness_verify_20261101_hypatia as audit
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as encoder

NAME = "p1_devroot_blind_payload_20261101_hypatia"
OUT = src.ROOT / ("results/" + NAME + "_LOCAL01")
VERIFY_RECEIPT = src.ROOT / "results/p1_devroot_witness_verify_20261101_hypatia_LOCAL01.json"
PINS = {
    VERIFY_RECEIPT: "7b2fdd7cbdf9e33acd5391df3b871c20d39167485b2d64556f9be16b32fc556c",
    Path(audit.__file__).resolve(): "1b500c54569ecc226ae7ae5fab005144f124720d49e04e1fd2113eac2f0b53a9",
    Path(encoder.__file__).resolve(): "246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63",
}


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def one(events, *, kind, slot, phase, step):
    found = [e for e in events if e.get("kind") == kind and e.get("slot") == slot
             and e.get("phase") == phase and e.get("step") == step]
    need(len(found) == 1, "unique source observation")
    return found[0]


def read_frame(event):
    path = src.OUT / event["file"]
    need(path.parent == src.OUT and path.is_file() and not path.is_symlink()
         and src.v4.sha(path) == event["file_sha256"], "source frame file")
    with Image.open(path) as im:
        pixels = np.asarray(im).copy()
    need(pixels.dtype == np.uint8 and pixels.shape == (224, 224, 3)
         and hashlib.sha256(pixels.tobytes()).hexdigest() == event["pixel_sha256"], "source pixels")
    return pixels


def warmup_actions(events, slot):
    selected = [e for e in events if e.get("kind") == "command" and e.get("slot") == slot
                and e.get("phase") == "replay0"][:10]
    need([e["step"] for e in selected] == list(range(1, 11)), "warmup command order")
    actual = np.frombuffer(b"".join(bytes.fromhex(e["raw_float32_bits"]) for e in selected),
                           dtype=np.float32).copy().reshape(10, 2)
    need(np.array_equal(actual, encoder.fixed_raw()), "warmup action bits")
    return actual


def collect():
    for path, digest in PINS.items():
        need(src.v4.sha(path) == digest, "pinned verifier or encoder drift")
    checked = audit.verify()
    need(checked == json.loads(VERIFY_RECEIPT.read_bytes()), "source verification receipt mismatch")
    prepared = json.loads(src.PREP.read_bytes())
    need(prepared["slots"] == [0, 1] and not prepared["planner_authorized"]
         and prepared["development_roots"] and prepared["freshness"] == "unknown",
         "development-only source")
    receipt = json.loads((src.OUT / "receipt.json").read_bytes())
    need([r["slot"] for r in receipt["rows"]] == [0, 1] and all(
        r["status"] == "construction_and_replays_completed" for r in receipt["rows"]),
        "fixed two complete source slots")
    events = [json.loads(line) for line in (src.OUT / "journal.jsonl").read_bytes().splitlines()]
    packets, provenance = [], []
    for slot in (0, 1):
        history_events = [one(events, kind="observation", slot=slot, phase="replay0", step=i)
                          for i in (0, 5, 10)]
        goal_event = one(events, kind="observation", slot=slot, phase="construction", step=20)
        goal_binding = json.loads((src.OUT / f"s{slot}_construction_goal.json").read_bytes())
        need(goal_binding["goal_rgb_file"] == goal_event["file"]
             and goal_binding["goal_pixel_sha256"] == goal_event["pixel_sha256"], "terminal goal binding")
        packet = {"rgb_history": np.stack([read_frame(e) for e in history_events]),
                  "raw_action_history": warmup_actions(events, slot),
                  "goal_rgb": read_frame(goal_event)}
        encoder.check_packet(packet)
        packets.append(packet)
        provenance.append({"slot": slot, "observed_frames": [e["file"] for e in history_events],
                           "observed_pixel_sha256": [e["pixel_sha256"] for e in history_events],
                           "goal_frame": goal_event["file"], "goal_pixel_sha256": goal_event["pixel_sha256"],
                           "raw_action_sha256": hashlib.sha256(packet["raw_action_history"].tobytes()).hexdigest()})
    return packets, provenance, checked


def build():
    packets, provenance, checked = collect()
    OUT.mkdir()
    rows = []
    for i, packet in enumerate(packets):
        buf = io.BytesIO()
        np.savez_compressed(buf, **packet)
        raw = buf.getvalue()
        decoded = encoder.decode_blind(raw)
        need(set(decoded) == set(packet) and all(np.array_equal(decoded[k], packet[k]) for k in packet),
             "encoder decoder readback")
        path = OUT / f"dev_slot{i}_blind_inputs.npz"
        with path.open("xb") as f:
            need(f.write(raw) == len(raw), "short payload write")
            f.flush()
            os.fsync(f.fileno())
        digest = src.v4.sha(path)
        checked_packet = encoder.load_blind(path, digest)
        need(all(np.array_equal(checked_packet[k], packet[k]) for k in packet), "encoder loader readback")
        rows.append({"slot": i, "path": str(path), "sha256": digest, **provenance[i]})
    result = {"schema": NAME + ".receipt.v1", "slots": rows, "source_verify": checked,
              "bindings": {str(k): v for k, v in PINS.items()},
              "builder_sha256": src.v4.sha(__file__), "with_target": False,
              "witness_goal": "construction_terminal20_actual_rgb", "fixed_denominator": 2,
              "scope": "development encoding only; no MAIN approval or planner authorization",
              "planner_authorized": False, "P1_admission": False, "freshness": "unknown",
              "training_disjoint": "unknown", "history_coverage": "unknown"}
    data = src.accounting.canonical(result) + b"\n"
    with (OUT / "receipt.json").open("xb") as f:
        need(f.write(data) == len(data), "short receipt")
        f.flush()
        os.fsync(f.fileno())
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
