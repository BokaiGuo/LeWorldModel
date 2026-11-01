"""Read-only independent artifact verification; no planner or simulator."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p1_devroot_witness_ledger_20261101_hypatia as l
from experiments.spectral_reachability import p1_devroot_witness_producer_20261101_hypatia as p


def verify():
    directory = p.OUT
    receipt = json.loads((directory / "receipt.json").read_bytes())
    if p.v4.sha(p.PREP) != receipt["prepare_sha256"]:
        raise ValueError("prepare hash")
    prep = json.loads(p.PREP.read_bytes())
    for path, sha in {**prep["files"], **prep["runtime"]["loaded_module_files"]}.items():
        if p.v4.sha(path) != sha:
            raise ValueError("binding drift: " + path)
    previous, events = None, []
    for idx, raw in enumerate((directory / "journal.jsonl").read_bytes().splitlines()):
        record = json.loads(raw)
        digest = record.pop("sha256")
        if (record["index"] != idx or record["previous_sha256"] != previous or
                hashlib.sha256(l.canonical(record)).hexdigest() != digest):
            raise ValueError("journal chain")
        previous = digest
        events.append(record)
    counts = {k: {"attempted": 0, "completed": 0} for k in l.CAPS}
    observations, commands = {}, {}
    for event in events:
        kind = event["kind"]
        if kind in ("attempt", "complete"):
            counts[event["resource"]]["attempted" if kind == "attempt" else "completed"] += 1
        if kind == "command":
            commands.setdefault((event["slot"], event["phase"]), []).append(
                (event["step"], event["raw_float32_bits"]))
        if kind != "observation":
            continue
        key = (event["slot"], event["phase"], event["step"])
        if key in observations:
            raise ValueError("duplicate observation")
        path = directory / event["file"]
        if p.v4.sha(path) != event["file_sha256"]:
            raise ValueError("frame file hash")
        with Image.open(path) as image:
            pixels = np.array(image)
        if pixels.shape != (224, 224, 3) or pixels.dtype != np.uint8:
            raise ValueError("frame format")
        if hashlib.sha256(pixels.tobytes()).hexdigest() != event["pixel_sha256"]:
            raise ValueError("frame pixels")
        observations[key] = event
    if counts != receipt["counts"]:
        raise ValueError("count mismatch")
    for slot in (0, 1):
        for phase in ("construction", "replay0", "replay1"):
            ordered = commands[(slot, phase)]
            if len(ordered) != 20:
                raise ValueError("command count")
            for step, action in enumerate(p.moving.warmup_commands(), 1):
                if ordered[step - 1] != (step, action.tobytes().hex()):
                    raise ValueError("warmup bits")
            commit = json.loads((directory / f"s{slot}_{phase}_commit.json").read_bytes())
            action_bytes = bytes.fromhex(commit["commands_bits"])
            action = np.frombuffer(action_bytes, dtype=np.float32).reshape(10, 2)
            if any(ordered[10 + i] != (i + 1, a.tobytes().hex()) for i, a in enumerate(action)):
                raise ValueError("witness bits")
            for step in (0, 5, 10, *range(11, 21)):
                actual = observations[(slot, phase, step)]
                ref = observations[(slot, "construction", step)]
                if any(actual[k] != ref[k] for k in ("state", "snapshot", "pixel_sha256")):
                    raise ValueError("replay mismatch")
        goal = json.loads((directory / f"s{slot}_construction_goal.json").read_bytes())
        last = observations[(slot, "construction", 20)]
        if goal["goal_state"] != last["state"] or goal["goal_pixel_sha256"] != last["pixel_sha256"]:
            raise ValueError("terminal goal mismatch")
    return {"verified": True, "prepare_sha256": receipt["prepare_sha256"],
            "receipt_sha256": p.v4.sha(directory / "receipt.json"),
            "journal_sha256": p.v4.sha(directory / "journal.jsonl"),
            "journal_events": len(events), "observations": len(observations),
            "planner_authorized": False, "freshness": "unknown", "training_disjoint": "unknown"}


if __name__ == "__main__":
    result = verify()
    data = l.canonical(result) + b"\n"
    path = p.ROOT / ("results/p1_devroot_witness_verify_20261101_hypatia_LOCAL01.json")
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        import os
        os.fsync(f.fileno())
    print(json.dumps(result, indent=2))
