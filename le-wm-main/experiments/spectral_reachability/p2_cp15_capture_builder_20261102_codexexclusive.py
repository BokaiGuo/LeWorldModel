"""Build one fixed capture from the saved B0 P2 native execution, CPU only."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p2_multistart_joint_physical_20261030_v1 as physical
from . import p2_shared_rgb_pair_wiring_20261102_codexexclusive as wire

NAME = "p2_cp15_capture_builder_20261102_codexexclusive"
ROOT = physical.ROOT
RUN = ROOT / "results/codex_p2_multistart_joint_physical_20261030_v1_B0_RUN_01"
RECEIPT = ROOT / "results/codex_p2_multistart_joint_physical_20261030_v1_B0_RECEIPT_01.json"
RECEIPT_SHA = "88a8d7bb3f04479dac0150a7c917b9b1840bfc67d962437c0cfaad7fcb98cddf"
REPORT_SHA = "7214ee3b3a428149091db4e013eb27efaed5b41311ab8c69e5fad113f9b4b0ac"
COMPLETION_SHA = "2227b1e83fdd4e1028fce69ab333b965f240a3ea312207b7d06efdaecb59aa39"
need = physical.require


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def derive():
    """Rebuild strict native receipt, then extract only actual B0 branch 0."""
    need(physical.sha(RECEIPT) == RECEIPT_SHA, "pinned physical receipt")
    receipt = physical.consume(RUN, REPORT_SHA, COMPLETION_SHA)
    need(receipt == json.loads(RECEIPT.read_bytes()) and receipt["job_index"] == 0
         and receipt["both_job_trajectories_full25"] is True
         and receipt["P1_admission"] is False and receipt["P2_admission"] is False,
         "physical receipt reconstruction")
    row = receipt["rows"][0]
    need(row["policy"] == "joint_multistart" and row["root_slot"] == 0
         and row["initial_exact"] is True and row["status"] == "horizon_completed", "fixed source branch")
    journal_path = RUN / "journal.jsonl"
    events = [physical.native.decode(line) for line in journal_path.read_bytes().splitlines()]
    actions = []
    transitions = [e for e in events if e["kind"] == "returned_transition"
                   and e.get("branch") == 0 and e["phase"] == "plan" and e["step"] <= 15]
    need([e["step"] for e in transitions] == list(range(1, 16)), "complete returned prefix")
    for e in transitions:
        step = e["step"]
        ops = [x for x in events if x["kind"] in ("operation_dispatch_intent", "operation_returned")
               and x["receipt"]["resource"] == "public_step" and x["receipt"].get("branch") == 0
               and x["receipt"].get("phase") == "plan" and x["receipt"].get("step") == step]
        need(len(ops) == 2 and [x["kind"] for x in ops] == ["operation_dispatch_intent", "operation_returned"]
             and ops[0]["index"] < ops[1]["index"] < e["index"]
             and ops[0]["receipt"]["raw_bits"] == ops[1]["receipt"]["raw_bits"]
             == e["requested_bits"] == e["actual_bits"] and e["actual_dtype"] == "<f4"
             and e["terminated"] is False and e["truncated"] is False, "physical action chain")
        value = np.frombuffer(bytes.fromhex(e["actual_bits"]), dtype="<f4")
        need(value.shape == (2,) and np.isfinite(value).all() and (np.abs(value) <= 1).all(), "action bytes")
        actions.append({"step": step, "dispatch": value.tolist(), "returned": value.tolist(),
                        "terminated": False, "truncated": False,
                        "source_dispatch_event": ops[0]["index"], "source_return_event": e["index"]})
    payloads = {}
    frames = []
    for step in (5, 10, 15):
        found = [e for e in events if e["kind"] == "fresh_observation" and e.get("branch") == 0
                 and e["receipt"]["phase"] == "plan" and e["receipt"]["step"] == step
                 and e["receipt"]["overlay_index"] == 0]
        need(len(found) == 1, "actual frame uniqueness")
        event = found[0]
        f = event["receipt"]
        previous = transitions[step - 1]
        need(previous["index"] < event["index"] and
             (step == 15 or event["index"] < transitions[step]["index"])
             and f["snapshot"] == previous["snapshot"] and f["state"] == previous["state"]
             and event["pre"] == event["post"]
             and event["pre"]["goal_rgb_sha256"] == row["goal_pixel_sha256"], "frame/transition/goal binding")
        raw = (RUN / f["png"]).read_bytes()
        need(digest(raw) == f["file_sha256"], "source image bytes")
        name = f"frame_{step:02d}.png"
        payloads[name] = raw
        frames.append({"step": step, "file": name, "file_sha256": digest(raw),
                       "pixel_sha256": f["pixel_sha256"], "source_event": event["index"]})
    data = physical.source_data()
    root = data["roots"][0]
    goal_rgb = np.asarray(root["packets"]["off"]["goal_rgb"])
    need(root["root_sha256"] == row["root_sha256"] and goal_rgb.shape == (224, 224, 3)
         and goal_rgb.dtype == np.uint8 and wire.pixel_sha(goal_rgb) == row["goal_pixel_sha256"],
         "original goal pixel/root binding")
    buffer = io.BytesIO()
    Image.fromarray(goal_rgb).save(buffer, format="PNG")
    goal_png = buffer.getvalue()
    payloads["goal.png"] = goal_png
    bundle = {"schema": wire.NAME + ".capture.v1", "base": 0, "frames": frames,
              "goal": {"file": "goal.png", "file_sha256": digest(goal_png),
                       "pixel_sha256": wire.pixel_sha(goal_rgb)}, "actions": actions,
              "source": {"schema": NAME + ".source.v1", "branch": 0, "job": 0,
                         "physical_report_sha256": REPORT_SHA, "completion_sha256": COMPLETION_SHA,
                         "physical_receipt_sha256": RECEIPT_SHA,
                         "journal_sha256": physical.sha(journal_path),
                         "root_sha256": row["root_sha256"], "task_id": row["task_id"],
                         "goal_pixel_sha256": row["goal_pixel_sha256"], "fresh_root": False,
                         "source_scope": "reconstructed_saved_native_receipt_known_root",
                         "P1_admission": False, "P2_admission": False}}
    return bundle, payloads


def _write(path, raw):
    with Path(path).open("xb") as stream:
        need(stream.write(raw) == len(raw), "short write")
        stream.flush()
        os.fsync(stream.fileno())


def create(output):
    bundle, payloads = derive()
    output = Path(output)
    output.mkdir(mode=0o700)
    for name, raw in payloads.items():
        _write(output / name, raw)
    raw = (json.dumps(bundle, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    _write(output / "capture.json", raw)
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    wire.load_capture(output / "capture.json", digest(raw))
    return digest(raw)


def verify(output, expected):
    output = Path(output).resolve(strict=True)
    bundle, payloads = derive()
    path = output / "capture.json"
    need(path.is_file() and not path.is_symlink() and physical.sha(path) == expected
         and json.loads(path.read_bytes()) == bundle, "capture reconstruction")
    need({p.name for p in output.iterdir()} == {*payloads, "capture.json"}, "capture inventory")
    for name, raw in payloads.items():
        path = output / name
        need(path.is_file() and not path.is_symlink() and path.read_bytes() == raw, "capture payload")
    wire.load_capture(output / "capture.json", expected)
    return {"status": "verified_saved_known_root_source", "capture_sha256": expected,
            "fresh_root": False, "P1_admission": False, "P2_admission": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("create", "verify"))
    p.add_argument("--output", required=True)
    p.add_argument("--sha256")
    args = p.parse_args()
    output = Path(args.output).absolute()
    need(output.parent == ROOT / "results" and output.name.startswith(NAME + "_"),
         "exclusive results path")
    if args.command == "create":
        need(args.sha256 is None and not os.path.lexists(output), "new output only")
        value = {"capture": str(output / "capture.json"), "sha256": create(output)}
    else:
        need(args.sha256 is not None, "external SHA required")
        value = verify(output, args.sha256)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
