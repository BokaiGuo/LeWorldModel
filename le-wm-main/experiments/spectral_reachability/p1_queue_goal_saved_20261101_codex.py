"""Read-only verifier for a development reset, goal RGB and history receipt."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_queue_goal_history_20261101_codex as prod

q = prod.q
require = q.require


def verify(queue_path, expected_queue, slot_id, attempt, expected_report):
    path = Path(queue_path).resolve(strict=True)
    require(prod.sha(path) == expected_queue, "external queue SHA")
    queue = q.read_json(path)
    require(queue == q.freeze(queue["spec"]) and all(
        queue["spec"][k] == v for k, v in prod.bindings().items()), "source closure")
    slot = next((s for s in queue["spec"]["slots"] if s["slot_id"] == slot_id), None)
    require(slot is not None and slot["role"] == "development"
            and type(attempt) is int and 0 <= attempt < len(slot["reset_seeds"]), "slot")
    seed = slot["reset_seeds"][attempt]
    base = path.parent / (path.name + ".native_reset")
    directory = base / f"{slot_id}_{attempt:04d}.warmup"
    report_path, journal = directory / "report.json", directory / "journal.jsonl"
    require(prod.sha(report_path) == expected_report, "external report SHA")
    report = q.read_json(report_path)
    require(report["status"] == "complete_development" and report["error"] is None
            and report["steps_returned"] == 10 and report["frames_recorded"] == 3
            and report["P1_admission"] is False and report["independent_confirmation"] is False,
            "complete development only")
    ledger = prod.events(base / "queue_ledger.jsonl")
    ledger_audit = q.validate(queue, ledger)
    matches = [e for e in ledger if e["slot_id"] == slot_id and e["attempt"] == attempt]
    require(len(matches) == 2 and [e["kind"] for e in matches] == ["reset_attempt", "reset_return"]
            and all(e["seed"] == seed for e in matches), "queue dispatch/return")
    require(prod.sha(journal) == report["journal_sha256"], "journal SHA")
    rows = prod.events(journal)
    previous = None
    for index, row in enumerate(rows):
        body = {k: v for k, v in row.items() if k != "sha256"}
        require(row["index"] == index and row["previous"] == previous and row["sha256"] == q.digest(body),
                "journal chain")
        previous = row["sha256"]
    require(rows and previous == report["journal_tail"] and rows[-1]["kind"] == "run_finished"
            and rows[0]["kind"] == "reset_authorized"
            and rows[0]["queue_attempt_sha256"] == matches[0]["sha256"]
            and report["queue_attempt_sha256"] == matches[0]["sha256"], "journal binding")
    reset_path, source_path = directory / "reset_receipt.json", directory / "source_receipt.json"
    reset, source = q.read_json(reset_path), q.read_json(source_path)
    require(source["queue_attempt_sha256"] == matches[0]["sha256"]
            and source["history"]["reset_receipt_sha256"] == prod.sha(reset_path)
            and reset["state"] == reset["snapshot"]["state"], "reset identity")
    binding = reset["goal_rgb"]
    goal_path = (directory / binding["file"]).resolve(strict=True)
    require(goal_path.parent == directory.resolve() and prod.sha(goal_path) == binding["file_sha256"]
            and binding["source"] == "same_instance_reset_info_goal"
            and source["history"]["goal_rgb"] == binding, "goal file/binding")
    with Image.open(goal_path) as image:
        goal_rgb = np.asarray(image.convert("RGB"), np.uint8)
    require(goal_rgb.shape == (224, 224, 3)
            and hashlib.sha256(goal_rgb.tobytes()).hexdigest() == binding["pixel_sha256"],
            "goal pixels")
    seals = [r for r in rows if r["kind"] == "source_sealed"]
    require(len(seals) == 1 and seals[0]["source_sha256"] == prod.sha(source_path)
            and seals[0]["history_sha256"] == source["history_sha256"]
            and q.digest(source["history"]) == source["history_sha256"], "source/history seal")
    dispatch = [r for r in rows if r["kind"] == "step_dispatch"]
    native_returns = [r for r in rows if r["kind"] == "step_native_return"]
    returned = [r for r in rows if r["kind"] == "step_return"]
    require(len(dispatch) == len(native_returns) == len(returned) == 10 and all(
        d["step"] == r["step"] == i and d["action"] == r["action"] == [0.0, 0.0]
        and n["step"] == i and d["index"] < n["index"] < r["index"]
        and n["terminated"] == r["terminated"] is False and n["truncated"] == r["truncated"] is False
        for i, (d, n, r) in enumerate(zip(dispatch, native_returns, returned, strict=True), 1)),
        "actual step sequence")
    history = source["history"]
    require(history["actions"] == [r["action"] for r in returned]
            and history["goal"] == reset["goal"] and history["final_state"] == returned[-1]["state"]
            and source["final_snapshot"]["state"] == history["final_state"], "source state/action")
    frames = history["frames"]
    captures = [r for r in rows if r["kind"] == "rgb_captured"]
    require([f["step"] for f in frames] == [0, 5, 10] and len(captures) == 3, "frame roster")
    for frame, capture in zip(frames, captures, strict=True):
        file = (directory / frame["file"]).resolve(strict=True)
        require(file.parent == directory.resolve() and prod.sha(file) == frame["file_sha256"], "frame file")
        with Image.open(file) as image:
            pixels = np.asarray(image.convert("RGB"), np.uint8)
        require(pixels.shape == (224, 224, 3)
                and hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"]
                and all(capture[k] == v for k, v in frame.items())
                and frame["state"] == (reset["state"] if frame["step"] == 0
                                       else returned[frame["step"] - 1]["state"])
                and frame["snapshot"]["state"] == frame["state"], "frame pixels/state")
    payload = matches[1]["payload"]
    require(payload == dict(state_sha256=q.digest(history["final_state"]),
                            goal_sha256=q.digest(history["goal"]), history_sha256=source["history_sha256"],
                            snapshot_sha256=q.digest(source["final_snapshot"])), "queue receipt identity")
    return dict(schema="p1_queue_goal_saved_20261101_codex.audit.v1",
                status="saved_integrity_checked_not_independent_replay", report_sha256=expected_report,
                queue_sha256=expected_queue, slot_id=slot_id, seed=seed, frames_verified=3,
                returned_steps=10, goal_rgb_verified=True, queue_ledger_audit=ledger_audit,
                historical_coverage="unknown", training_membership="unknown",
                independent_confirmation=False, P1_admission=False)


def main():
    parser = argparse.ArgumentParser()
    for name in ("queue", "queue-sha256", "slot", "report-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--attempt", required=True, type=int)
    args = parser.parse_args()
    value = verify(args.queue, args.queue_sha256, args.slot, args.attempt, args.report_sha256)
    print(q.write_new(args.output, value))


if __name__ == "__main__":
    main()
