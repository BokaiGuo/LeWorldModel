"""Verify complete pairs in a canvas-guard partial without imputing failed rows."""
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p3_raw_h3_guard_continuation_20261031 as guard
from . import p3_raw_h3_physical_pair_20261031 as pair

source, host, require = guard.source, guard.host, guard.require
kernel, batch = source.batch.kernel, source.batch


def validate_complete(row, plan, ref, events, frame_reader):
    i = row["branch_index"]
    arm = f"d{plan['decision_index']}_c{plan['candidate']}_{plan['variant']}"
    require(row["selected_plan"] == {**plan, "arm": arm}
            and row["root_slot"] == plan["root_slot"] and row["initial_exact"] is True
            and row["first_error"] is None and not row["secondary_errors"]
            and row["warmup_returned"] == 10, "complete row binding")
    require(row["goal"] == ref["goal"].tolist()
            and kernel.source.root_hash(kernel.source.vector(row["root_state"]),
                                        kernel.source.vector(row["goal"]))
            == ref["root_sha256"] == plan["root_sha256"], "root and goal")
    ev = [e for e in events if e.get("branch") == i]
    warm = [e for e in ev if e["kind"] == "warmup_return"]
    warm_actions = kernel.moving.warmup_commands()
    require([e["step"] for e in warm] == list(range(1, 11))
            and all(e["raw_float32_bits"] == warm_actions[j].tobytes().hex()
                    for j, e in enumerate(warm)), "warmup commands")
    transitions = [e for e in ev if e["kind"] == "returned_transition" and e["phase"] == "plan"]
    require(len(transitions) == row["own_attempted"] == row["own_returned"]
            and [e["step"] for e in transitions] == list(range(1, len(transitions) + 1))
            and 0 < len(transitions) <= 20, "transition census")
    commands = batch.branch.raw20(plan["commands"])
    for j, e in enumerate(transitions):
        metric = kernel.metric(row["goal"], e["state"])
        require(e["command"] == plan["commands"][j]
                and e["command_bits"] == e["actual_bits"] == commands[j].tobytes().hex()
                and e["returned_goal_state"] == row["goal"]
                and e["returned_goal_rgb_sha256"] == kernel.moving.image_sha(ref["goal_rgb"])
                and type(e["terminated"]) is bool and type(e["truncated"]) is bool
                and e["terminated"] is metric["native_success"]
                and e["reward"] == -metric["native_distance"]
                and (j == len(transitions) - 1 or not (e["terminated"] or e["truncated"])),
                "transition binding")
    last = transitions[-1]
    stopped = last["terminated"] or last["truncated"]
    require(row["final_state"] == last["state"] and row["terminated"] == last["terminated"]
            and row["truncated"] == last["truncated"]
            and row["initial_metrics"] == kernel.metric(row["goal"], row["root_state"])
            and row["final_metrics"] == kernel.metric(row["goal"], row["final_state"])
            and row["native_success"] == row["final_metrics"]["native_success"]
            and row["full_horizon"] == (len(transitions) == 20 and not stopped)
            and ((row["status"] == "horizon_completed" and len(transitions) == 20 and not stopped)
                 or (row["status"] == "native_success" and last["terminated"] and not last["truncated"])
                 or (row["status"] == "native_stopped" and last["truncated"])), "endpoint semantics")
    images = [e["receipt"] for e in ev if e["kind"] == "fresh_observation"]
    guards = [e for e in ev if e["kind"] == "guard_receipt"]
    axis = [("warmup", j) for j in range(11)] + [("plan", j) for j in range(len(transitions) + 1)]
    require([(v["phase"], v["step"]) for v in images] == axis
            and [(g["phase"], g["step"]) for g in guards] == axis
            and len(images) == row["PNG"]
            and all(g["accepted"] is True and g["render_preserved_dynamics"] is True for g in guards),
            "frame axis")
    for image in images:
        frame = frame_reader(image["png"], image["file_sha256"])
        require(frame.dtype == np.uint8 and frame.shape == (224, 224, 3)
                and kernel.moving.image_sha(frame) == image["pixel_sha256"]
                and image["state"] == image["observation_state"] == image["snapshot"]["state"],
                "frame identity")
    for j, step in enumerate((0, 5, 10)):
        image = images[step]
        require(image["pixel_sha256"] == kernel.moving.image_sha(ref["history"][j])
                and image["state"] == ref["states"][j].tolist()
                and image["snapshot"] == ref["dynamics"][j], "warmup source")
    require(images[11]["state"] == images[10]["state"]
            and images[11]["snapshot"] == images[10]["snapshot"]
            and images[11]["pixel_sha256"] == images[10]["pixel_sha256"], "plan root continuity")
    require(all(e["state"] == image["state"] for e, image in zip(transitions, images[12:], strict=True)),
            "transition frame time")
    require(row["checkpoints"] == [v for v in images
            if v["phase"] == "plan" and v["step"] in (0, 5, 10, 15, 20)], "checkpoints")


def consume(directory, report_sha, prepare_sha, index):
    directory = Path(directory)
    require(host.sha(directory / "report.json") == report_sha
            and host.sha(directory / "prepare.json") == prepare_sha, "external hashes")
    expected, data = source.reconstruct(index)
    require(host.helper.read(directory / "prepare.json") == expected, "prepare reconstruction")
    report = host.helper.read(directory / "report.json")
    require(host.sha(directory / "journal.jsonl") == report["journal_sha256"], "journal hash")
    events = kernel.journal(directory / "journal.jsonl", report["journal_chain_tail"])
    failed = guard.allowed_partial(report, events)
    require(events[0]["kind"] == "run_started" and events[0]["prepare"] == expected
            and events[-1]["kind"] == "run_finished"
            and events[-1]["counts"] == report["counts"]
            and [e["receipt"] for e in events if e["kind"] == "branch_closed"] == report["rows"],
            "journal boundaries")
    counts = {k: {"attempted": 0, "completed": 0} for k in expected["contract"]["caps"]}
    for event in events:
        if event["kind"] in ("operation_attempted", "operation_completed"):
            require(event["resource"] in counts, "unknown resource")
            counts[event["resource"]]["attempted" if event["kind"] == "operation_attempted"
                                      else "completed"] += 1
    require(counts == report["counts"], "operation census")

    def frame_reader(name, digest):
        path = directory / name
        require(Path(name).name == name and host.sha(path) == digest, "frame hash")
        with Image.open(BytesIO(path.read_bytes())) as image:
            return np.asarray(image.convert("RGB")).copy()

    for i, (row, plan) in enumerate(zip(report["rows"], expected["plans"], strict=True)):
        arm = f"d{plan['decision_index']}_c{plan['candidate']}_{plan['variant']}"
        require(row["branch_index"] == i and row["selected_plan"] == {**plan, "arm": arm},
                "row order and manifest identity")
        if row["status"] != "failed":
            validate_complete(row, plan, data["references"][plan["root_slot"]], events, frame_reader)
    pairs = pair.pair_rows(report)
    require(all(host.sha(path) == digest for path, digest in expected["bindings"].items())
            and host.sha(directory / "report.json") == report_sha, "source drift")
    return {"schema": "p3_raw_h3_partial_pair_20261031",
            "status": "complete_rows_checked_guard_rows_censored", "batch_index": index,
            "report_sha256": report_sha, "prepare_sha256": prepare_sha,
            "completed_rows": 6 - failed, "guard_censored_rows": failed, "pairs": pairs,
            "failed_branch_full_trajectory_verified": False,
            "independent_confirmation": False, "equal_budget_certified": False, "P3_admission": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", required=True)
    ap.add_argument("--report-sha", required=True)
    ap.add_argument("--prepare-sha", required=True)
    ap.add_argument("--batch", required=True, type=int)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    output = Path(args.output)
    require(output.is_absolute() and output.parent == source.ROOT / "results"
            and output.name.startswith("p3_raw_h3_partial_pair_20261031_")
            and not output.exists() and not output.is_symlink(), "exclusive output")
    result = consume(args.directory, args.report_sha, args.prepare_sha, args.batch)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"sha256": hashlib.sha256(raw).hexdigest(), "status": result["status"]}))


if __name__ == "__main__":
    main()
