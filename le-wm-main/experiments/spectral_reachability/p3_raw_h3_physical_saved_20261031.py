"""Validate a completed H3 batch's saved events, source roots and frames."""
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from . import p3_raw_h3_physical_batch_20261031 as batch

require, kernel, host = batch.require, batch.kernel, batch.host


def validate(p, report, events, references, frame_reader):
    require(p["schema"] == batch.NAME + ".prepare" and p["contract"] == batch.contract(),
            "prepare contract")
    batch.validate_plans(p["plans"])
    require(report["schema"] == batch.NAME + ".report" and report["status"] == "recorded_not_admitted"
            and report["errors"] == [] and report["fixed_denominator"] == 6
            and report["scientific_pass"] is False and report["new_neural_calls"] == 0,
            "report terminal")
    require(events[0]["kind"] == "run_started" and events[0]["prepare"] == p
            and events[-1]["kind"] == "run_finished" and events[-1]["status"] == report["status"]
            and events[-1]["counts"] == report["counts"], "journal boundaries")
    rows = report["rows"]
    require(len(rows) == 6 and [e["receipt"] for e in events if e["kind"] == "branch_closed"] == rows,
            "complete roster")
    counts = {k: {"attempted": 0, "completed": 0} for k in p["contract"]["caps"]}
    for e in events:
        if e["kind"] in ("operation_attempted", "operation_completed"):
            require(e["resource"] in counts, "known operation")
            counts[e["resource"]]["attempted" if e["kind"] == "operation_attempted" else "completed"] += 1
    require(counts == report["counts"] and all(v["attempted"] == v["completed"] for v in counts.values()),
            "operation census")
    steps = sum(r["warmup_returned"] + r["own_returned"] for r in rows)
    expected = dict(environment_construct=6, environment_close=6, reset=12, set_state=24,
                    goal_render=12, goal_bind=6, public_step=steps, physics_step=24 + 10 * steps,
                    explicit_render=sum(r["PNG"] for r in rows))
    require(all(counts[k]["completed"] == v <= p["contract"]["caps"][k] for k, v in expected.items()),
            "resource census")
    frames, summaries = [], []
    for i, (row, plan) in enumerate(zip(rows, p["plans"], strict=True)):
        arm = f"d{plan['decision_index']}_c{plan['candidate']}_{plan['variant']}"
        require(row["branch_index"] == i and row["selected_plan"] == {**plan, "arm": arm}
                and row["root_slot"] == plan["root_slot"] and row["arm"] == arm
                and row["selection_digest"] == plan["selection_digest"] and row["initial_exact"] is True
                and row["first_error"] is None and not row["secondary_errors"]
                and row["warmup_returned"] == 10, "branch binding")
        ref = references[plan["root_slot"]]
        require(row["goal"] == ref["goal"].tolist()
                and kernel.source.root_hash(kernel.source.vector(row["root_state"]),
                                             kernel.source.vector(row["goal"])) == ref["root_sha256"]
                == plan["root_sha256"], "root and goal identity")
        ev = [e for e in events if e.get("branch") == i]
        warm = [e for e in ev if e["kind"] == "warmup_return"]
        warm_actions = kernel.moving.warmup_commands()
        require([e["step"] for e in warm] == list(range(1, 11))
                and all(e["raw_float32_bits"] == warm_actions[j].tobytes().hex()
                        for j, e in enumerate(warm)), "warmup action binding")
        transitions = [e for e in ev if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        require(len(transitions) == row["own_attempted"] == row["own_returned"]
                and [e["step"] for e in transitions] == list(range(1, 1 + len(transitions)))
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
                    and e["reward"] == -metric["native_distance"], "transition action goal or metric")
            require(j == len(transitions) - 1 or not (e["terminated"] or e["truncated"]),
                    "no actions after stop")
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
                     or (row["status"] == "native_stopped" and last["truncated"])),
                "stop and endpoint semantics")
        images = [e["receipt"] for e in ev if e["kind"] == "fresh_observation"]
        guards = [e for e in ev if e["kind"] == "guard_receipt"]
        expected_axis = [("warmup", j) for j in range(11)] + [("plan", j) for j in range(len(transitions) + 1)]
        require([(v["phase"], v["step"]) for v in images] == expected_axis
                and len(images) == len(guards) == row["PNG"]
                and all(g["accepted"] is True and g["render_preserved_dynamics"] is True for g in guards),
                "frame axis and guards")
        for image in images:
            frame = frame_reader(image["png"], image["file_sha256"])
            require(frame.dtype == np.uint8 and frame.shape == (224, 224, 3)
                    and kernel.moving.image_sha(frame) == image["pixel_sha256"]
                    and image["state"] == image["observation_state"] == image["snapshot"]["state"],
                    "frame identity")
            frames.append(image["png"])
        for j, step in enumerate((0, 5, 10)):
            image = images[step]
            require(image["pixel_sha256"] == kernel.moving.image_sha(ref["history"][j])
                    and image["state"] == ref["states"][j].tolist()
                    and image["snapshot"] == ref["dynamics"][j], "warmup source identity")
        for e, image in zip(transitions, images[12:], strict=True):
            require(e["state"] == image["state"], "transition frame same time")
        checkpoints = [v for v in images if v["phase"] == "plan" and v["step"] in (0, 5, 10, 15, 20)]
        require(row["checkpoints"] == checkpoints, "checkpoint roster")
        summaries.append({"branch": i, "status": row["status"], "returned": len(transitions),
                          "endpoint": row["final_metrics"] if row["full_horizon"] else None,
                          "last_metrics": row["final_metrics"]})
    require(len(frames) == len(set(frames)), "unique frames")
    return {"status": "saved_semantics_checked_not_scientifically_admitted", "slots": summaries,
            "frames": frames, "counts": counts, "scientific_pass": False,
            "independent_reexecution": False}


def consume(directory, report_sha, prepare_sha, expected_prepare, references):
    directory = Path(directory)
    require(host.sha(directory / "report.json") == report_sha
            and host.sha(directory / "prepare.json") == prepare_sha, "external hashes")
    p = host.helper.read(directory / "prepare.json")
    require(p == expected_prepare and all(host.sha(path) == digest
                                          for path, digest in p["bindings"].items()),
            "independent prepare and source binding")
    report = host.helper.read(directory / "report.json")
    require(host.sha(directory / "journal.jsonl") == report["journal_sha256"], "journal hash")
    events = kernel.journal(directory / "journal.jsonl", report["journal_chain_tail"])

    def frame_reader(name, expected):
        require(Path(name).name == name and host.sha(directory / name) == expected, "frame path and hash")
        with Image.open(BytesIO((directory / name).read_bytes())) as image:
            return np.asarray(image.convert("RGB")).copy()
    result = validate(p, report, events, references, frame_reader)
    require({x.name for x in directory.iterdir()} == set(result["frames"]) |
            {"prepare.json", "report.json", "journal.jsonl"}, "artifact census")
    require(all(host.sha(path) == digest for path, digest in p["bindings"].items()),
            "source drift after consume")
    return result
