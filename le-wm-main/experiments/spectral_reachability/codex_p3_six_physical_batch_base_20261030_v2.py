"""Independent saved six-slot checks; no environment or neural execution."""
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image

from experiments.spectral_reachability import codex_p3_six_physical_host_20261030_v2 as host
from experiments.spectral_reachability import codex_p3_six_physical_saved_20261030_v1 as saved

require, kernel = host.require, host.kernel


def validate(p, report, events, frame_reader):
    require(report["schema"] == host.NAME + ".report" and report["status"] == "recorded_not_admitted"
            and report["errors"] == [] and report["fixed_denominator"] == 6 and
            report["scientific_pass"] is False and report["new_neural_calls"] == 0, "report terminal")
    require(events[0]["kind"] == "run_started" and events[0]["prepare"] == p and
            events[-1]["kind"] == "run_finished" and events[-1]["status"] == report["status"]
            and events[-1]["counts"] == report["counts"], "journal boundaries")
    rows = report["rows"]
    require(len(rows) == 6 and [e["receipt"] for e in events if e["kind"] == "branch_closed"] == rows,
            "complete roster/terminal")
    counts = {k: {"attempted": 0, "completed": 0} for k in p["contract"]["caps"]}
    for event in events:
        if event["kind"] in ("operation_attempted", "operation_completed"):
            kind = event["resource"]
            require(kind in counts, "known operation")
            counts[kind]["attempted" if event["kind"] == "operation_attempted" else "completed"] += 1
    require(counts == report["counts"] and all(x["attempted"] == x["completed"] for x in counts.values()),
            "actual resource attempt/return")
    steps = sum(r["warmup_returned"] + r["own_returned"] for r in rows)
    expected = dict(environment_construct=6, environment_close=6, reset=12, set_state=24,
                    goal_render=12, goal_bind=6, public_step=steps, physics_step=24 + 10 * steps,
                    explicit_render=sum(r["PNG"] for r in rows))
    require(all(counts[k]["completed"] == v and v <= p["contract"]["caps"][k]
                for k, v in expected.items()), "resource census")
    frames = []
    summaries = []
    for i, (row, plan) in enumerate(zip(rows, p["plans"], strict=True)):
        require(row["branch_index"] == i and row["selected_plan"] == plan and row["root_slot"] == plan["root_slot"]
                and row["arm"] == plan["arm"] and row["initial_exact"] is True and row["first_error"] is None
                and row["secondary_errors"] == [] and row["warmup_returned"] == 10, "slot binding")
        ev = [e for e in events if e.get("branch") == i]
        acks = [e["receipt"] for e in ev if e["kind"] == "step_ack_observed"]
        require(len(acks) == 10 + row["own_returned"], "ACK census")
        warm = kernel.moving.warmup_commands()
        for j, ack in enumerate(acks[:10]):
            require(ack["ordinal"] == j and ack["sent_dtype"] == ack["actual_dtype"] == "float32"
                    and ack["sent_shape"] == ack["actual_shape"] == [2]
                    and ack["sent_hex"] == ack["actual_hex"] == warm[j].tobytes().hex()
                    and not ack["terminated"] and not ack["truncated"], "warmup ACK")
        selected = [dict(a, ordinal=j) for j, a in enumerate(acks[10:])]
        status = "horizon_completed" if row["status"] == "horizon_completed" else "native_stopped"
        summaries.append(saved.verify_slot(plan, selected, {"status": status, "returned_steps": row["own_returned"]}))
        transitions = [e for e in ev if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        require(len(transitions) == row["own_attempted"] == row["own_returned"] and
                [e["step"] for e in transitions] == list(range(1, 1 + len(transitions))), "returned transition census")
        for j, e in enumerate(transitions):
            require(e["command"] == plan["commands"][j] and e["returned_goal_state"] == row["goal"]
                    and e["terminated"] == selected[j]["terminated"] and e["truncated"] == selected[j]["truncated"],
                    "plan transition and goal")
        require(transitions and row["final_state"] == transitions[-1]["state"] and
                row["final_metrics"] == kernel.metric(row["goal"], row["final_state"]) and
                row["initial_metrics"] == kernel.metric(row["goal"], row["root_state"]) and
                row["native_success"] == row["final_metrics"]["native_success"] and
                row["full_horizon"] == (row["own_returned"] == 10), "saved outcome metric")
        images = [e["receipt"] for e in ev if e["kind"] == "fresh_observation"]
        guards = [e for e in ev if e["kind"] == "guard_receipt"]
        require(len(images) == len(guards) == row["PNG"] and
                all(g["accepted"] is True and g["render_preserved_dynamics"] is True for g in guards), "frame guards")
        for image in images:
            frame = frame_reader(image["png"], image["file_sha256"])
            require(frame.dtype == np.uint8 and frame.shape == (224, 224, 3) and
                    kernel.moving.image_sha(frame) == image["pixel_sha256"] and
                    image["state"] == image["observation_state"] == image["snapshot"]["state"], "frame/identity")
            frames.append(image["png"])
    require(len(frames) == len(set(frames)), "unique frames")
    return dict(status="saved_semantics_checked_not_scientifically_admitted", slots=summaries,
                counts=counts, frames=frames, physical_certified=False, scientific_pass=False)


def consume(output, report_sha, prepare_sha):
    output = Path(output)
    require(host.sha(output / "report.json") == report_sha, "external report sha")
    p, _ = host.verify(output / "prepare.json", prepare_sha)
    report = host.helper.read(output / "report.json")
    require(host.sha(output / "journal.jsonl") == report["journal_sha256"], "journal hash")
    events = kernel.journal(output / "journal.jsonl", report["journal_chain_tail"])
    def frame_reader(name, expected):
        require(Path(name).name == name and host.sha(output / name) == expected, "frame hash/path")
        with Image.open(BytesIO((output / name).read_bytes())) as im:
            return np.asarray(im.convert("RGB")).copy()
    result = validate(p, report, events, frame_reader)
    require({f.name for f in output.iterdir()} == set(result["frames"]) |
            {"prepare.json", "report.json", "journal.jsonl"}, "exact file census")
    return result
