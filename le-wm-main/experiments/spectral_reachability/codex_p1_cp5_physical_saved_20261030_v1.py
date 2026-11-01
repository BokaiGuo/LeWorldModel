"""Saved verification for the fixed 24-row cp5 physical exploration."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp5_physical_24_20261030_v1 as run
from . import p1_backtrack_physical_consume_20261030 as saved

need = run.need


def check_row(path, anchor, item, expected_sha):
    need(run.sha(path/"report.json") == expected_sha, "row report SHA")
    report = run.h10.native.read(path/"report.json", cap=64*1024**2)
    row = report["row"]
    need(report["schema"] == run.NAME+".row" and report["status"] == "recorded_not_consumed"
         and not report["errors"] and report["raw10_sha256"] == item["raw10_sha256"]
         and row["anchor_index"] == item["index"] and row["task_id"] == item["task_id"]
         and row["candidate_role"] == item["role"] and row["root_slot"] == item["root_slot"]
         and row["condition"] == item["condition"] and row["arm"] == item["arm"]
         and row["start_verified"] is True and row["start_state"] == anchor["checkpoint"]["state"]
         and row["status"] in ("probe_completed", "native_stopped", "censored_out_of_domain")
         and not row["secondary_errors"], "row identity or completion")
    journal = path/"journal.jsonl"
    need(run.sha(journal) == report["journal_sha256"], "journal SHA")
    events = saved.events(journal, report["journal_chain_tail"])
    need(events[0]["kind"] == "run_started" and events[0]["caps"] == run.CAPS
         and events[0]["raw10_sha256"] == item["raw10_sha256"]
         and events[-1]["status"] == report["status"] and events[-1]["counts"] == report["counts"]
         and events[-1]["errors"] == [], "journal boundary")
    intents = [e["receipt"] for e in events if e["kind"] == "operation_dispatch_intent"]
    returns = sorted((e["receipt"] for e in events if e["kind"] == "operation_returned"),
                     key=lambda x: x["index"])
    need(not any(e["kind"] == "operation_failed" for e in events)
         and returns == report["receipts"] and len(intents) == len(returns)
         and [x["index"] for x in intents] == list(range(len(intents)))
         and all(dict(a, status="returned") == b for a, b in zip(intents, returns, strict=True)),
         "operation ledger")
    for kind, cap in run.CAPS.items():
        n = sum(x["resource"] == kind for x in returns)
        need(report["counts"][kind] == {"attempted": n, "completed": n} and n <= cap,
             "resource cap or count")
    c = {k: v["completed"] for k, v in report["counts"].items()}
    need(c["constructor"] == c["close"] == 1 and c["reset"] == 2 and c["set_state"] == 4
         and c["goal_bind"] == 1 and c["goal_render"] == 2 and c["goal_setter"] == 3
         and c["public_step"] == row["warmup_returned"]+row["own_returned"]
         and c["physics_step"] == 2*c["reset"]+10*c["public_step"]
         and c["explicit_render"] == row["PNG"], "resource census")
    frames = [e for e in events if e["kind"] == "fresh_observation"]
    need(len(frames) == row["PNG"] and len(frames) >= 51, "frame count")
    for actual, original in zip(frames[:51], anchor["trace"], strict=True):
        need(actual["pre"] == actual["post"] == original["pre"] == original["post"] and
             all(actual["receipt"][k] == original["receipt"][k] for k in
                 ("phase", "step", "condition", "overlay_index", "file_sha256", "pixel_sha256", "state", "snapshot")),
             "exact warm/prefix cp5 trace")
    starts = [e for e in events if e["kind"] == "suffix_start_verified"]
    need(len(starts) == 1 and all(starts[0]["checkpoint"][k] == anchor["checkpoint"][k] for k in
         ("step", "state", "snapshot", "file_sha256", "pixel_sha256")), "cp5 binding")
    transitions = [e for e in events if e["kind"] == "returned_transition"]
    suffix = np.frombuffer(bytes.fromhex(item["raw10_float32_hex"]), "<f4").reshape(10, 2)
    accepted = len(row["responses"])
    returned = accepted + (row["status"] == "censored_out_of_domain")
    need(row["warmup_returned"] == 10 and row["own_returned"] == 5+returned and returned <= 10,
         "step denominator")
    for phase, commands in (("warmup", run.h10.base.moving.warmup_commands()),
                            ("plan", np.frombuffer(bytes.fromhex(item["prefix5_float32_hex"]), "<f4").reshape(5, 2)),
                            ("probe", suffix[:returned])):
        seq = [e for e in transitions if e["phase"] == phase]
        need(len(seq) == len(commands), "transition count")
        for step, (event, action) in enumerate(zip(seq, commands, strict=True), 1):
            need(event["step"] == step and event["requested_bits"] == event["actual_bits"] == action.tobytes().hex()
                 and event["actual_dtype"] == "<f4" and
                 (phase != "probe" or event["index"] > starts[0]["index"]), "action identity/order")
    expected_frames = [(phase, step, overlay) for phase, steps in
                       (("warmup", range(11)), ("plan", range(6)), ("probe", range(1, returned+1)))
                       for step in steps for overlay in range(3)]
    if row["status"] == "censored_out_of_domain":
        expected_frames = expected_frames[:-2]
    need([(e["receipt"]["phase"], e["receipt"]["step"], e["receipt"]["overlay_index"])
          for e in frames] == expected_frames, "frame schedule")
    for event in frames:
        rec = event["receipt"]
        path_png = path/rec["png"]
        need(Path(rec["png"]).name == rec["png"] and not path_png.is_symlink()
             and run.sha(path_png) == rec["file_sha256"], "frame file")
        with Image.open(path_png) as img:
            pixels = np.asarray(img)
        need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8 and
             run.h10.base.moving.image_sha(pixels) == rec["pixel_sha256"] and event["pre"] == event["post"]
             and rec["snapshot"] == event["pre"]["snapshot"] and rec["state"] == rec["snapshot"]["state"],
             "image/state invariant")
        if rec["step"] > 0:
            matching = [x for x in transitions if x["phase"] == rec["phase"] and x["step"] == rec["step"]]
            need(len(matching) == 1 and matching[0]["state"] == rec["state"]
                 and matching[0]["snapshot"] == rec["snapshot"] and matching[0]["index"] < event["index"],
                 "frame transition")
        if not (row["status"] == "censored_out_of_domain" and event is frames[-1]):
            state = np.asarray(rec["state"])
            need(np.count_nonzero(np.all(pixels == [65, 105, 225], axis=-1)) > 0
                 and ((state[:4] >= 0) & (state[:4] <= 512)).all(), "RGB/canvas guard")
    probe_events = [e for e in transitions if e["phase"] == "probe"]
    for response, trans in zip(row["responses"], probe_events, strict=False):
        need(response["step"] == trans["step"] and response["state"] == trans["state"] and
             response["terminated"] == trans["terminated"] and response["truncated"] == trans["truncated"],
             "response identity")
    if row["status"] == "censored_out_of_domain":
        probe = {"index": 0, "name": row["probe"], "raw10_float32_hex": item["raw10_float32_hex"],
                 "raw10_sha256": item["raw10_sha256"]}
        trial = copy.deepcopy(row)
        trial.update(status="engineering_failed", error={"stage": "branch", "type": "Rejected",
                                                       "message": "fresh RGB/canvas guard"})
        with Image.open(path/frames[-1]["receipt"]["png"]) as image:
            proof = run.h10.proof.incident(events, anchor, probe, trial, np.asarray(image))
        need(row["censor"] == {k: proof[k] for k in ("anchor", "probe", "step", "action_float32_hex",
             "returned_state", "out_of_bounds", "frame_sha256", "blue_pixels")}, "censor evidence")
    else:
        need(accepted == 10 if row["status"] == "probe_completed" else
             bool(row["responses"]) and (row["responses"][-1]["terminated"] or row["responses"][-1]["truncated"]),
             "native stop/completion")
    need(all(not r["terminated"] and not r["truncated"] for r in row["responses"][:-1]), "continued after stop")
    endpoint = row["responses"][-1]["state"] if row["responses"] else None
    # Censored and native-early-stop states are observations, never completed endpoints.
    metrics = run.h10.base.previous.metric(anchor["goal"], endpoint) if endpoint is not None else None
    return {"index": item["index"], "root_slot": item["root_slot"], "condition": item["condition"],
            "arm": item["arm"], "role": item["role"], "status": row["status"], "returned_suffix": returned,
            "completed_endpoint": endpoint if row["status"] == "probe_completed" else None,
            "last_accepted_state_metrics": metrics, "counts": report["counts"]}


def consume(directory, report_sha, prepared, prepare_sha):
    p = run.verify(prepared, prepare_sha)
    directory = Path(directory)
    need(run.sha(directory/"run_report.json") == report_sha, "run SHA")
    report = run.h10.native.read(directory/"run_report.json")
    need(report["status"] == "recorded_not_consumed" and report["fixed_rows"] == 24 and
         report["prepare_sha256"] == prepare_sha and len(report["shards"]) == 24 and
         run.sha(directory/"binding.json") == report["binding_sha256"], "complete run")
    anchors = run.h10.native.read(run.h10.INPUT, cap=64*1024**2)["anchors"]
    rows = []
    for i, (anchor, item, shard) in enumerate(zip(anchors, p["roster"], report["shards"], strict=True)):
        need(shard["index"] == i and shard["status"] == "recorded_not_consumed", "ordered shards")
        rows.append(check_row(directory/f"row_{i:02d}", anchor, item, shard["report_sha256"]))
    return {"schema": run.NAME+".consume", "status": "saved_checks_passed", "fixed_rows": 24, "rows": rows,
            "P1_admission": False, "independent_confirmation": False, "os_isolation_certified": False,
            "fairness_proven": False, "known_roots": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha", required=True)
    parser.add_argument("--output", required=True)
    a = parser.parse_args()
    result = consume(a.directory, a.report_sha, a.prepared, a.prepare_sha)
    output = Path(a.output)
    need(output.is_absolute() and output.parent == run.ROOT/"results" and
         output.name.startswith("codex_p1_cp5_physical_saved_20261030_v1_"), "exclusive output")
    print(json.dumps({"sha256": run.h10._write_new(output, result), "status": result["status"]}))


if __name__ == "__main__":
    main()
