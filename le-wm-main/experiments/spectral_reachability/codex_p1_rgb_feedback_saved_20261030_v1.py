"""Fail-closed saved consumer for the 24-row two-stage exploration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_rgb_feedback_launcher_20261030_v1 as run
from . import p1_backtrack_physical_consume_20261030 as journal_reader
from . import codex_p1_h10_paired_table_20261030_main as metrics

need, sha = run.need, run.sha


def one(values, message):
    need(len(values) == 1, message)
    return values[0]


def check_stage(folder, row, event, stage_number, source_hashes, png_sha, previous_sha, packet_sha):
    report_path = folder/f"stage_{stage_number}/report.json"
    need(sha(report_path) == event["stage_report_sha256"], "stage report binding")
    report = json.loads(report_path.read_bytes())
    binding = {"rgb_sha256": png_sha, "previous_latent_sha256": previous_sha,
               "executed_packet_sha256": packet_sha}
    policy, contract = run.stage.contract_for(row, source_hashes, stage_number, binding)
    inp = json.loads((report_path.parent/"input.json").read_bytes())
    need(inp == {"index": row["index"], "stage": stage_number, "binding": binding,
                 "contract": contract, "policy": policy}
         and report["index"] == row["index"] and report["stage"] == stage_number
         and report["binding"] == binding and report["status"] == "complete" and report["error"] is None,
         "stage input and identity")
    receipt = report["planning_receipt"]
    account = run.base.consumer.consume(report_path.parent/"encoder.jsonl", report["encoder_sha256"],
              run.stage.ENC_CAPS, run.stage.WALL, report_path.parent/"planning/events.jsonl",
              receipt["sha256"], run.base.spec.digest(contract), "structured", "fixed",
              f"row_{row['index']:02d}_stage_{stage_number}")
    result = report["result"]
    need(account == result["account"] and account["complete"]
         and account["planning"]["charged_upper_bounds"] == run.stage.CAPS
         and account["encoding"]["attempted"] == run.stage.ENC_CAPS
         and result["execute_packet"] == 0 and run.base.spec.is_hash(result["latent_sha256"]), "ledger")
    raw = bytes.fromhex(result["raw_horizon_f32_hex"])
    need(len(raw) == 80 and hashlib.sha256(raw).hexdigest() == event["plan_sha256"]
         and raw[:40].hex() == event["packet_float32_hex"], "actual plan event")
    packet = np.frombuffer(raw[:40], dtype="<f4").copy().reshape(5, 2)
    run.episode.feedback.packet_bytes(packet)
    return result, packet, account


def check_row(folder, anchor, blind, shard, source_hashes, task_data, prepared_row):
    physical = folder/"physical"
    need(sha(physical/"report.json") == shard["report_sha256"], "row report SHA")
    report = json.loads((physical/"report.json").read_bytes())
    row = report["row"]
    need(report["schema"] == run.NAME+".row" and report["status"] == "recorded_not_consumed"
         and not report["errors"] and not row["secondary_errors"] and row["start_verified"]
         and row["start_state"] == anchor["checkpoint"]["state"] and row["anchor_index"] == anchor["index"]
         and row["root_slot"] == anchor["root_slot"] and row["condition"] == anchor["condition"]
         and row["candidate_role"] == anchor["role"] and row["task_id"] == anchor["task_id"]
         and row["status"] in ("probe_completed", "native_stopped", "rgb_rejected"), "row identity/status")
    path = physical/"journal.jsonl"
    need(sha(path) == report["journal_sha256"], "journal file hash")
    events = journal_reader.events(path, report["journal_chain_tail"])
    need(events[0]["kind"] == "run_started" and events[0]["caps"] == run.physical.CAPS
         and events[-1]["status"] == report["status"] and events[-1]["counts"] == report["counts"]
         and events[-1]["errors"] == [], "journal boundary")
    intents = [x["receipt"] for x in events if x["kind"] == "operation_dispatch_intent"]
    returned = sorted((x["receipt"] for x in events if x["kind"] == "operation_returned"), key=lambda x: x["index"])
    need(not any(x["kind"] == "operation_failed" for x in events) and returned == report["receipts"]
         and len(intents) == len(returned) and [x["index"] for x in intents] == list(range(len(intents)))
         and all(dict(a, status="returned") == b for a, b in zip(intents, returned, strict=True)), "operation ledger")
    for kind, cap in run.physical.CAPS.items():
        n = sum(x["resource"] == kind for x in returned)
        need(report["counts"][kind] == {"attempted": n, "completed": n} and n <= cap, "physical resource")
    count = {k: v["completed"] for k, v in report["counts"].items()}
    need(count["constructor"] == count["close"] == 1 and count["reset"] == 2 and count["set_state"] == 4
         and count["goal_bind"] == 1 and count["goal_render"] == 2 and count["goal_setter"] == 3
         and count["public_step"] == row["warmup_returned"]+row["own_returned"]
         and count["physics_step"] == 2*count["reset"]+10*count["public_step"]
         and count["explicit_render"] == row["PNG"], "physical census")
    frames = [e for e in events if e["kind"] == "fresh_observation"]
    need(len(frames) == row["PNG"] and len(frames) >= 51, "image census")
    for actual, original in zip(frames[:51], anchor["trace"], strict=True):
        need(actual["pre"] == actual["post"] == original["pre"] == original["post"]
             and all(actual["receipt"][k] == original["receipt"][k] for k in
                     ("phase", "step", "condition", "overlay_index", "file_sha256", "pixel_sha256", "state", "snapshot")),
             "frozen prefix trace")
    start = one([e for e in events if e["kind"] == "suffix_start_verified"], "start event")
    need(all(start["checkpoint"][k] == anchor["checkpoint"][k]
             for k in ("step", "state", "snapshot", "file_sha256", "pixel_sha256")), "cp5")
    transitions = [e for e in events if e["kind"] == "returned_transition"]
    for phase, actions in (("warmup", run.h10.base.moving.warmup_commands()),
                           ("plan", np.frombuffer(bytes.fromhex(anchor["prefix5_float32_hex"]),
                                                  "<f4").reshape(5, 2))):
        seq = [e for e in transitions if e["phase"] == phase]
        need(len(seq) == len(actions), "prefix transitions")
        for step, (tr, cmd) in enumerate(zip(seq, actions, strict=True), 1):
            need(tr["step"] == step and tr["requested_bits"] == tr["actual_bits"] == cmd.tobytes().hex()
                 and tr["actual_dtype"] == "<f4" and not tr["terminated"] and not tr["truncated"], "prefix actions")
    probe = [e for e in transitions if e["phase"] == "probe"]
    n = len(probe)
    need(1 <= n <= 10 and len(row["responses"]) == n and row["own_returned"] == 5+n, "suffix count")
    need(all(not x["terminated"] and not x["truncated"] for x in probe[:-1]), "continued after stop")
    if row["status"] == "native_stopped":
        need(probe[-1]["terminated"] or probe[-1]["truncated"], "stop proof")
    elif row["status"] == "rgb_rejected":
        need(n == 5 and not probe[-1]["terminated"] and not probe[-1]["truncated"], "RGB rejection timing")
    else:
        need(n == 10 and not probe[-1]["terminated"] and not probe[-1]["truncated"], "full horizon")
    for response, tr in zip(row["responses"], probe, strict=True):
        need(response["step"] == tr["step"] and response["state"] == tr["state"]
             and response["terminated"] == tr["terminated"] and response["truncated"] == tr["truncated"],
             "response binding")
    plans = [e for e in events if e["kind"] == "phase_plan_returned"]
    expected_stages = 2 if n > 5 or row["status"] == "rgb_rejected" else 1
    need([e["phase"] for e in plans] == list(range(1, expected_stages+1)), "phase plan denominator")
    cp5_frame = one([e for e in frames if e["receipt"]["phase"] == "plan" and e["receipt"]["step"] == 5
                     and e["receipt"]["condition"] == anchor["condition"]
                     and e["receipt"]["overlay_index"] == (1 if anchor["condition"] == "on" else 0)], "cp5 RGB")
    empty = hashlib.sha256(b"").hexdigest()
    first, packet1, account1 = check_stage(folder, blind, plans[0], 1, source_hashes,
                                           cp5_frame["receipt"]["file_sha256"], empty, empty)
    need(start["index"] < plans[0]["index"] < probe[0]["index"], "phase1 order")
    accounts = [account1]
    if expected_stages == 2:
        feedback_event = one([e for e in events if e["kind"] == "feedback_verified"], "feedback event")
        prefix = [e for e in events if e["index"] <= feedback_event["last_render_index"]]
        bound = run.feedback_journal.from_journal(prefix, physical, anchor["index"], anchor["condition"],
                                                 packet1, after_index=start["index"])
        need(bound["png_sha256"] == feedback_event["png_sha256"]
             and bound["pixel_sha256"] == feedback_event["pixel_sha256"]
             and feedback_event["executed_packet_sha256"] == hashlib.sha256(packet1.tobytes()).hexdigest()
             and bound["last_render_index"] < feedback_event["index"] < plans[1]["index"],
             "phase2 feedback ordering")
        _, model_packet2, account2 = check_stage(folder, blind, plans[1], 2, source_hashes,
              bound["png_sha256"], first["latent_sha256"], hashlib.sha256(packet1.tobytes()).hexdigest())
        accounts.append(account2)
        decision_event = one([e for e in events if e["kind"] == "rgb_decision"], "RGB decision event")
        need(plans[1]["index"] < decision_event["index"]
             and (n == 5 or decision_event["index"] < probe[5]["index"]), "RGB decision order")
        goal_rgb = np.asarray(task_data["roots"][anchor["root_slot"]]["packets"][anchor["condition"]]["goal_rgb"])
        goal_sha = hashlib.sha256(np.ascontiguousarray(goal_rgb).tobytes()).hexdigest()
        need(goal_sha == prepared_row["goal_rgb_pixel_sha256"], "fixed goal source")
        receipt = run.rgb_bridge.verify(decision_event["receipt"], prefix, physical, anchor["index"],
                                        anchor["condition"], packet1, start["index"], goal_rgb,
                                        goal_sha, model_packet2)
        outcome = run.rgb_bridge.verify_dispatch(receipt, probe[5:])
        need(outcome == row["status"], "RGB decision versus status")
        packet2 = None if receipt["executed_packet_f32_hex"] is None else np.frombuffer(
            bytes.fromhex(receipt["executed_packet_f32_hex"]), dtype="<f4").copy().reshape(5, 2)
    else:
        need(not any(e["kind"] == "feedback_verified" for e in events) and not (folder/"stage_2").exists(),
             "no phase2 after stop")
        need(not any(e["kind"] == "rgb_decision" for e in events), "no RGB decision after stop")
        receipt = None
        packet2 = None
    for i, tr in enumerate(probe):
        command = packet1[i] if i < 5 else packet2[i-5]
        need(tr["step"] == i+1 and tr["actual_dtype"] == "<f4"
             and tr["actual_bits"] == tr["requested_bits"] == command.tobytes().hex(), "executed plan action")
    for event in frames:
        rec = event["receipt"]
        name = rec["png"]
        image_path = physical/name
        need(Path(name).name == name and not image_path.is_symlink() and sha(image_path) == rec["file_sha256"],
             "image file")
        pixels = run.episode.feedback.rgb_pixels(image_path.read_bytes())
        need(hashlib.sha256(pixels.tobytes()).hexdigest() == rec["pixel_sha256"]
             and event["pre"] == event["post"] and rec["snapshot"] == event["pre"]["snapshot"]
             and rec["state"] == rec["snapshot"]["state"], "image content and state")
        if rec["step"]:
            transition = one([x for x in transitions if x["phase"] == rec["phase"]
                              and x["step"] == rec["step"]], "frame transition")
            need(transition["index"] < event["index"] and transition["state"] == rec["state"]
                 and transition["snapshot"] == rec["snapshot"], "fresh state")
        state = np.asarray(rec["state"], dtype=np.float64)
        need(state.shape == (7,) and np.isfinite(state).all() and ((state[:4] >= 0) & (state[:4] <= 512)).all()
             and np.any(np.all(pixels == [65, 105, 225], axis=-1)), "RGB/canvas guard")
    schedule = [(phase, step, overlay) for phase, steps in
                (("warmup", range(11)), ("plan", range(6)), ("probe", range(1, n+1)))
                for step in steps for overlay in range(3)]
    need([(x["receipt"]["phase"], x["receipt"]["step"], x["receipt"]["overlay_index"])
          for x in frames] == schedule, "complete frame schedule")
    endpoint = probe[-1]["state"] if n == 10 else None
    return {"index": anchor["index"], "logical_pair": anchor["logical_pair"], "root_slot": anchor["root_slot"],
            "condition": anchor["condition"], "arm": anchor["arm"], "role": anchor["role"],
            "status": row["status"], "accepted_suffix_steps": n, "early_stop": n < 10,
            "native_stop_at_horizon": row["status"] == "native_stopped" and n == 10,
            "endpoint_metrics": metrics.physical_metrics(endpoint, anchor["goal"]),
            "last_observed_metrics": metrics.physical_metrics(probe[-1]["state"], anchor["goal"]),
            "physical_resources": report["counts"], "stage_resources": accounts,
            "rgb_decision": None if receipt is None else receipt["decision"],
            "rgb_rejection_reason": None if receipt is None else receipt["reason"],
            "suffix_type": "independently_optimized_two_stage_rgb_feedback"}


def consume(directory, report_sha, prepared, prepare_sha):
    p = run.verify(prepared, prepare_sha)
    directory = Path(directory)
    need(sha(directory/"run_report.json") == report_sha, "run SHA")
    report = json.loads((directory/"run_report.json").read_bytes())
    need(report["status"] == "recorded_not_consumed" and report["prepare_sha256"] == prepare_sha
         and report["fixed_rows"] == 24 and len(report["shards"]) == 24 and not report["not_started_indices"]
         and report["model_before"] == report["model_after"] == run.base.MODEL_HASH, "complete run")
    need(json.loads((directory/"binding.json").read_bytes()) == {"prepare_sha256": prepare_sha, "rows": 24},
         "binding")
    anchors = run.h10.native.read(run.h10.INPUT, cap=64*1024**2)["anchors"]
    rows = json.loads(run.base.BUNDLE.read_bytes())["rows"]
    source_report = run.h10.native.read(run.h10.old.prep.DIRECTORY/"report.json", cap=64*1024**2)
    _, task_data = run.h10.runner.physical.verify(source_report["execution_binding"]["prepared"],
                                                   run.h10.old.prep.PREP_SHA)
    results = []
    for i, (anchor, row, shard) in enumerate(zip(anchors, rows, report["shards"], strict=True)):
        need(shard["index"] == i and shard["status"] == "recorded_not_consumed", "roster order")
        results.append(check_row(directory/f"row_{i:02d}", anchor, row, shard, p["sources"], task_data,
                                 p["roster"][i]))
    need(sha(directory/"run_report.json") == report_sha, "run drift")
    return {"schema": run.NAME+".saved", "status": "saved_checks_passed", "rows": results, "fixed_rows": 24,
            "known_roots": True, "independent_confirmation": False, "os_isolation_certified": False,
            "equal_budget_certified": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    for field in ("directory", "report-sha", "prepared", "prepare-sha", "output"):
        parser.add_argument("--"+field, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.parent == run.ROOT/"results"
         and out.name.startswith("codex_p1_rgb_feedback_saved_20261030_v1_"), "exclusive output")
    result = consume(args.directory, args.report_sha, args.prepared, args.prepare_sha)
    print(json.dumps({"sha256": run.h10._write_new(out, result), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
