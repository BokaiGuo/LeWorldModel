"""Saved-only consumer of all 336 post-MAIN02 exploratory branches."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p1_matched_censored_runner_20261030_codex as run
from experiments.spectral_reachability import p1_backtrack_physical_consume_20261030 as saved

require = run.require


def _one(rows, label):
    require(len(rows) == 1, "unique " + label)
    return rows[0]


def _image(directory, receipt):
    name = receipt["png"]
    require(type(name) is str and Path(name).name == name and name not in ("", ".", ".."), "flat PNG")
    path = directory / name
    require(path.resolve() == path and path.is_file() and not path.is_symlink()
            and run.runner.physical.sha(path) == receipt["file_sha256"], "PNG bytes")
    with Image.open(path) as image:
        pixels = np.asarray(image)
    require(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
            and run.base.moving.image_sha(pixels) == receipt["pixel_sha256"], "PNG pixels")
    return pixels


def check_shard(directory, anchor, probes, report_sha256):
    directory = Path(directory)
    require(directory.is_absolute() and directory.resolve() == directory and directory.is_dir(), "shard path")
    path = directory / "report.json"
    require(run.runner.physical.sha(path) == report_sha256, "external shard report SHA")
    report = run.native.read(path, cap=64*1024**2)
    require(report["schema"] == run.NAME + ".shard.v1" and report["status"] == "recorded_not_consumed"
            and report["anchor"] == anchor["index"] and report["fixed_rows"] == len(report["rows"]) == 21
            and report["P1_admission"] is False and not report["errors"], "complete shard")
    journal_path = directory / "journal.jsonl"
    require(run.runner.physical.sha(journal_path) == report["journal_sha256"], "journal SHA")
    events = saved.events(journal_path, report["journal_chain_tail"])
    require(events[0]["kind"] == "run_started" and events[0]["anchor"] == anchor["index"]
            and events[0]["fixed_rows"] == 21 and events[0]["caps"] == run.runner.CAPS
            and events[-1]["status"] == report["status"] and events[-1]["counts"] == report["counts"]
            and events[-1]["errors"] == [], "run boundary")
    intents = [e["receipt"] for e in events if e["kind"] == "operation_dispatch_intent"]
    returns = sorted((e["receipt"] for e in events if e["kind"] == "operation_returned"),
                     key=lambda x: x["index"])
    require(not any(e["kind"] == "operation_failed" for e in events)
            and [e["index"] for e in intents] == list(range(len(intents)))
            and returns == report["receipts"] and len(intents) == len(returns)
            and all(dict(a, status="returned") == b and a["status"] == "pending"
                    for a, b in zip(intents, returns, strict=True)), "operation ledger")
    counts = report["counts"]
    require(set(counts) == set(run.runner.CAPS), "resource roster")
    for kind, cap in run.runner.CAPS.items():
        n = sum(r["resource"] == kind for r in returns)
        require(counts[kind] == dict(attempted=n, completed=n) and n <= cap, "resource count")
    c = {k: v["completed"] for k, v in counts.items()}
    require(c["constructor"] == c["close"] == 21 and c["reset"] == 42
            and c["set_state"] == 2*c["reset"] and c["physics_step"] == 2*c["reset"]+10*c["public_step"]
            and c["goal_render"] == c["reset"] and c["goal_setter"] == c["reset"]+c["goal_bind"]
            and c["goal_bind"] == 21
            and c["explicit_render"] == sum(row["PNG"] for row in report["rows"])
            and c["public_step"] == sum(row["warmup_returned"]+row["own_returned"] for row in report["rows"]),
            "resource census")
    endpoints, statuses = [], []
    warm = run.base.moving.warmup_commands()
    prefix = np.frombuffer(bytes.fromhex(anchor["prefix5_float32_hex"]), dtype="<f4").reshape(5, 2)
    closures = [e for e in events if e["kind"] == "branch_closed"]
    require([e["branch"] for e in closures] == list(range(21)), "ordered branch closure")
    for row, probe, closed in zip(report["rows"], probes, closures, strict=True):
        index = probe["index"]
        status = row["status"]
        require(row["index"] == index and row["anchor_index"] == anchor["index"]
                and row["task_id"] == anchor["task_id"] and row["root_slot"] == anchor["root_slot"]
                and row["condition"] == anchor["condition"] and row["arm"] == anchor["arm"]
                and row["candidate_role"] == anchor["role"] and row["probe"] == probe["name"]
                and row["start_verified"] is True and row["start_state"] == anchor["checkpoint"]["state"]
                and status in ("probe_completed", "native_stopped", "censored_out_of_domain")
                and closed["status"] == status and not row["secondary_errors"] and "error" not in row
                and ("censor" in row) == (status == "censored_out_of_domain"), "branch identity")
        scope = [e for e in events if e.get("branch") == index]
        require(all(e.get("anchor_index") == anchor["index"] for e in scope
                    if e["kind"] in ("returned_transition", "fresh_observation", "suffix_start_verified",
                                     "censored_out_of_domain")), "event anchor")
        frames = [e for e in scope if e["kind"] == "fresh_observation"]
        require(len(frames) == row["PNG"] and len(frames) >= 51, "frame denominator")
        for actual, reference in zip(frames[:51], anchor["trace"], strict=True):
            require(actual["pre"] == actual["post"] == reference["pre"] == reference["post"]
                    and all(actual["receipt"][k] == reference["receipt"][k] for k in
                            ("phase", "step", "condition", "overlay_index", "file_sha256",
                             "pixel_sha256", "state", "snapshot")), "exact anchor trace")
        transitions = [e for e in scope if e["kind"] == "returned_transition"]
        start = _one([e for e in scope if e["kind"] == "suffix_start_verified"], "suffix start")
        require(all(start["checkpoint"][k] == anchor["checkpoint"][k] for k in
                    ("step", "state", "snapshot", "file_sha256", "pixel_sha256")), "start checkpoint")
        suffix = np.frombuffer(bytes.fromhex(probe["raw5_float32_hex"]), dtype="<f4").reshape(5, 2)
        accepted_count = len(row["responses"])
        returned_count = accepted_count + (status == "censored_out_of_domain")
        require(0 <= accepted_count <= 5 and returned_count <= 5 and row["warmup_returned"] == 10
                and row["own_returned"] == 5 + returned_count, "returned denominator")
        for phase, commands in (("warmup", warm), ("plan", prefix), ("probe", suffix[:returned_count])):
            phase_events = [e for e in transitions if e["phase"] == phase]
            require(len(phase_events) == len(commands), "transition denominator")
            for step, (event, command) in enumerate(zip(phase_events, commands, strict=True), 1):
                require(event["step"] == step and event["requested_bits"] == event["actual_bits"]
                        == command.tobytes().hex() and event["actual_dtype"] == "<f4", "literal action")
                if phase != "probe":
                    require(event["terminated"] is False and event["truncated"] is False, "early prefix stop")
                ops = [e for e in events if e["kind"] == "operation_returned"
                       and e["receipt"].get("resource") == "public_step"
                       and e["receipt"].get("branch") == index and e["receipt"].get("phase") == phase
                       and e["receipt"].get("step") == step]
                op = _one(ops, "public step return")
                require(op["receipt"]["raw_bits"] == command.tobytes().hex()
                        and op["index"] < event["index"] < closed["index"]
                        and (phase != "probe" or start["index"] < op["index"]), "action order")
        expected_frames = [(phase, step, overlay) for phase, steps in
                           (("warmup", range(11)), ("plan", range(6)),
                            ("probe", range(1, returned_count+1)))
                           for step in steps for overlay in range(3)]
        if status == "censored_out_of_domain":
            expected_frames = expected_frames[:-2]
        require([(f["receipt"]["phase"], f["receipt"]["step"], f["receipt"]["overlay_index"])
                 for f in frames] == expected_frames, "exact frame schedule")
        for event in frames:
            rec = event["receipt"]
            pixels = _image(directory, rec)
            state = np.asarray(rec["state"], dtype=np.float64)
            require(state.shape == (7,) and np.isfinite(state).all()
                    and rec["snapshot"] == event["pre"]["snapshot"] == event["post"]["snapshot"]
                    and rec["state"] == rec["snapshot"]["state"] and event["pre"] == event["post"],
                    "render invariant")
            if rec["step"] > 0:
                trans = _one([t for t in transitions if t["phase"] == rec["phase"]
                              and t["step"] == rec["step"]], "frame transition")
                require(trans["index"] < event["index"] and trans["state"] == rec["state"]
                        and trans["snapshot"] == rec["snapshot"], "frame/transition state")
            is_last_censor = status == "censored_out_of_domain" and event is frames[-1]
            if not is_last_censor:
                require(np.count_nonzero(np.all(pixels == [65, 105, 225], axis=-1)) > 0
                        and ((state[:4] >= 0) & (state[:4] <= 512)).all(), "accepted RGB/canvas guard")
        for response, trans in zip(row["responses"], [e for e in transitions if e["phase"] == "probe"], strict=False):
            delta = np.asarray(trans["state"]) - np.asarray(row["start_state"])
            require(response["step"] == trans["step"] and response["state"] == trans["state"]
                    and response["delta_native7"] == delta.tolist() and response["terminated"] == trans["terminated"]
                    and response["truncated"] == trans["truncated"]
                    and response["delta_angle_wrapped"] == float(np.arctan2(np.sin(delta[4]), np.cos(delta[4]))),
                    "response transition")
        selected = [f["receipt"] for f in frames if f["receipt"]["overlay_index"] ==
                    (1 if row["condition"] == "on" else 0)]
        if status == "censored_out_of_domain" and row["condition"] == "off":
            selected.pop()
        require(row["trajectory"] == selected, "selected trajectory")
        checkpoints = [{k: r[k] for k in ("step", "png", "file_sha256", "pixel_sha256", "state", "snapshot")}
                       for r in selected if r["phase"] == "plan" and r["step"] in (0, 5)]
        require(row["checkpoints"] == checkpoints and row["missing_checkpoints"] == [10], "checkpoint receipts")
        if status == "censored_out_of_domain":
            trial = copy.deepcopy(row)
            trial["status"] = "engineering_failed"
            trial["error"] = dict(stage="branch", type="Rejected", message="fresh RGB/canvas guard")
            info = run.proof.incident(events, anchor, probe, trial, _image(directory, frames[-1]["receipt"]))
            expected = {k: info[k] for k in ("anchor", "probe", "step", "action_float32_hex",
                                              "returned_state", "out_of_bounds", "frame_sha256", "blue_pixels")}
            censor = _one([e for e in scope if e["kind"] == "censored_out_of_domain"], "censor")
            require(row["censor"] == censor["proof"] == expected and frames[-1]["index"] < censor["index"]
                    < closed["index"], "censor proof/ordering")
            snapshot = frames[-1]["receipt"]["snapshot"]
            require(snapshot["bodies"]["agent"]["position"] == snapshot["state"][:2]
                    and snapshot["bodies"]["block"]["position"] == snapshot["state"][2:4],
                    "native body position/state binding")
        else:
            require(not [e for e in scope if e["kind"] == "censored_out_of_domain"], "unexpected censor")
            require(accepted_count == 5 if status == "probe_completed" else
                    bool(row["responses"]) and (row["responses"][-1]["terminated"]
                                                or row["responses"][-1]["truncated"]), "terminal status")
        require(all(not (r["terminated"] or r["truncated"]) for r in row["responses"][:-1]),
                "no continuation after native stop")
        if status in ("probe_completed", "censored_out_of_domain"):
            require(all(not r["terminated"] and not r["truncated"] for r in row["responses"]),
                    "unexpected prior native stop")
        endpoints.append(row["responses"][-1]["state"] if status == "probe_completed" else None)
        statuses.append(status)
    require(run.runner.physical.sha(path) == report_sha256
            and run.runner.physical.sha(journal_path) == report["journal_sha256"], "shard drift")
    return dict(anchor=anchor["index"], statuses=statuses, endpoints=endpoints)


def consume(directory, report_sha256, prepared, prepare_sha256):
    """All-or-nothing saved verification; caller independently pins run report."""
    p = run.verify_prepare(prepared, prepare_sha256)
    directory = Path(directory)
    require(directory.is_absolute() and directory.resolve() == directory and directory.is_dir(), "run directory")
    path = directory / "run_report.json"
    run.native.actual_hash(report_sha256)
    require(run.runner.physical.sha(path) == report_sha256, "external run report SHA")
    report = run.native.read(path, cap=4*1024**2)
    require(report["schema"] == run.NAME + ".run.v1" and report["status"] == "recorded_not_consumed"
            and report["fixed_branches"] == 336 and report["P1_admission"] is False
            and report["independent_confirmation"] is False and len(report["shards"]) == 16,
            "complete run required")
    bp = directory / "run_binding.json"
    require(run.runner.physical.sha(bp) == report["run_binding_sha256"], "binding SHA")
    binding = run.native.read(bp, cap=64*1024**2)
    expected_bindings = dict(p["bindings"], **{str(Path(prepared)): prepare_sha256})
    require(binding == dict(prepared=str(prepared), prepare_sha256=prepare_sha256,
                            bindings=expected_bindings, fixed_branches=336,
                            independent_confirmation=False, P1_admission=False), "run binding")
    rows = []
    for i, (anchor, probes, shard) in enumerate(zip(p["original"]["anchors"], p["original"]["probe_sets"],
                                                    report["shards"], strict=True)):
        require(shard["anchor"] == i and shard["status"] == "recorded_not_consumed", "ordered shards")
        rows.append(check_shard(directory / f"anchor_{i:02d}", anchor, probes, shard["report_sha256"]))
    for file, digest in expected_bindings.items():
        require(run.runner.physical.sha(file) == digest, "source drift")
    require(run.runner.physical.sha(path) == report_sha256
            and run.runner.physical.sha(bp) == report["run_binding_sha256"], "run metadata drift")
    counts = {s: sum(row["statuses"].count(s) for row in rows) for s in
              ("probe_completed", "native_stopped", "censored_out_of_domain")}
    require(sum(counts.values()) == 336, "fixed denominator")
    return dict(schema=run.NAME + ".consume.v1", fixed_branches=336, counts=counts, rows=rows,
                saved_checks_passed=True, runtime_authenticity_proved=False, independent_confirmation=False,
                P1_admission=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", required=True)
    ap.add_argument("--report-sha256", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--prepare-sha256", required=True)
    a = ap.parse_args()
    print(json.dumps(consume(a.directory, a.report_sha256, a.prepared, a.prepare_sha256), sort_keys=True))
