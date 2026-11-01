"""Pair fixed known-root 20-step open loop with two-stage RGB feedback.

Read-only re-consumption precedes joining. Different neural budgets and the
post-observation exploratory design preclude an equal-cost or P1 claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

from . import codex_p1_cp5_twenty_physical_saved_20261030_v1 as open_saved
from . import codex_p1_two_stage_saved_20261030_v1 as feedback_saved

ROOT = open_saved.ROOT
need, sha = open_saved.need, open_saved.sha
NAME = "codex_p1_feedback_open_paired_20261030_v1"
OPEN_DIR = ROOT/"results/codex_p1_cp5_twenty_physical_20261030_v1_run_CODEX01"
OPEN_PREP = ROOT/"results/codex_p1_cp5_twenty_physical_20261030_v1_prepare_CODEX01.json"
OPEN_SAVED = ROOT/"results/codex_p1_cp5_twenty_physical_saved_20261030_v1_CODEX01.json"
FB_DIR = ROOT/"results/codex_p1_two_stage_launcher_20261030_v1_run_CODEX01"
FB_PREP = ROOT/"results/codex_p1_two_stage_launcher_20261030_v1_prepare_CODEX01.json"
FB_SAVED = ROOT/"results/codex_p1_two_stage_saved_20261030_v1_CODEX01.json"
OPEN_PREP_SHA = "27c7256672b5abecaf12cdf565ccd70b17dcd9a805c11d50307c834a3e2b2665"
OPEN_RUN_SHA = "0702e43833afc9d00ee69a7b07889389d893eb2a17d2b5890b7f229b63b36273"
OPEN_SAVED_SHA = "61860286e0a351cfd553fb5aaa5bd986924489bf2a9a6c66601ef2ec8da6147f"
FB_PREP_SHA = "edea86663dd55caa9a7c5a6f48862030fb84248164df1691f38a82b2ca9672c5"
FB_RUN_SHA = "2edad31c9315b6b2ab7cb9f8424c7b51a4f91ef283293066ccd784c91d3d2ded"
FB_SAVED_SHA = "e5a1eebfbf12e14c9705b6020dcdb7dba5a6eb0e1c41e2168678f1da56a48c82"
IDENTITY = ("index", "logical_pair", "root_slot", "condition", "arm", "role", "task_id",
            "physical_root_sha256", "prefix5_float32_hex")
METRICS = ("object_position_error", "object_angle_error_wrapped", "pusher_position_error")


def checked(path, digest):
    need(path.is_file() and not path.is_symlink() and sha(path) == digest, "source hash: "+str(path))
    value = json.loads(path.read_bytes())
    need(sha(path) == digest, "source drift")
    return value


def pair_row(anchor, op, fb, fb_roster, action_sources):
    for key in IDENTITY:
        need(anchor[key] == op[key] == fb_roster[key], "identity mismatch: "+key)
    for key in ("index", "logical_pair", "root_slot", "condition", "arm", "role"):
        need(fb[key] == anchor[key], "feedback identity mismatch: "+key)
    checkpoint = anchor["checkpoint"]
    need(checkpoint == fb_roster["checkpoint"] and
         op["cp5_file_sha256"] == checkpoint["file_sha256"] and
         op["cp5_pixel_sha256"] == checkpoint["pixel_sha256"], "cp5 source mismatch")
    old = op["twenty_step"]
    n, m = old["accepted_suffix_steps"], fb["accepted_suffix_steps"]
    need(type(n) is int and type(m) is int and 0 <= n <= 10 and 0 <= m <= 10, "step counts")
    need(old["early_native_stop"] == (old["status"] == "native_stopped" and n < 10)
         and old["native_stop_at_horizon"] == (old["status"] == "native_stopped" and n == 10)
         and old["censored"] == (old["status"] == "censored_out_of_domain")
         and fb["early_stop"] == (fb["status"] == "native_stopped" and m < 10)
         and fb["native_stop_at_horizon"] == (fb["status"] == "native_stopped" and m == 10),
         "stop/censor classification")
    a, b = old["horizon_metrics"], fb["endpoint_metrics"]
    need((a is not None) == (n == 10 and not old["censored"])
         and (b is not None) == (m == 10), "endpoint denominator")
    delta = {k: b[k]-a[k] for k in METRICS} if a is not None and b is not None else None
    return {"identity": {k: anchor[k] for k in IDENTITY},
            "cp5": {"file_sha256": checkpoint["file_sha256"], "pixel_sha256": checkpoint["pixel_sha256"],
                    "state": checkpoint["state"], "snapshot": checkpoint["snapshot"]},
            "open_20": {"status": old["status"], "steps": n, "early_stop": old["early_native_stop"],
                        "stop_at_horizon": old["native_stop_at_horizon"], "censored": old["censored"],
                        "endpoint": a, "physical_resources": old["physical_resources"],
                        "stage_resources": [old["planning_resources"]], "actions": action_sources["open"]},
            "feedback_20_plus_20": {"status": fb["status"], "steps": m, "early_stop": fb["early_stop"],
                                    "stop_at_horizon": fb["native_stop_at_horizon"], "censored": False,
                                    "endpoint": b, "physical_resources": fb["physical_resources"],
                                    "stage_resources": fb["stage_resources"], "actions": action_sources["feedback"]},
            "feedback_minus_open": delta}


def action_sources(i, op, feedback_row, feedback_run):
    open_item = open_saved.run.read(OPEN_PREP, OPEN_PREP_SHA)["roster"][i]
    raw = bytes.fromhex(open_item["raw10_float32_hex"])
    need(len(raw) == 80 and hashlib.sha256(raw).hexdigest() ==
         open_item["raw10_sha256"] == op["twenty_step"]["raw10_sha256"], "open action SHA")
    stage_records = []
    for phase in range(1, len(feedback_row["stage_resources"])+1):
        path = FB_DIR/f"row_{i:02d}"/f"stage_{phase}"/"report.json"
        rec = json.loads(path.read_bytes())
        raw_horizon = bytes.fromhex(rec["result"]["raw_horizon_f32_hex"])
        need(len(raw_horizon) == 80, "feedback horizon length")
        stage_records.append({"phase": phase, "report_sha256": sha(path),
                              "input_rgb_sha256": rec["binding"]["rgb_sha256"],
                              "executed_first_packet_float32_hex": raw_horizon[:40].hex(),
                              "executed_first_packet_sha256": hashlib.sha256(raw_horizon[:40]).hexdigest(),
                              "full_planned_horizon_sha256": hashlib.sha256(raw_horizon).hexdigest()})
    need(feedback_run["shards"][i]["index"] == i, "feedback shard")
    return {"open": {"raw10_float32_hex": raw.hex(), "raw10_sha256": hashlib.sha256(raw).hexdigest(),
                     "plan_row_sha256": open_item["plan_row_sha256"]},
            "feedback": stage_records}


def build():
    original_open = checked(OPEN_SAVED, OPEN_SAVED_SHA)
    original_fb = checked(FB_SAVED, FB_SAVED_SHA)
    op_prepare = checked(OPEN_PREP, OPEN_PREP_SHA)
    fb_prepare = checked(FB_PREP, FB_PREP_SHA)
    fb_run = checked(FB_DIR/"run_report.json", FB_RUN_SHA)
    need(open_saved.consume(OPEN_DIR, OPEN_RUN_SHA, OPEN_PREP, OPEN_PREP_SHA) == original_open,
         "open saved re-consumption")
    need(feedback_saved.consume(FB_DIR, FB_RUN_SHA, FB_PREP, FB_PREP_SHA) == original_fb,
         "feedback saved re-consumption")
    anchors = open_saved.run.h10.native.read(open_saved.run.h10.INPUT, cap=64*1024**2)["anchors"]
    need(len(anchors) == len(op_prepare["roster"]) == len(fb_prepare["roster"]) ==
         len(original_open["rows"]) == len(original_fb["rows"]) == 24, "full 24-row denominator")
    rows = []
    for i in range(24):
        anchor, op, fb = anchors[i], original_open["rows"][i], original_fb["rows"][i]
        need(anchor["index"] == op_prepare["roster"][i]["index"] == i, "ordered roster")
        actions = action_sources(i, op, fb, fb_run)
        rows.append(pair_row(anchor, op, fb, fb_prepare["roster"][i], actions))
    for path, digest in ((OPEN_PREP, OPEN_PREP_SHA), (OPEN_SAVED, OPEN_SAVED_SHA),
                         (FB_PREP, FB_PREP_SHA), (FB_SAVED, FB_SAVED_SHA),
                         (OPEN_DIR/"run_report.json", OPEN_RUN_SHA), (FB_DIR/"run_report.json", FB_RUN_SHA)):
        need(sha(path) == digest, "source drift after consumption")
    summary = {}
    for role in ("selected", "zero", "greedy"):
        group = [r for r in rows if r["identity"]["role"] == role]
        eligible = [r for r in group if r["feedback_minus_open"] is not None]
        need(len(group) == 8, "role denominator")
        summary[role] = {"rows": len(group), "complete_pairs": len(eligible),
                         "mean_feedback_minus_open": {key: sum(r["feedback_minus_open"][key] for r in eligible)
                                                      / len(eligible) if eligible else None for key in METRICS},
                         "open_native_success": sum(bool(r["open_20"]["endpoint"]["native_success"])
                                                    for r in eligible),
                         "feedback_native_success": sum(bool(r["feedback_20_plus_20"]["endpoint"]["native_success"])
                                                        for r in eligible)}
    return {"schema": NAME, "status": "24_pairs_saved_verified", "rows": rows, "summary": summary,
            "source_hashes": {"open_prepare": OPEN_PREP_SHA, "open_run": OPEN_RUN_SHA,
                              "open_saved": OPEN_SAVED_SHA, "feedback_prepare": FB_PREP_SHA,
                              "feedback_run": FB_RUN_SHA, "feedback_saved": FB_SAVED_SHA},
            "comparison": "feedback minus open; negative error difference favors feedback",
            "denominator_note": "Known roots and duplicate trajectories; rows are not independent samples.",
            "budgets_differ": True, "equal_budget_certified": False, "isolation_certified": False,
            "independent_confirmation": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    need(path.is_absolute() and path.parent == ROOT/"results" and path.name.startswith(NAME+"_"),
         "exclusive result namespace")
    value = build()
    digest = open_saved.run.h10._write_new(path, value)
    print(json.dumps({"status": value["status"], "rows": len(value["rows"]), "sha256": digest,
                      "summary": value["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
