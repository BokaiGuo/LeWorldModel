"""Fixed 24-row physical replay of 20-step cp5 plans; no neural inference."""
import argparse
import json
import os
from pathlib import Path
import shutil

from . import codex_p1_cp5_physical_24_20261030_v1 as old
from . import codex_p1_cp5_unified_consumer_20261030_v1 as ledger

ROOT, h10, sha, need = old.ROOT, old.h10, old.sha, old.need
NAME = "codex_p1_cp5_twenty_physical_20261030_v1"
PLAN_DIR = ROOT/"results/codex_p1_cp5_twenty_step_cuda_20261030_main_RUN01"
PLAN_RECEIPT = ROOT/"results/codex_p1_cp5_twenty_step_saved_20261030_main_RUN01.json"
PLAN_RECEIPT_SHA = "21a9c84a563730805688ddf5a9dab5a049a1646fd2a75b39ab32812790c91215"
MANIFEST_SHA = "59e50f0277e6757d182455268c793a781d4992df2b39e94067fbe255b46bf1b2"
REPORT_SHA = "c1e80a1e1606bc829ea7d10b7dc18bef0baa19086ce0e3bdc6b1d1331ccc2fb6"
OLD_PREP = ROOT/"results/codex_p1_cp5_physical_24_20261030_v1_prepare_CODEX01.json"
OLD_SHA = "367091192c51c3be2565ee8c7d811cde164c155a4720c0c19531836c8b3fb737"
CONFIG = {"steps": 20, "learning_rate": 0.05, "history_size": 3}


def read(path, digest):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and sha(path) == digest, "source SHA: "+str(path))
    data = json.loads(path.read_bytes())
    need(sha(path) == digest, "source drift")
    return data


def planner_sources():
    receipt = read(PLAN_RECEIPT, PLAN_RECEIPT_SHA)
    manifest = read(PLAN_DIR/"manifest.json", MANIFEST_SHA)
    report = read(PLAN_DIR/"report.json", REPORT_SHA)
    need(receipt["schema"] == "codex_p1_cp5_twenty_step_saved_20261030_main"
         and receipt["status"] == "24_rows_consumed" and len(receipt["rows"]) == 24
         and receipt["manifest_sha256"] == MANIFEST_SHA and receipt["report_sha256"] == REPORT_SHA
         and manifest["optimizer"] == CONFIG and manifest["rows"] == 24
         and manifest["bundle_sha256"] == old.plans.run.BUNDLE_SHA
         and manifest["expected_model"] == old.plans.run.MODEL_HASH
         and report["status"] == "complete" and report["rows_completed"] == 24
         and report["manifest_sha256"] == MANIFEST_SHA and report["physical_execution"] is False
         and report["model_before"] == report["model_after"] == manifest["expected_model"],
         "20-step frozen planner identity")
    events = []
    for index, rec in enumerate(receipt["rows"]):
        need(rec["index"] == index, "planning roster order")
        row_path = PLAN_DIR/f"row_{index:02d}_result.json"
        event = read(row_path, rec["row_sha256"])
        need(event["kind"] == "row_complete" and event["index"] == event["result"]["index"] == index,
             "planning row identity")
        saved = event["result"]["ledger_receipt"]
        need(saved["encoding"]["sha256"] == rec["encoder_sha256"] and
             saved["planning"]["journal_sha256"] == rec["planning_sha256"], "planner receipt hashes")
        with (PLAN_DIR/f"row_{index:02d}"/"planning/events.jsonl").open("rb") as stream:
            header = json.loads(stream.readline())
        identity = header["payload"]["contract"]["identity"]
        need(identity["optimizer_config"] == old.plans.run.spec.digest(CONFIG)
             and identity["model"] == manifest["expected_model"]["original_state_dict"],
             "planner configuration/model binding")
        account = ledger.consume(PLAN_DIR/f"row_{index:02d}"/"encoder.jsonl", rec["encoder_sha256"],
                    manifest["encoder_caps"], manifest["wall_seconds"],
                    PLAN_DIR/f"row_{index:02d}"/"planning/events.jsonl", rec["planning_sha256"],
                    saved["planning"]["contract_sha256"], "structured", "fixed", f"row_{index:02d}")
        counts = account["planning"]["charged_upper_bounds"]
        need(account == saved and account["complete"] and counts["action_encode_calls"] == 42
             and counts["predict_calls"] == counts["predict_samples"] == 42
             and counts["backward_calls"] == counts["vjp_rows"] == 20
             and counts["replay_calls"] == counts["selection_evaluations"] == 1,
             "actual 20-step planning ledger")
        events.append(event)
    return receipt, manifest, events


def build():
    original = h10.design.verify(h10.INPUT, h10.INPUT_SHA)
    old_p = read(OLD_PREP, OLD_SHA)
    receipt, manifest, events = planner_sources()
    blind = read(old.plans.run.BUNDLE, old.plans.run.BUNDLE_SHA)
    roster = old.make_roster(original["anchors"], blind["rows"], events, receipt)
    for item, previous in zip(roster, old_p["roster"], strict=True):
        for key in ("index", "logical_pair", "root_slot", "condition", "arm", "role", "task_id",
                    "prefix5_float32_hex", "cp5_file_sha256", "cp5_pixel_sha256"):
            need(item[key] == previous[key], "same 2-step/20-step cp5 identity: "+key)
    bindings = dict(old_p["bindings"])
    files = [OLD_PREP, PLAN_RECEIPT, PLAN_DIR/"manifest.json", PLAN_DIR/"report.json",
             Path(__file__).resolve(), Path(__file__).with_name("codex_p1_cp5_twenty_physical_saved_20261030_v1.py"),
             ROOT/"tests/test_codex_p1_cp5_twenty_physical_20261030_v1.py"]
    for item in receipt["rows"]:
        files.extend([PLAN_DIR/f"row_{item['index']:02d}_result.json",
                      PLAN_DIR/f"row_{item['index']:02d}"/"encoder.jsonl",
                      PLAN_DIR/f"row_{item['index']:02d}"/"planning/events.jsonl"])
    for path in files:
        bindings[str(path)] = sha(path)
    need(all(sha(path) == digest for path, digest in bindings.items()), "source binding drift")
    return {"schema": NAME+".prepare", "roster": roster, "fixed_rows": 24, "caps_per_row": old.CAPS,
            "h10_prepare_sha256": h10.INPUT_SHA, "two_step_prepare_sha256": OLD_SHA,
            "twenty_step_manifest_sha256": MANIFEST_SHA, "twenty_step_report_sha256": REPORT_SHA,
            "twenty_step_receipt_sha256": PLAN_RECEIPT_SHA, "optimizer": CONFIG,
            "bindings": bindings, "known_roots": True, "os_isolation_certified": False,
            "independent_confirmation": False, "equal_budget_certified": False, "P1_admission": False}


def prepare(path):
    path = Path(path)
    need(path.is_absolute() and path.parent == ROOT/"results" and path.name.startswith(NAME+"_prepare_")
         and not os.path.lexists(path), "exclusive prepare")
    return h10._write_new(path, build())


def verify(path, expected):
    saved = read(path, expected)
    need(saved == build() and sha(path) == expected, "prepare/source mismatch")
    return saved


def run(prepared, expected, output):
    p = verify(prepared, expected)
    source_report = h10.native.read(h10.old.prep.DIRECTORY/"report.json", cap=64*1024**2)
    _, data = h10.runner.physical.verify(source_report["execution_binding"]["prepared"], h10.old.prep.PREP_SHA)
    h10.native.constructor_contract()
    anchors = h10.native.read(h10.INPUT, cap=64*1024**2)["anchors"]
    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT/"results" and
         output.name.startswith(NAME+"_run_") and not os.path.lexists(output) and
         shutil.disk_usage(output.parent).free >= 2*1024**3, "exclusive output/disk")
    output.mkdir(mode=0o700)
    binding_sha = h10._write_new(output/"binding.json", {"prepare_sha256": expected, "fixed_rows": 24})
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
    shards, failure = [], None
    try:
        for anchor, item in zip(anchors, p["roster"], strict=True):
            folder = output/f"row_{anchor['index']:02d}"
            result = old.execute(anchor, item, data, folder, factory)
            shards.append({"index": anchor["index"], "report_sha256": sha(folder/"report.json"),
                           "status": result["status"]})
            if result["status"] != "recorded_not_consumed":
                failure = {"index": anchor["index"], "errors": result["errors"]}
                break
        need(all(sha(path) == digest for path, digest in p["bindings"].items()), "postrun source drift")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    report = {"schema": NAME+".run", "status": "recorded_not_consumed" if len(shards) == 24 and not failure
              else "partial_or_failed", "shards": shards, "failure": failure, "fixed_rows": 24,
              "binding_sha256": binding_sha, "prepare_sha256": expected, "P1_admission": False,
              "independent_confirmation": False, "equal_budget_certified": False}
    digest = h10._write_new(output/"run_report.json", report)
    print(json.dumps({"status": report["status"], "sha256": digest, "rows": len(shards),
                      "failure": failure}), flush=True)
    return 0 if report["status"] == "recorded_not_consumed" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--prepared")
    ap.add_argument("--sha256")
    a = ap.parse_args()
    if a.command == "prepare":
        print(json.dumps({"sha256": prepare(a.output)}))
        return 0
    need(a.prepared and a.sha256, "prepared contract required")
    return run(a.prepared, a.sha256, a.output)


if __name__ == "__main__":
    raise SystemExit(main())
