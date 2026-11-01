"""Replay fixed real cp5 prefixes and execute their CUDA-planned suffixes."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from . import p1_response_prefix_cp5_bundle_20261031 as bundle
from . import codex_p1_cp5_physical_24_20261030_v1 as engine
from . import codex_p1_cp5_physical_saved_20261030_v1 as saved
from . import p1_backtrack_physical_consume_20261030 as journal
from . import codex_p1_multistart_cp5_20261030_main as common

ROOT, need, sha = bundle.ROOT, bundle.need, bundle.sha
NAME = "p1_response_prefix_cp5_physical_20261031"
BUNDLE = ROOT / "results/p1_response_prefix_cp5_bundle_20261031_RUN01.json"
BUNDLE_SHA = "f8ffbfe57afdcf248c0150a99165d621db707b9d92db5e955e8c0db864b9e2c6"
PLAN = ROOT / "results/p1_response_prefix_cp5_adam_20261031_RUN01.json"
PLAN_SHA = "705b75f68bfbb6663e7731428580e1639344a7afb7e9c329aa2a9ebc17d3327b"
PROTOCOL = ROOT.parent / "docs/D2P_P1_RESPONSE_PREFIX_CP5_PHYSICAL_PROTOCOL_20261031.md"


def build():
    read = bundle.physical.prior.native.read
    need(sha(BUNDLE) == BUNDLE_SHA and sha(PLAN) == PLAN_SHA and
         sha(bundle.RUN / "report.json") == bundle.REPORT_SHA and sha(bundle.RECEIPT) == bundle.RECEIPT_SHA,
         "input hash")
    blind, plan, report, receipt = (read(BUNDLE, cap=64 * 1024**2), read(PLAN, cap=64 * 1024**2),
                                    read(bundle.RUN / "report.json", cap=64 * 1024**2), read(bundle.RECEIPT))
    need(plan["status"] == "complete" and plan["physical_execution"] is False and
         len(plan["rows"]) == len(blind["rows"]) == len(report["rows"]) == 16 and
         receipt["report_sha256"] == bundle.REPORT_SHA, "source contract")
    events = journal.events(bundle.RUN / "journal.jsonl", report["journal_chain_tail"])
    bindings = dict(plan["sources"])
    for path in (BUNDLE, PLAN, bundle.RUN / "report.json", bundle.RUN / "journal.jsonl", bundle.RECEIPT,
                 Path(__file__).resolve(), Path(engine.__file__).resolve(), Path(saved.__file__).resolve(), PROTOCOL):
        digest = sha(path)
        need(str(path) not in bindings or bindings[str(path)] == digest, "binding collision")
        bindings[str(path)] = digest
    anchors, items, censored_rows = [], [], []
    for i, (b, p, row) in enumerate(zip(blind["rows"], plan["rows"], report["rows"], strict=True)):
        need(b["index"] == p["index"] == row["index"] == i and
             b["candidate_index"] == p["candidate_index"] == row["plan"]["candidate_index"], "row identity")
        if b["blind"] is None:
            need(p["status"] == "censored_before_cp5" and p["actions"] is None, "original censor")
            censored_rows.append(i)
            continue
        need(p["status"] == "planned" and p["cp5_pixel_sha256"] == b["blind"]["cp5_pixel_sha256"] and
             p["prefix_raw_f32_hex"] == b["blind"]["prefix_raw_f32_hex"], "plan input identity")
        init = common.initial_actions(b["logical_pair"])
        need(hashlib.sha256(init.tobytes()).hexdigest() == p["initial_f32_sha256"]
             and common.pick(p["candidate_scores"]) == p["selected_index"], "initialization and selection")
        actions = np.asarray(p["actions"], dtype="<f4")
        need(actions.shape == (2, 5, 2) and np.isfinite(actions).all() and
             (np.abs(actions) <= 1).all() and hashlib.sha256(actions.tobytes()).hexdigest() ==
             p["actions_sha256"], "planned action")
        cp = next(c for c in row["checkpoints"] if c["step"] == 5)
        trace = [e for e in events if e["kind"] == "fresh_observation" and e.get("branch") == i
                 and (e["receipt"]["phase"] == "warmup" or
                      (e["receipt"]["phase"] == "plan" and e["receipt"]["step"] <= 5))]
        expected = [(phase, step, overlay) for phase, count in (("warmup", 11), ("plan", 6))
                    for step in range(count) for overlay in range(3)]
        need([(e["receipt"]["phase"], e["receipt"]["step"], e["receipt"]["overlay_index"])
              for e in trace] == expected, "original trace")
        for event in trace:
            rec = event["receipt"]
            path = bundle.RUN / rec["png"]
            need(path.parent == bundle.RUN and sha(path) == rec["file_sha256"], "source image")
            bindings[str(path)] = rec["file_sha256"]
        source_plan = row["plan"]
        anchor = {k: row[k] for k in ("index", "root_slot", "condition", "arm", "role", "task_id")}
        anchor.update(logical_pair=b["logical_pair"], goal=row["goal"], root_state=row["root_state"],
                      prefix5_float32_hex=p["prefix_raw_f32_hex"], checkpoint=cp,
                      trace=[{k: copy.deepcopy(e[k]) for k in ("receipt", "pre", "post")} for e in trace])
        need(cp["pixel_sha256"] == p["cp5_pixel_sha256"] and
             source_plan["raw15_float32_hex"][:80] == p["prefix_raw_f32_hex"], "cp5 and prefix")
        item = {k: anchor[k] for k in ("index", "logical_pair", "root_slot", "condition", "arm", "role", "task_id",
                                        "prefix5_float32_hex")}
        item.update(raw10_float32_hex=actions.tobytes().hex(), raw10_sha256=p["actions_sha256"])
        anchors.append(anchor)
        items.append(item)
    need(len(anchors) + len(censored_rows) == 16 and len(anchors) == plan["planned_rows"]
         and all(sha(p) == h for p, h in bindings.items()), "roster and source drift")
    return {"schema": NAME + ".prepare", "anchors": anchors, "roster": items,
            "censored_before_cp5": censored_rows, "fixed_rows": 16, "bindings": bindings,
            "P1_admission": False, "independent_confirmation": False}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--report-sha")
    parser.add_argument("--directory")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    if args.command == "prepare":
        fixed, _ = build()
        result = {"sha256": bundle.physical.write(out, fixed), "planned": len(fixed["anchors"])}
    else:
        need(args.prepared and args.sha256 and sha(args.prepared) == args.sha256, "prepare hash")
        fixed, _ = build()
        need(fixed == bundle.physical.prior.native.read(args.prepared, cap=64 * 1024**2), "prepare rebuild")
        if args.command == "run":
            need(args.authorize_physics and shutil.disk_usage(ROOT / "results").free >= 2 * 1024**3,
                 "physics authorization and disk")
            _, data = bundle.physical.pilot.material(bundle.physical.JOIN_SHA)
            bundle.physical.prior.native.constructor_contract()
            out.mkdir(mode=0o700)
            def factory():
                import gymnasium as gym
                import stable_worldmodel  # noqa: F401
                return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
            shards, failure = [], None
            for anchor, item in zip(fixed["anchors"], fixed["roster"], strict=True):
                path = out / f"row_{anchor['index']:02d}"
                r = engine.execute(anchor, item, data, path, factory)
                shards.append({"index": anchor["index"], "status": r["status"], "sha256": sha(path / "report.json")})
                print(json.dumps(shards[-1]), flush=True)
                if r["status"] != "recorded_not_consumed":
                    failure = r["errors"]
                    break
            need(all(sha(p) == h for p, h in fixed["bindings"].items()), "source drift")
            report = {"schema": NAME + ".run", "status": "recorded_not_consumed"
                      if len(shards) == len(fixed["anchors"]) and failure is None else "partial_or_failed",
                      "shards": shards, "failure": failure, "prepare_sha256": args.sha256,
                      "censored_before_cp5": fixed["censored_before_cp5"], "fixed_rows": 16,
                      "P1_admission": False}
            result = {"status": report["status"], "sha256": bundle.physical.write(out / "run_report.json", report)}
        else:
            directory = Path(args.directory)
            need(directory.is_absolute() and directory.is_dir() and not directory.is_symlink()
                 and sha(directory / "run_report.json") == args.report_sha, "run report hash")
            report = bundle.physical.prior.native.read(directory / "run_report.json")
            need(report["status"] == "recorded_not_consumed" and report["prepare_sha256"] == args.sha256
                 and len(report["shards"]) == len(fixed["anchors"]), "complete report")
            rows = []
            for anchor, item, shard in zip(fixed["anchors"], fixed["roster"], report["shards"], strict=True):
                need(shard["index"] == anchor["index"] and shard["status"] == "recorded_not_consumed",
                     "shard identity")
                rows.append(saved.check_row(directory / f"row_{anchor['index']:02d}", anchor, item, shard["sha256"]))
            result = {"status": "saved_checks_passed", "rows": rows, "fixed_rows": 16,
                      "censored_before_cp5": fixed["censored_before_cp5"], "P1_admission": False,
                      "independent_confirmation": False}
            digest = bundle.physical.write(out, result)
            result = {"status": result["status"], "rows": len(rows), "sha256": digest}
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
