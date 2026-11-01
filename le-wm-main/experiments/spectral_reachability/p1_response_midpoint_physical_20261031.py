"""Execute fixed midpoint-replanned actions with a pre-dispatch midpoint guard."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from . import p1_response_cp5_same_action_20261031 as source
from . import p1_response_prefix_cp5_physical_20261031 as prior

ROOT, need, sha = prior.ROOT, prior.need, prior.sha
NAME = "p1_response_midpoint_physical_20261031"
PLAN = ROOT / "results/p1_response_midpoint_replan_20261031_RUN01.json"
PLAN_SHA = "619d6ba332127b47a506c735857d7be7db3340fb406e680761087ded62ee0473"
PROTOCOL = ROOT.parent / "docs/D2P_P1_RESPONSE_MIDPOINT_PHYSICAL_PROTOCOL_20261031.md"


def build():
    need(sha(PLAN) == PLAN_SHA and sha(source.PREP) == source.PREP_SHA, "sealed source")
    read = prior.bundle.physical.prior.native.read
    plan = read(PLAN, cap=64 * 1024**2)
    prepared = read(source.PREP, cap=64 * 1024**2)
    entries, excluded, sources = source.collect()
    need(plan["status"] == "complete" and len(entries) == len(plan["rows"]) == 13
         and plan["excluded"] == excluded and plan["fixed_rows"] == 16, "fixed roster")
    need(all(sha(p) == digest for p, digest in plan["sources"].items()), "planning source drift")
    originals = {item["index"]: (anchor, item) for anchor, item in zip(
        prepared["anchors"], prepared["roster"], strict=True)}
    anchors, roster = [], []
    for entry, planned in zip(entries, plan["rows"], strict=True):
        i = entry["index"]
        anchor, old = originals[i]
        need(all(planned[k] == entry[k] for k in ("index", "logical_pair", "condition", "arm", "role", "root_slot"))
             and planned["first_packet_hex"] == entry["raw"][0].tobytes().hex()
             and planned["original_second_packet_hex"] == entry["raw"][1].tobytes().hex()
             and planned["original_action_sha256"] == old["raw10_sha256"], "plan action identity")
        need(planned["midpoint_pixel_sha256"] == hashlib.sha256(entry["mid"].tobytes()).hexdigest(),
             "midpoint pixel identity")
        first, second = (bytes.fromhex(planned[k]) for k in ("first_packet_hex", "second_packet_hex"))
        action = np.frombuffer(first + second, dtype="<f4")
        need(action.shape == (20,) and np.isfinite(action).all() and (np.abs(action) <= 1).all()
             and planned["chosen"] == ("optimized" if planned["optimized_score"] < planned["baseline_score"]
                                       else "original")
             and (planned["chosen"] != "original" or second == entry["raw"][1].tobytes()), "selected action")
        original_report_path = source.RUN / f"row_{i:02d}" / "report.json"
        original_report = read(original_report_path, cap=64 * 1024**2)
        frames = [f for f in original_report["row"]["trajectory"]
                  if f["phase"] == "probe" and f["step"] == 5]
        need(len(frames) == 1 and frames[0]["pixel_sha256"] == planned["midpoint_pixel_sha256"],
             "midpoint receipt")
        item = dict(old, raw10_float32_hex=(first + second).hex(),
                    raw10_sha256=hashlib.sha256(first + second).hexdigest(),
                    midpoint={k: frames[0][k] for k in ("pixel_sha256", "state", "snapshot")})
        anchors.append(anchor)
        roster.append(item)
    bindings = dict(plan["sources"])
    for mapping in (sources, {str(PLAN): PLAN_SHA, str(source.PREP): source.PREP_SHA,
                              str(PROTOCOL): sha(PROTOCOL), str(Path(__file__).resolve()): sha(__file__)}):
        for path, digest in mapping.items():
            need(path not in bindings or bindings[path] == digest, "binding collision")
            bindings[path] = digest
    need(all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "anchors": anchors, "roster": roster,
            "excluded": excluded, "fixed_rows": 16, "bindings": bindings,
            "P1_admission": False, "independent_confirmation": False}


def guarded_execute(anchor, item, data, path, factory):
    """Reject changed actual midpoint before any second-packet dispatch."""
    base = prior.engine.h10.base
    original = base.capture
    matched = []

    def capture(env, obs, phase, step, row, meter, store, reference=None):
        pixels = original(env, obs, phase, step, row, meter, store, reference)
        if phase == "probe" and step == 5:
            receipt = row["trajectory"][-1]
            need(all(receipt[k] == item["midpoint"][k] for k in ("pixel_sha256", "state", "snapshot"))
                 and hashlib.sha256(pixels.tobytes()).hexdigest() == item["midpoint"]["pixel_sha256"],
                 "midpoint replay mismatch")
            matched.append(True)
        return pixels

    base.capture = capture
    try:
        report = prior.engine.execute(anchor, item, data, path, factory)
    finally:
        base.capture = original
    if len(report["row"]["responses"]) >= 5:
        need(matched == [True], "midpoint guard absent")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--directory")
    parser.add_argument("--report-sha")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    fixed = build()
    if args.command == "prepare":
        result = {"sha256": prior.bundle.physical.write(out, fixed), "rows": len(fixed["anchors"])}
    else:
        need(args.prepared and args.sha256 and sha(args.prepared) == args.sha256
             and prior.bundle.physical.prior.native.read(args.prepared, cap=64 * 1024**2) == fixed,
             "prepared binding")
        if args.command == "run":
            need(args.authorize_physics and shutil.disk_usage(ROOT / "results").free >= 2 * 1024**3,
                 "physics authorization and disk")
            _, data = prior.bundle.physical.pilot.material(prior.bundle.physical.JOIN_SHA)
            prior.bundle.physical.prior.native.constructor_contract()
            out.mkdir(mode=0o700)

            def factory():
                import gymnasium as gym
                import stable_worldmodel  # noqa: F401
                return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

            shards = []
            for anchor, item in zip(fixed["anchors"], fixed["roster"], strict=True):
                path = out / f"row_{anchor['index']:02d}"
                report = guarded_execute(anchor, item, data, path, factory)
                shards.append({"index": anchor["index"], "status": report["status"],
                               "sha256": sha(path / "report.json")})
                print(json.dumps(shards[-1]), flush=True)
                if report["status"] != "recorded_not_consumed":
                    break
            need(all(sha(p) == digest for p, digest in fixed["bindings"].items()), "source drift")
            report = {"schema": NAME + ".run", "status": "recorded_not_consumed"
                      if len(shards) == len(fixed["anchors"]) and all(
                          shard["status"] == "recorded_not_consumed" for shard in shards) else "partial_or_failed",
                      "shards": shards, "prepare_sha256": args.sha256, "excluded": fixed["excluded"],
                      "fixed_rows": 16, "P1_admission": False}
            result = {"status": report["status"], "sha256": prior.bundle.physical.write(out / "run_report.json", report)}
        else:
            directory = Path(args.directory)
            need(directory.is_absolute() and directory.is_dir() and not directory.is_symlink()
                 and sha(directory / "run_report.json") == args.report_sha, "run report hash")
            report = prior.bundle.physical.prior.native.read(directory / "run_report.json")
            need(report["status"] == "recorded_not_consumed" and report["prepare_sha256"] == args.sha256
                 and len(report["shards"]) == len(fixed["anchors"]), "complete report")
            rows = []
            for anchor, item, shard in zip(fixed["anchors"], fixed["roster"], report["shards"], strict=True):
                need(shard["index"] == anchor["index"] and shard["status"] == "recorded_not_consumed",
                     "shard identity")
                path = directory / f"row_{anchor['index']:02d}"
                checked = prior.saved.check_row(path, anchor, item, shard["sha256"])
                row = prior.bundle.physical.prior.native.read(path / "report.json", cap=64 * 1024**2)["row"]
                if checked["returned_suffix"] >= 5:
                    frames = [f for f in row["trajectory"] if f["phase"] == "probe" and f["step"] == 5]
                    need(len(frames) == 1 and all(frames[0][k] == item["midpoint"][k]
                         for k in ("pixel_sha256", "state", "snapshot")), "saved midpoint mismatch")
                rows.append(checked)
            value = {"status": "saved_checks_passed", "rows": rows, "fixed_rows": 16,
                     "excluded": fixed["excluded"], "P1_admission": False, "independent_confirmation": False}
            result = {"status": value["status"], "rows": len(rows), "sha256": prior.bundle.physical.write(out, value)}
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
