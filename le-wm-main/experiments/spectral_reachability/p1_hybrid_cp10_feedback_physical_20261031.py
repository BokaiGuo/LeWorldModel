"""Physically execute cp10 feedback after exact replay of fixed first ten steps."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import p1_hybrid_cp5_physical_20261031 as old
from . import p1_hybrid_cp10_feedback_cuda_20261031 as feedback
from . import codex_p1_multistart_cp5_20261030_main as common

ROOT, need, sha = old.ROOT, old.need, old.sha
NAME = "p1_hybrid_cp10_feedback_physical_20261031"
PLAN = ROOT / "results/p1_hybrid_cp10_feedback_cuda_20261031_RUN01.json"
PLAN_SHA = "a9c457d2e2ff1ae38c4cd4d7eece9bbb48fd784931c15bea58c0f09228ac6eec"
SOURCE_PREPARE = old.ROOT / "results/p1_hybrid_cp5_physical_20261031_prepare_RUN01.json"
SOURCE_PREPARE_SHA = "b91993a25e7c504d16015f0509685ef76566e9655d451a92918c10f06d6b243a"


def build():
    need(sha(PLAN) == PLAN_SHA and sha(SOURCE_PREPARE) == SOURCE_PREPARE_SHA, "pinned inputs")
    plan, original = old.h.read(PLAN), old.h.read(SOURCE_PREPARE)
    need(original == old.build() and plan["status"] == "complete" and len(plan["rows"]) == 2
         and plan["model_before"] == plan["model_after"] and plan["host"]["device"] == "cuda:0",
         "verified input")
    old_report = old.h.read(feedback.source.DIRECTORY / "run_report.json")
    need(sha(feedback.source.DIRECTORY / "run_report.json") == feedback.source.REPORT_SHA, "old physical report")
    bindings = dict(original["bindings"])
    bindings.update(plan["sources"])
    for path in (PLAN, SOURCE_PREPARE, Path(__file__).resolve()):
        bindings[str(path)] = sha(path)
    items, midpoints, probe_traces = [], [], []
    for anchor, old_item, row, shard in zip(original["anchors"], original["items"], plan["rows"],
                                             old_report["shards"], strict=True):
        need(anchor["index"] == row["index"] == shard["index"] and row["root_slot"] == anchor["root_slot"],
             "row binding")
        init = common.initial_actions(anchor["logical_pair"])[:, 1]
        need(hashlib.sha256(init.tobytes()).hexdigest() == row["initial_sha256"]
             and common.pick(row["candidate_scores"]) == row["selected_index"], "optimizer identity")
        actions = np.asarray(row["actions"], dtype="<f4")
        previous = bytes.fromhex(old_item["raw10_float32_hex"])
        need(actions.shape == (2, 5, 2) and np.isfinite(actions).all() and (abs(actions) <= 1).all()
             and hashlib.sha256(actions.tobytes()).hexdigest() == row["actions_sha256"]
             and actions[0].tobytes() == previous[:40]
             and hashlib.sha256(previous[:40]).hexdigest() == row["first_packet_sha256"]
             and row["prefix_raw_f32_hex"] == anchor["prefix5_float32_hex"], "action identity")
        folder = feedback.source.DIRECTORY / f"row_{anchor['index']:02d}"
        report_path = folder / "report.json"
        need(sha(report_path) == shard["sha256"], "old row hash")
        report = old.h.read(report_path)
        checkpoint = [x for x in report["row"]["trajectory"] if x["phase"] == "probe" and x["step"] == 5]
        need(len(checkpoint) == 1 and checkpoint[0]["pixel_sha256"] == row["midpoint_pixel_sha256"],
             "old midpoint")
        trace = [x for x in report["row"]["trajectory"] if x["phase"] == "probe" and 1 <= x["step"] <= 5]
        need([x["step"] for x in trace] == list(range(1, 6)), "original first-packet frames")
        probe_traces.append([{k: frame[k] for k in ("step", "state", "snapshot", "pixel_sha256")}
                             for frame in trace])
        midpoints.append({k: checkpoint[0][k] for k in ("state", "snapshot", "pixel_sha256")})
        item = dict(old_item, raw10_float32_hex=actions.tobytes().hex(), raw10_sha256=row["actions_sha256"])
        items.append(item)
        bindings[str(report_path)] = shard["sha256"]
    need(len(items) == 2 and all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "anchors": original["anchors"], "items": items,
            "midpoints": midpoints, "probe_traces": probe_traces,
            "bindings": bindings, "fixed_rows": 2, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results" and
         output.name.startswith(NAME + "_") and not output.exists(), "exclusive output")
    fixed = build()
    if args.command == "prepare":
        print(json.dumps({"sha256": old.h.engine._write_new(output, fixed)}))
        return
    need(args.authorize_physics and args.prepared and sha(args.prepared) == args.sha256
         and old.h.read(args.prepared) == fixed and shutil.disk_usage(ROOT / "results").free >= 2 * 1024**3,
         "prepare/physics/disk")
    _, data = old.h.existing.physical.bundle.physical.pilot.material(
        old.h.existing.physical.bundle.physical.JOIN_SHA)
    old.h.existing.physical.bundle.physical.prior.native.constructor_contract()
    output.mkdir(mode=0o700)

    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

    dispatch = old.h.engine.base.dispatch
    midpoints = {a["index"]: mid for a, mid in zip(fixed["anchors"], fixed["midpoints"], strict=True)}
    expected_traces = {a["index"]: trace for a, trace in
                       zip(fixed["anchors"], fixed["probe_traces"], strict=True)}

    def verify_before_second_packet(env, command, phase, step, row, meter, trace, goal):
        if phase == "probe" and step == 6:
            records = [x for x in row["trajectory"] if x["phase"] == "probe" and 1 <= x["step"] <= 5]
            expected = midpoints[row["anchor_index"]]
            expected_trace = expected_traces[row["anchor_index"]]
            need(len(records) == len(expected_trace) == 5 and
                 all(all(record[k] == ref[k] for k in ref)
                     for record, ref in zip(records, expected_trace, strict=True)) and
                 all(records[-1][k] == expected[k] for k in expected),
                 "first packet or cp10 mismatch before replanning action")
            trace.emit("midpoint_refresh_verified", **meter.where, checkpoint=expected, trace=expected_trace)
        return dispatch(env, command, phase, step, row, meter, trace, goal)

    shards = []
    with patch.object(old.h.engine.prior, "compare_trace", old.h.compare_trace), \
         patch.object(old.h.engine.base, "dispatch", verify_before_second_packet):
        for anchor, item in zip(fixed["anchors"], fixed["items"], strict=True):
            path = output / f"row_{anchor['index']:02d}"
            result = old.engine.execute(anchor, item, data, path, factory)
            shards.append({"index": anchor["index"], "status": result["status"], "sha256": sha(path / "report.json")})
            print(json.dumps(shards[-1]), flush=True)
            if result["status"] != "recorded_not_consumed":
                break
    need(all(sha(path) == digest for path, digest in fixed["bindings"].items()), "source drift")
    status = ("recorded_not_consumed" if len(shards) == 2 and
              all(s["status"] == "recorded_not_consumed" for s in shards) else "partial_or_failed")
    result = {"schema": NAME + ".run", "status": status, "shards": shards,
              "prepare_sha256": args.sha256, "fixed_rows": 2, "P1_admission": False}
    print(json.dumps({"status": status, "sha256": old.h.engine._write_new(output / "run_report.json", result)}))


if __name__ == "__main__":
    main()
