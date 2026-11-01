"""Fixed selected-candidate suffixes applied to paired zero and greedy prefixes."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from . import p1_uphill_nonwinner_physical_20261031 as source
from . import p1_packet_feedback_crossover_consume_20261031 as validation
from . import p1_packet_feedback_crossover_20261031 as crossover
from . import codex_p1_h10_paired_table_20261030_main as metric

original = source.original
physical = original.physical
ROOT = physical.ROOT
NAME = "p1_selected_candidates_crossover_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_NONWINNER_CROSSOVER_PROTOCOL_20261031.md"
need, sha = source.need, source.sha


def build():
    prior = original.build()
    need(sha(source.REPLAY) == source.REPLAY_SHA, "replay hash")
    replay = json.loads(source.REPLAY.read_bytes())
    plan = json.loads(original.PLAN.read_bytes())
    need(sha(physical.bundle.PREP) == physical.PREP_SHA, "anchor hash")
    anchors = json.loads(physical.bundle.PREP.read_bytes())["anchors"]
    need(replay["status"] == "complete" and replay["source_sha256"] == original.PLAN_SHA
         and len(replay["rows"]) == len(plan["rows"]) == len(prior["roster"]) == len(anchors) == 24,
         "fixed roster")
    banks = []
    for pair in range(8):
        triplet = anchors[3 * pair:3 * pair + 3]
        need(all(all(a[k] == triplet[0][k] for k in
                     ("root_slot", "condition", "arm", "task_id", "root_state", "goal"))
                 for a in triplet), "paired task")
        for offset, role in enumerate(("selected", "zero", "greedy")):
            i = 3 * pair + offset
            row, item, anchor, planned = replay["rows"][i], prior["roster"][i], anchors[i], plan["rows"][i]
            need(row["index"] == item["index"] == anchor["index"] == planned["index"] == i
                 and row["role"] == item["role"] == anchor["role"] == planned["role"] == role
                 and row["logical_pair"] == item["logical_pair"] == anchor["logical_pair"] == pair
                 and item["prefix5_float32_hex"] == anchor["prefix5_float32_hex"]
                 and row["winner"] == planned["selected_index"]
                 and row["scores"] == planned["candidate_scores"]
                 and row["candidate_sha256"][row["winner"]] == planned["selected_action_sha256"],
                 "source identity")
        row = replay["rows"][3 * pair]
        need(len(row["candidate_actions"]) == len(row["candidate_sha256"]) == len(row["scores"]) == 4,
             "candidate count")
        bank = []
        for index in range(4):
            action = np.asarray(row["candidate_actions"][index], dtype="<f4")
            digest = hashlib.sha256(action.tobytes()).hexdigest()
            need(action.shape == (2, 5, 2) and np.isfinite(action).all()
                 and (np.abs(action) <= 1).all() and digest == row["candidate_sha256"][index],
                 "candidate identity")
            bank.append({"index": index, "raw10_float32_hex": action.tobytes().hex(),
                         "raw10_sha256": digest, "winner": index == row["winner"]})
        banks.append(bank)
    bindings = dict(prior["bindings"])
    for path in (source.REPLAY, physical.bundle.PREP, PROTOCOL, Path(__file__).resolve(),
                 Path(validation.__file__).resolve(), Path(metric.__file__).resolve()):
        bindings[str(path)] = sha(path)
    need(all(sha(p) == h for p, h in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "anchors": anchors, "roster": prior["roster"],
            "banks": banks, "bindings": bindings, "fixed_branches": 64,
            "P1_admission": False, "independent_confirmation": False, "fairness_proven": False}


def fixed_branches(fixed):
    for pair in range(8):
        for offset in (1, 2):
            i = 3 * pair + offset
            for probe in fixed["banks"][pair]:
                item = copy.deepcopy(fixed["roster"][i])
                item.update(raw10_float32_hex=probe["raw10_float32_hex"],
                            raw10_sha256=probe["raw10_sha256"])
                yield fixed["anchors"][i], item, probe


def verify(prepared, digest):
    need(sha(prepared) == digest and json.loads(Path(prepared).read_bytes()) == build(), "prepare binding")
    return json.loads(Path(prepared).read_bytes())


def execute(prepared, digest, output):
    fixed = verify(prepared, digest)
    out = Path(output)
    need(out.is_absolute() and out.parent == ROOT / "results" and
         out.name.startswith(NAME + "_run_") and not out.exists() and not out.is_symlink()
         and shutil.disk_usage(out.parent).free >= 3 * 1024**3, "exclusive output and disk")
    report = physical.design.uphill.runner.native.read(physical.design.PHYS / "report.json", cap=64 * 1024**2)
    binding = report["execution_binding"]
    _, data = physical.design.uphill.verify(binding["prepared"], binding["prepare_sha256"],
                                             physical.design.JOIN_SHA)
    physical.design.uphill.runner.native.constructor_contract()
    out.mkdir(mode=0o700)

    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

    shards, failure = [], None
    try:
        for anchor, item, probe in fixed_branches(fixed):
            folder = out / f"anchor_{anchor['index']:02d}_probe_{probe['index']:02d}"
            result = physical.engine.execute(anchor, item, data, folder, factory)
            shard = {"anchor": anchor["index"], "probe": probe["index"], "status": result["status"],
                     "sha256": sha(folder / "report.json")}
            shards.append(shard)
            print(json.dumps(shard), flush=True)
            if shard["status"] != "recorded_not_consumed":
                failure = {"anchor": anchor["index"], "probe": probe["index"], "errors": result["errors"]}
                break
        need(all(sha(p) == h for p, h in fixed["bindings"].items()), "final source drift")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    status = "recorded_not_consumed" if len(shards) == 64 and failure is None else "partial_or_failed"
    value = {"schema": NAME + ".run", "status": status, "shards": shards, "failure": failure,
             "prepare_sha256": digest, "fixed_branches": 64, "P1_admission": False}
    return {"status": status, "sha256": physical.design.run._write_new(out / "run_report.json", value)}


def consume(prepared, digest, directory, report_sha):
    fixed = verify(prepared, digest)
    directory = Path(directory)
    path = directory / "run_report.json"
    need(sha(path) == report_sha, "report hash")
    report = json.loads(path.read_bytes())
    need(report["status"] == "recorded_not_consumed" and report["prepare_sha256"] == digest
         and len(report["shards"]) == 64, "complete run")
    rows = []
    for (anchor, item, probe), shard in zip(fixed_branches(fixed), report["shards"], strict=True):
        need(shard["anchor"] == anchor["index"] and shard["probe"] == probe["index"]
             and shard["status"] == "recorded_not_consumed", "shard identity")
        folder = directory / f"anchor_{anchor['index']:02d}_probe_{probe['index']:02d}"
        checked = physical.saved_physical.check_row(folder, anchor, item, shard["sha256"])
        saved = json.loads((folder / "report.json").read_bytes())["row"]
        responses = saved["responses"]
        validation.validate_responses(checked, responses)
        endpoint = None
        if len(responses) == 10:
            need(checked["status"] in ("probe_completed", "native_stopped"), "horizon status")
            frame = crossover.feedback.planner.source.earlier.frame_at(saved, 10, anchor["condition"])
            need(frame["state"] == responses[-1]["state"], "horizon frame")
            endpoint = metric.physical_metrics(responses[-1]["state"], anchor["goal"])
        rows.append({"anchor": anchor["index"], "logical_pair": anchor["logical_pair"],
                     "root_slot": anchor["root_slot"], "condition": anchor["condition"], "arm": anchor["arm"],
                     "role": anchor["role"], "candidate_index": probe["index"], "winner": probe["winner"],
                     "probe_sha256": probe["raw10_sha256"], "status": checked["status"],
                     "returned_suffix": checked["returned_suffix"], "accepted_suffix_steps": len(responses),
                     "native_success": checked["last_accepted_state_metrics"]["native_success"],
                     "root": metric.physical_metrics(anchor["root_state"], anchor["goal"]),
                     "prefix": metric.physical_metrics(anchor["checkpoint"]["state"], anchor["goal"]),
                     "endpoint": endpoint, "last_accepted": checked["last_accepted_state_metrics"]})
    need(sha(path) == report_sha and all(sha(p) == h for p, h in fixed["bindings"].items()), "final drift")
    return {"schema": NAME + ".consume", "status": "saved_checks_passed", "rows": rows,
            "fixed_branches": 64, "input_prepare_sha256": digest, "input_report_sha256": report_sha,
            "P1_admission": False, "independent_confirmation": False, "fairness_proven": False}


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
    if args.command == "run":
        need(args.authorize_physics, "physics authorization")
        result = execute(args.prepared, args.sha256, out)
    else:
        need(out.is_absolute() and out.parent == ROOT / "results" and
             out.name.startswith(NAME + "_" + args.command + "_") and not out.exists()
             and not out.is_symlink(), "output")
        value = build() if args.command == "prepare" else consume(
            args.prepared, args.sha256, args.directory, args.report_sha)
        result = {"sha256": physical.design.run._write_new(out, value)}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
