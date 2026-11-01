"""Matched 42-suffix exploration for the three new diverse prefix slots."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from . import p1_diverse_physical_runner_20261030_main as physical
from . import codex_p1_three_prefix_h10_prepare_20261030 as design
from . import codex_p1_three_prefix_h10_runner_20261030 as runner
from . import codex_p1_three_prefix_h10_saved_20261030 as consumer

ROOT = physical.ROOT
NAME = "p1_diverse_h10_20261030_main"
DIRECTORY = ROOT / "results/p1_diverse_physical_runner_20261030_main_RUN01"
REPORT_SHA = "7d72e84253f746bd23071b0283cacdc9494ef2f2a2f95b9d6f460a6ec3eec6f4"
JOIN_SHA = "bbf2bdb0fcb02308e7a98615c8a526c612833e9afd83a5c64cdb42259256a06f"
OLD = ROOT / "results/codex_p1_three_prefix_h10_prepare_20261030_CODEX01.json"
OLD_SHA = "2f19d026c39602db743eea8191d563e820979d682c31a1b32f9bcb40e3bb1852"
PROTOCOL = ROOT.parent / "docs/D2P_P1_DIVERSE_H10_PROTOCOL_20261030_MAIN.md"
SLOTS = ((2, 5), (6, 13), (7, 15))
need, sha, native = runner.require, physical.runner.sha, physical.runner.native


def build():
    need(sha(DIRECTORY / "report.json") == REPORT_SHA and sha(OLD) == OLD_SHA, "source report SHA")
    report = native.read(DIRECTORY / "report.json", cap=64*1024**2)
    binding = report["execution_binding"]
    prepared, data = physical.verify(binding["prepared"], binding["prepare_sha256"], JOIN_SHA)
    need(binding == native.read(DIRECTORY / "execution_binding.json"), "execution binding")
    physical.consumer.check(DIRECTORY, report, prepared)
    events = physical.consumer.events(DIRECTORY / "journal.jsonl", report["journal_chain_tail"])
    old = native.read(OLD, cap=64*1024**2)
    anchors, banks = [], []
    bindings = copy.deepcopy(prepared["bindings"])
    paths = [DIRECTORY / "report.json", DIRECTORY / "journal.jsonl",
             DIRECTORY / "execution_binding.json", Path(binding["prepared"]), OLD,
             Path(__file__).resolve(), PROTOCOL, *runner.source_files(),
             Path(design.__file__).resolve(), Path(consumer.__file__).resolve()]
    for path in paths:
        digest = sha(path)
        need(str(path) not in bindings or bindings[str(path)] == digest, "source collision")
        bindings[str(path)] = digest
    for logical, i in SLOTS:
        row, plan = report["rows"][i], prepared["plans"][i]
        old_selected = old["anchors"][3 * logical]
        old_zero, old_greedy = old["anchors"][3 * logical + 1:3 * logical + 3]
        need(row["index"] == i and plan["role"] == "model_selected"
             and row["own_returned"] >= 5, "new prefix source")
        prefix = bytes.fromhex(plan["raw15_float32_hex"])[:40]
        for prior in (old_selected, old_zero, old_greedy):
            need(all(prior[k] == plan[k] for k in ("root_slot", "condition", "arm", "task_id",
                                                     "physical_root_sha256")), "paired task identity")
        need(old_selected["prefix5_float32_hex"] != prefix.hex()
             and old_zero["prefix5_float32_hex"] == bytes(40).hex(), "different selected and zero")
        trace = [e for e in events if e["kind"] == "fresh_observation" and e.get("branch") == i
                 and (e["receipt"]["phase"] == "warmup" or
                      (e["receipt"]["phase"] == "plan" and e["receipt"]["step"] <= 5))]
        expected = [(phase, step, overlay) for phase, count in (("warmup", 11), ("plan", 6))
                    for step in range(count) for overlay in range(3)]
        need([(e["receipt"]["phase"], e["receipt"]["step"], e["receipt"]["overlay_index"])
              for e in trace] == expected, "51 source frames")
        for event in trace:
            rec = event["receipt"]
            path = native.paired.flat(DIRECTORY, rec["png"]).resolve()
            need(event["pre"] == event["post"] and rec["snapshot"] == event["post"]["snapshot"]
                 and rec["state"] == rec["snapshot"]["state"] and sha(path) == rec["file_sha256"],
                 "source frame identity")
            bindings[str(path)] = rec["file_sha256"]
        anchor = dict(index=3*logical, logical_pair=logical, root_slot=plan["root_slot"],
                      condition=plan["condition"], arm=plan["arm"], role="selected", task_id=plan["task_id"],
                      physical_root_sha256=plan["physical_root_sha256"], prefix5_float32_hex=prefix.hex(),
                      prefix5_sha256=hashlib.sha256(prefix).hexdigest(), root_state=row["root_state"],
                      goal=row["goal"], checkpoint=next(c for c in row["checkpoints"] if c["step"] == 5),
                      trace=[{k: copy.deepcopy(e[k]) for k in ("receipt", "pre", "post")} for e in trace])
        bank = design.bank(bytes.fromhex(plan["raw15_float32_hex"])[40:])
        need(bank[:41] == old["probe_banks"][logical][:41], "same fixed common probe bytes")
        anchors.append(anchor)
        banks.append(bank)
    need(all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    value = dict(schema=NAME+".prepare", anchors=anchors, probe_banks=banks, bindings=bindings,
                 source_report_sha256=REPORT_SHA, old_prepare_sha256=OLD_SHA, fixed_branches=126,
                 common_control_probes=41, last_probe_control_executed=False, P1_admission=False,
                 independent_confirmation=False, equal_budget_certified=False, new_neural_calls=0)
    return value, data


def verify(path, digest):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink() and sha(path) == digest, "prepare SHA")
    value = native.read(path, cap=64*1024**2)
    current, data = build()
    need(value == current and sha(path) == digest, "prepare drift")
    return value, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--report-sha")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    path = Path(args.output)
    if args.command == "prepare":
        need(path.is_absolute() and path.parent == ROOT / "results" and path.name.startswith(NAME+"_prepare_")
             and not path.exists(), "exclusive prepare")
        value, _ = build()
        print(json.dumps({"sha256": runner._write_new(path, value), "fixed_branches": 126}))
        return
    p, data = verify(args.prepared, args.prepare_sha)
    if args.command == "run":
        need(args.authorize_physics and path.is_absolute() and path.parent == ROOT / "results"
             and path.name.startswith(NAME+"_run_") and not path.exists()
             and shutil.disk_usage(path.parent).free >= runner.FLOOR + 3*runner.CAP, "run authorization/reserve")
        native.constructor_contract()
        path.mkdir()
        runner.base.paired.fsync_dir(path.parent)
        bindings = dict(p["bindings"], **{args.prepared: args.prepare_sha})
        binding = dict(prepared=args.prepared, prepare_sha256=args.prepare_sha, bindings=bindings,
                       fixed_branches=126, P1_admission=False, independent_confirmation=False)
        binding_sha = runner._write_new(path / "run_binding.json", binding)

        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

        shards = []
        print("Executing three fixed anchors with 42 matched ten-step suffixes each.", flush=True)
        for anchor, bank in zip(p["anchors"], p["probe_banks"], strict=True):
            shard_path = path / f"anchor_{anchor['index']:02d}"
            result = runner.execute_shard(anchor, bank, data, shard_path, factory, bindings)
            shard = dict(anchor=anchor["index"], status=result["status"],
                         sha256=sha(shard_path / "report.json"))
            shards.append(shard)
            print(json.dumps(shard), flush=True)
            if result["status"] != "recorded_not_consumed":
                break
        need(all(sha(file) == digest for file, digest in bindings.items()), "final source drift")
        complete = len(shards) == 3 and all(s["status"] == "recorded_not_consumed" for s in shards)
        result = dict(schema=NAME+".run", status="recorded_not_consumed" if complete else "partial_or_failed",
                      shards=shards, run_binding_sha256=binding_sha, fixed_branches=126,
                      P1_admission=False, independent_confirmation=False)
        digest = runner._write_new(path / "run_report.json", result)
        print(json.dumps({"sha256": digest, "status": result["status"]}))
        if not complete:
            raise SystemExit(1)
    else:
        report_path = path / "run_report.json"
        need(path.is_absolute() and path.is_dir() and not path.is_symlink()
             and sha(report_path) == args.report_sha, "run report identity")
        report = native.read(report_path, cap=4*1024**2)
        binding = native.read(path / "run_binding.json", cap=64*1024**2)
        need(report["schema"] == NAME+".run" and report["status"] == "recorded_not_consumed"
             and report["fixed_branches"] == 126 and len(report["shards"]) == 3
             and sha(path / "run_binding.json") == report["run_binding_sha256"]
             and binding == dict(prepared=args.prepared, prepare_sha256=args.prepare_sha,
                                 bindings=dict(p["bindings"], **{args.prepared: args.prepare_sha}),
                                 fixed_branches=126, P1_admission=False, independent_confirmation=False),
             "run source binding")
        checked = []
        for anchor, bank, shard in zip(p["anchors"], p["probe_banks"], report["shards"], strict=True):
            need(shard["anchor"] == anchor["index"] and shard["status"] == "recorded_not_consumed", "shard")
            checked.append(consumer.check_shard(path / f"anchor_{anchor['index']:02d}",
                                                anchor, bank, shard["sha256"]))
        need(all(sha(file) == digest for file, digest in binding["bindings"].items()), "final source drift")
        print(json.dumps({"saved_checks_passed": True, "rows": sum(len(x["statuses"]) for x in checked),
                          "statuses": {s: sum(x["statuses"].count(s) for x in checked)
                                       for s in ("probe_completed", "native_stopped", "censored_out_of_domain")},
                          "P1_admission": False}))


if __name__ == "__main__":
    main()
