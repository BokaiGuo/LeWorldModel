"""Run the sealed common H10 bank across all uncensored response-prefix anchors."""
import argparse
import json
import shutil
from pathlib import Path

from . import p1_response_prefix_cp5_physical_20261031 as physical
from . import codex_p1_three_prefix_h10_runner_20261030 as engine
from . import codex_p1_three_prefix_h10_saved_20261030 as saved

ROOT, need, sha = physical.ROOT, physical.need, physical.sha
NAME = "p1_response_prefix_h10_20261031"
INPUT = ROOT / "results/p1_response_prefix_cp5_physical_20261031_prepare_RUN01.json"
INPUT_SHA = "0aa3b133b22e58e1673fb546ef8a4a7679a8c17c5b652cae9fbabc6c012c2102"
BANKS = ROOT / "results/p1_uphill_h10_20261030_prepare_RUN01.json"
BANKS_SHA = "890e38ecf7758ae739c25e9d355533e4d3b31be91a8c063c9625d1886c77d56d"
PROTOCOL = ROOT.parent / "docs/D2P_P1_RESPONSE_PREFIX_H10_PROTOCOL_20261031.md"


def build():
    read = physical.bundle.physical.prior.native.read
    need(sha(INPUT) == INPUT_SHA and sha(BANKS) == BANKS_SHA, "fixed source hash")
    source, previous = read(INPUT, cap=64 * 1024**2), read(BANKS, cap=64 * 1024**2)
    need(source["fixed_rows"] == 16 and len(source["anchors"]) == 14 and
         len(source["censored_before_cp5"]) == 2 and len(previous["probe_banks"]) == 8 and
         len(previous["anchors"]) == 24, "source roster")
    bindings = dict(source["bindings"])
    for path in (INPUT, BANKS, PROTOCOL, Path(__file__).resolve(), Path(engine.__file__).resolve(),
                 Path(saved.__file__).resolve()):
        digest = sha(path)
        need(str(path) not in bindings or bindings[str(path)] == digest, "source collision")
        bindings[str(path)] = digest
    banks = previous["probe_banks"]
    for anchor in source["anchors"]:
        logical = anchor["logical_pair"]
        reference = previous["anchors"][3 * logical]
        need(all(anchor[k] == reference[k] for k in ("root_slot", "condition", "arm", "task_id",
                                                       "root_state", "goal"))
             and len(banks[logical]) == 42 and
             [x["index"] for x in banks[logical]] == list(range(42)), "common task and bank identity")
    need(all(sha(p) == d for p, d in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "anchors": source["anchors"], "banks": banks,
            "bindings": bindings, "censored_before_cp5": source["censored_before_cp5"],
            "fixed_rows": 16, "fixed_branches": 588, "P1_admission": False,
            "independent_confirmation": False}


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
    current = build()
    if args.command == "prepare":
        result = {"sha256": physical.bundle.physical.write(out, current), "branches": 588}
    else:
        need(args.prepared and sha(args.prepared) == args.sha256 and
             physical.bundle.physical.prior.native.read(args.prepared, cap=64 * 1024**2) == current,
             "prepared/source rebuild")
        if args.command == "run":
            need(args.authorize_physics and shutil.disk_usage(out.parent).free >= engine.FLOOR + 14 * engine.CAP,
                 "physics authorization and full disk reserve")
            _, data = physical.bundle.physical.pilot.material(physical.bundle.physical.JOIN_SHA)
            physical.bundle.physical.prior.native.constructor_contract()
            out.mkdir(mode=0o700)
            def factory():
                import gymnasium as gym
                import stable_worldmodel  # noqa: F401
                return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
            shards = []
            bindings = dict(current["bindings"], **{str(Path(args.prepared)): args.sha256})
            for anchor in current["anchors"]:
                folder = out / f"anchor_{anchor['index']:02d}"
                report = engine.execute_shard(anchor, current["banks"][anchor["logical_pair"]],
                                               data, folder, factory, bindings)
                shards.append({"index": anchor["index"], "status": report["status"],
                               "sha256": sha(folder / "report.json")})
                print(json.dumps(shards[-1]), flush=True)
                if report["status"] != "recorded_not_consumed":
                    break
            need(all(sha(p) == d for p, d in bindings.items()), "source drift")
            status = "recorded_not_consumed" if len(shards) == 14 and all(
                x["status"] == "recorded_not_consumed" for x in shards) else "partial_or_failed"
            value = {"schema": NAME + ".run", "status": status, "shards": shards,
                     "prepare_sha256": args.sha256, "fixed_branches": 588,
                     "censored_before_cp5": current["censored_before_cp5"], "P1_admission": False}
            result = {"status": status, "sha256": physical.bundle.physical.write(out / "run_report.json", value)}
        else:
            directory = Path(args.directory)
            need(directory.is_absolute() and directory.is_dir() and not directory.is_symlink() and
                 sha(directory / "run_report.json") == args.report_sha, "run report hash")
            report = physical.bundle.physical.prior.native.read(directory / "run_report.json")
            need(report["status"] == "recorded_not_consumed" and report["prepare_sha256"] == args.sha256
                 and len(report["shards"]) == 14, "complete shard roster")
            rows = []
            for anchor, shard in zip(current["anchors"], report["shards"], strict=True):
                need(shard["index"] == anchor["index"] and shard["status"] == "recorded_not_consumed",
                     "shard identity")
                rows.append(saved.check_shard(directory / f"anchor_{anchor['index']:02d}", anchor,
                                               current["banks"][anchor["logical_pair"]], shard["sha256"]))
            value = {"schema": NAME + ".consume", "status": "saved_checks_passed", "rows": rows,
                     "fixed_branches": 588, "censored_before_cp5": current["censored_before_cp5"],
                     "P1_admission": False, "independent_confirmation": False}
            result = {"status": value["status"], "rows": len(rows),
                      "sha256": physical.bundle.physical.write(out, value)}
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
