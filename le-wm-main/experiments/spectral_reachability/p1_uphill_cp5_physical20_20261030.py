"""Verify and physically execute the 24 uphill cp5 CUDA plans."""
import argparse
import json
import os
import shutil
from pathlib import Path

from . import p1_uphill_h10_20261030 as design
from . import p1_uphill_cp5_bundle_20261030 as bundle
from . import p1_uphill_cp5_cuda20_20261030 as cuda
from . import codex_p1_cp5_exploratory_saved_20261030_v1 as saved_plan
from . import codex_p1_cp5_physical_24_20261030_v1 as engine
from . import codex_p1_cp5_physical_saved_20261030_v1 as saved_physical

ROOT = design.ROOT
NAME = "p1_uphill_cp5_physical20_20261030"
PLAN_DIR = ROOT / "results/p1_uphill_cp5_cuda20_20261030_RUN01"
PREP_SHA = "890e38ecf7758ae739c25e9d355533e4d3b31be91a8c063c9625d1886c77d56d"
BUNDLE_SHA = "376bb8276e1544010ea2767d9c46f141c2665e223154a12b2e6c28786636676a"
MANIFEST_SHA = "f732458eeaaf01bb2425a04157f8bd060de2a10a373483ac080b800891b3949d"
REPORT_SHA = "9fa1a338b73e38334cf0ed6e8be11091a964b53ceea21f6c0c3933f9144a0638"
need, sha = engine.need, engine.sha


def configure():
    planner = cuda.planner
    planner.BUNDLE, planner.BUNDLE_SHA, planner.NAME = bundle.OUT, BUNDLE_SHA, cuda.NAME
    planner.CONFIG = {"steps": 20, "learning_rate": 0.05, "history_size": 3}
    planner.bindings, planner.make_contract = cuda.bindings, cuda.prior.make_contract
    engine.NAME = NAME


def build():
    configure()
    fixed = design.uphill.runner.native.read(bundle.PREP, cap=64 * 1024**2)
    need(sha(bundle.PREP) == PREP_SHA and fixed == design.build(), "H10 source")
    receipt = saved_plan.consume(PLAN_DIR, MANIFEST_SHA, REPORT_SHA)
    manifest = json.loads((PLAN_DIR / "manifest.json").read_bytes())
    need(all(sha(k) == v for k, v in manifest["sources"].items()), "planner source drift")
    need(sha(bundle.OUT) == BUNDLE_SHA, "bundle hash")
    blind = json.loads(bundle.OUT.read_bytes())
    events = []
    for item in receipt["rows"]:
        path = PLAN_DIR / f"row_{item['index']:02d}_result.json"
        need(sha(path) == item["row_sha256"], "plan row hash")
        events.append(json.loads(path.read_bytes()))
    roster = engine.make_roster(fixed["anchors"], blind["rows"], events, receipt)
    bindings = dict(fixed["bindings"])
    paths = [Path(__file__).resolve(), Path(engine.__file__).resolve(),
             Path(saved_physical.__file__).resolve(), Path(saved_plan.__file__).resolve(),
             bundle.PREP, bundle.OUT, PLAN_DIR / "manifest.json", PLAN_DIR / "report.json",
             design.PROTOCOL, cuda.PROTOCOL]
    paths += [PLAN_DIR / f"row_{event['index']:02d}_result.json" for event in receipt["rows"]]
    for path in paths:
        digest = sha(path)
        need(str(path) not in bindings or bindings[str(path)] == digest, "source collision")
        bindings[str(path)] = digest
    for path, digest in manifest["sources"].items():
        need(path not in bindings or bindings[path] == digest, "planner source collision")
        bindings[path] = digest
    need(all(sha(k) == v for k, v in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "roster": roster, "bindings": bindings,
            "plan_manifest_sha256": MANIFEST_SHA, "plan_report_sha256": REPORT_SHA,
            "fixed_rows": 24, "P1_admission": False, "independent_confirmation": False,
            "fairness_proven": False, "os_isolation_certified": False}


def verify(path, digest):
    need(sha(path) == digest, "prepare hash")
    value = design.uphill.runner.native.read(path, cap=64 * 1024**2)
    need(value == build() and sha(path) == digest, "source rebuild")
    return value


def execute(prepared, digest, output):
    p = verify(prepared, digest)
    report = design.uphill.runner.native.read(design.PHYS / "report.json", cap=64 * 1024**2)
    binding = report["execution_binding"]
    _, data = design.uphill.verify(binding["prepared"], binding["prepare_sha256"], design.JOIN_SHA)
    anchors = design.uphill.runner.native.read(bundle.PREP, cap=64 * 1024**2)["anchors"]
    design.uphill.runner.native.constructor_contract()
    directory = Path(output)
    need(directory.is_absolute() and directory.parent == ROOT / "results"
         and directory.name.startswith(NAME + "_run_") and not os.path.lexists(directory)
         and shutil.disk_usage(directory.parent).free >= 2 * 1024**3, "exclusive output/disk")
    directory.mkdir(mode=0o700)
    binding_sha = design.run._write_new(directory / "binding.json",
                                       {"prepared": prepared, "prepare_sha256": digest, "fixed_rows": 24})

    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

    shards, failure = [], None
    try:
        for anchor, item in zip(anchors, p["roster"], strict=True):
            path = directory / f"row_{anchor['index']:02d}"
            result = engine.execute(anchor, item, data, path, factory)
            shards.append({"index": anchor["index"], "report_sha256": sha(path / "report.json"),
                           "status": result["status"]})
            print(json.dumps(shards[-1]), flush=True)
            if result["status"] != "recorded_not_consumed":
                failure = {"index": anchor["index"], "errors": result["errors"]}
                break
        need(all(sha(k) == v for k, v in p["bindings"].items()), "final source drift")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    result = {"schema": NAME + ".run", "status": "recorded_not_consumed"
              if len(shards) == 24 and not failure else "partial_or_failed",
              "shards": shards, "failure": failure, "binding_sha256": binding_sha,
              "prepare_sha256": digest, "fixed_rows": 24, "P1_admission": False}
    return {"status": result["status"], "sha256": design.run._write_new(directory / "run_report.json", result)}


def consume(directory, report_sha, prepared, digest):
    p = verify(prepared, digest)
    directory = Path(directory)
    path = directory / "run_report.json"
    need(sha(path) == report_sha, "report hash")
    report = design.uphill.runner.native.read(path, cap=4 * 1024**2)
    need(report["status"] == "recorded_not_consumed" and len(report["shards"]) == 24
         and report["prepare_sha256"] == digest
         and sha(directory / "binding.json") == report["binding_sha256"], "complete run")
    anchors = design.uphill.runner.native.read(bundle.PREP, cap=64 * 1024**2)["anchors"]
    rows = []
    for i, (anchor, item, shard) in enumerate(zip(anchors, p["roster"], report["shards"], strict=True)):
        need(shard["index"] == i and shard["status"] == "recorded_not_consumed", "shard order")
        rows.append(saved_physical.check_row(directory / f"row_{i:02d}", anchor, item,
                                             shard["report_sha256"]))
    need(sha(path) == report_sha and all(sha(k) == v for k, v in p["bindings"].items()), "final drift")
    return {"schema": NAME + ".consume", "rows": rows, "fixed_rows": 24, "P1_admission": False,
            "independent_confirmation": False, "fairness_proven": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--report-sha")
    parser.add_argument("--directory")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    if args.command == "prepare":
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(NAME + "_prepare_"), "prepare path")
        result = {"sha256": design.run._write_new(out, build())}
    elif args.command == "run":
        need(args.authorize_physics, "physics authorization")
        result = execute(args.prepared, args.sha256, args.output)
    else:
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(NAME + "_consume_"), "consume path")
        value = consume(args.directory, args.report_sha, args.prepared, args.sha256)
        result = {"sha256": design.run._write_new(out, value), "rows": len(value["rows"])}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
