"""Bind and execute all 24 independently planned joint H10 suffixes."""
import argparse
import json
import os
import shutil
from pathlib import Path

from . import p1_joint_h10_20261030 as design
from . import p1_joint_cp5_bundle_20261030 as bundle
from . import p1_joint_cp5_cuda_20261030 as cuda
from . import codex_p1_cp5_exploratory_saved_20261030_v1 as saved_plan
from . import codex_p1_cp5_physical_24_20261030_v1 as engine
from . import codex_p1_cp5_physical_saved_20261030_v1 as saved_physical

ROOT = design.ROOT
NAME = "p1_joint_cp5_physical_20261030"
PLAN_DIR = ROOT / "results/p1_joint_cp5_cuda_20261030_RUN01"
MANIFEST_SHA = "0203745431777190fc25ac43babcf6e265ea50eb9be0e79be874bc19335d55ae"
REPORT_SHA = "20bc0c9af23c6625bd2708adb1bc26af96bf147ea36474691181717fc244577e"
need, sha = engine.need, engine.sha


def configure():
    cuda.planner.BUNDLE, cuda.planner.BUNDLE_SHA = bundle.OUT, cuda.BUNDLE_SHA
    cuda.planner.NAME = cuda.NAME
    cuda.planner.bindings = cuda.bindings
    engine.NAME = NAME


def build():
    configure()
    fixed = design.joint.runner.native.read(bundle.PREP, cap=64*1024**2)
    need(sha(bundle.PREP) == bundle.PREP_SHA and fixed == design.build(), "H10 source")
    receipt = saved_plan.consume(PLAN_DIR, MANIFEST_SHA, REPORT_SHA)
    manifest = json.loads((PLAN_DIR / "manifest.json").read_bytes())
    need(all(sha(k) == v for k, v in manifest["sources"].items()), "planner source drift")
    blind = json.loads(bundle.OUT.read_bytes())
    events = []
    for item in receipt["rows"]:
        path = PLAN_DIR / f"row_{item['index']:02d}_result.json"
        need(sha(path) == item["row_sha256"], "plan row SHA")
        events.append(json.loads(path.read_bytes()))
    roster = engine.make_roster(fixed["anchors"], blind["rows"], events, receipt)
    bindings = dict(fixed["bindings"])
    paths = [Path(__file__).resolve(), Path(engine.__file__).resolve(),
             Path(saved_physical.__file__).resolve(), Path(saved_plan.__file__).resolve(),
             bundle.PREP, bundle.OUT, PLAN_DIR / "manifest.json", PLAN_DIR / "report.json",
             design.PROTOCOL, cuda.PROTOCOL]
    for event in receipt["rows"]:
        paths.append(PLAN_DIR / f"row_{event['index']:02d}_result.json")
    for path in paths:
        digest = sha(path)
        need(str(path) not in bindings or bindings[str(path)] == digest, "source collision")
        bindings[str(path)] = digest
    for path, digest in manifest["sources"].items():
        need(path not in bindings or bindings[path] == digest, "planner binding collision")
        bindings[path] = digest
    need(all(sha(k) == v for k, v in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "roster": roster, "bindings": bindings,
            "plan_manifest_sha256": MANIFEST_SHA, "plan_report_sha256": REPORT_SHA,
            "fixed_rows": 24, "P1_admission": False, "independent_confirmation": False,
            "fairness_proven": False, "os_isolation_certified": False}


def verify(path, digest):
    need(sha(path) == digest, "prepare hash")
    value = design.joint.runner.native.read(path, cap=64*1024**2)
    need(value == build() and sha(path) == digest, "source rebuild")
    return value


def execute(prepared, digest, output):
    p = verify(prepared, digest)
    report = design.joint.runner.native.read(design.PHYS / "report.json", cap=64*1024**2)
    binding = report["execution_binding"]
    _, data = design.joint.verify(binding["prepared"], binding["prepare_sha256"], design.JOIN_SHA)
    anchors = design.joint.runner.native.read(bundle.PREP, cap=64*1024**2)["anchors"]
    design.joint.runner.native.constructor_contract()
    directory = Path(output)
    need(directory.is_absolute() and directory.parent == ROOT / "results"
         and directory.name.startswith(NAME + "_run_") and not os.path.lexists(directory)
         and shutil.disk_usage(directory.parent).free >= 2*1024**3, "exclusive output/disk")
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
    result = {"schema": NAME + ".run", "status": "recorded_not_consumed" if len(shards) == 24 and not failure
              else "partial_or_failed", "shards": shards, "failure": failure, "binding_sha256": binding_sha,
              "prepare_sha256": digest, "fixed_rows": 24, "P1_admission": False}
    return {"status": result["status"], "sha256": design.run._write_new(directory / "run_report.json", result)}


def consume(directory, report_sha, prepared, digest):
    p = verify(prepared, digest)
    directory = Path(directory)
    path = directory / "run_report.json"
    need(sha(path) == report_sha, "report hash")
    report = design.joint.runner.native.read(path, cap=4*1024**2)
    need(report["status"] == "recorded_not_consumed" and len(report["shards"]) == 24
         and report["prepare_sha256"] == digest
         and sha(directory / "binding.json") == report["binding_sha256"], "complete run")
    anchors = design.joint.runner.native.read(bundle.PREP, cap=64*1024**2)["anchors"]
    rows = []
    for i, (anchor, item, shard) in enumerate(zip(anchors, p["roster"], report["shards"], strict=True)):
        need(shard["index"] == i and shard["status"] == "recorded_not_consumed", "shard order")
        rows.append(saved_physical.check_row(directory / f"row_{i:02d}", anchor, item, shard["report_sha256"]))
    need(sha(path) == report_sha and all(sha(k) == v for k, v in p["bindings"].items()), "final drift")
    return {"schema": NAME + ".consume", "rows": rows, "fixed_rows": 24, "P1_admission": False,
            "independent_confirmation": False, "fairness_proven": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "run", "consume"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--prepared")
    ap.add_argument("--sha256")
    ap.add_argument("--report-sha")
    ap.add_argument("--directory")
    ap.add_argument("--authorize-physics", action="store_true")
    a = ap.parse_args()
    out = Path(a.output)
    if a.command == "prepare":
        need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_prepare_"),
             "prepare path")
        result = {"sha256": design.run._write_new(out, build())}
    elif a.command == "run":
        need(a.authorize_physics, "physical authorization")
        result = execute(a.prepared, a.sha256, a.output)
    else:
        need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_consume_"),
             "consume path")
        result = consume(a.directory, a.report_sha, a.prepared, a.sha256)
        result = {"sha256": design.run._write_new(out, result), "rows": len(result["rows"])}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
