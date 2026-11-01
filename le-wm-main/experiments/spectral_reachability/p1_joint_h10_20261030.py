"""Fixed 24-anchor H10 exploration for joint-selected prefixes."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from . import p1_joint_physical_20261030 as joint
from . import codex_p1_three_prefix_h10_runner_20261030 as run
from . import codex_p1_three_prefix_h10_saved_20261030 as saved
from . import p1_backtrack_physical_consume_20261030 as events

ROOT = joint.ROOT
NAME = "p1_joint_h10_20261030"
OLD = ROOT / "results/codex_p1_three_prefix_h10_runner_20261030_prepare_CODEX01.json"
OLD_SHA = "ee136f0a7537c8328aa4ec7417e6d4d0e0c1324c50dc5f59bc4e0c871f020f30"
PHYS = ROOT / "results/p1_joint_physical_20261030_RUN01"
PHYS_SHA = "3ff76885ec7eaf68411355ac278b658cabd22b10b4940bdbf73e7d12c48ff404"
JOIN_SHA = "c951fbe737cf8e38b3e13341aeaed80968a8f866c6eef78fdf29f67badd6e908"
PROTOCOL = ROOT.parent / "docs/D2P_P1_JOINT_H10_PROTOCOL_20261030_MAIN.md"
require, sha = joint.runner.require, joint.runner.sha


def build():
    old = run.verify_prepare(OLD, OLD_SHA)
    report = joint.runner.native.read(PHYS / "report.json", cap=64*1024**2)
    require(sha(PHYS / "report.json") == PHYS_SHA, "physical report")
    binding = report["execution_binding"]
    physical, _ = joint.verify(binding["prepared"], binding["prepare_sha256"], JOIN_SHA)
    joint.consumer.check(PHYS, report, physical)
    journal = events.events(PHYS / "journal.jsonl", report["journal_chain_tail"])
    bindings = copy.deepcopy(old["bindings"])
    for path in (OLD, PHYS / "report.json", PHYS / "journal.jsonl", PHYS / "execution_binding.json",
                 Path(binding["prepared"]), Path(__file__).resolve(), PROTOCOL,
                 Path(saved.__file__).resolve(), Path(joint.__file__).resolve()):
        digest = sha(path)
        require(str(path) not in bindings or bindings[str(path)] == digest, "binding collision")
        bindings[str(path)] = digest
    for path, digest in physical["bindings"].items():
        require(path not in bindings or bindings[path] == digest, "physical source collision")
        bindings[path] = digest
    anchors, banks = [], []
    for logical in range(8):
        row, plan = report["rows"][2*logical+1], physical["plans"][2*logical+1]
        old_triplet = old["original"]["anchors"][3*logical:3*logical+3]
        require(row["role"] == plan["role"] == "model_selected"
                and all(row[k] == plan[k] == old_triplet[0][k]
                        for k in ("root_slot", "condition", "arm", "task_id"))
                and row["root_state"] == old_triplet[0]["root_state"]
                and row["goal"] == old_triplet[0]["goal"], "anchor identity")
        trace = [e for e in journal if e["kind"] == "fresh_observation" and e.get("branch") == row["index"]
                 and (e["receipt"]["phase"] == "warmup" or
                      (e["receipt"]["phase"] == "plan" and e["receipt"]["step"] <= 5))]
        schedule = [(phase, step, overlay) for phase, count in (("warmup", 11), ("plan", 6))
                    for step in range(count) for overlay in range(3)]
        require([(e["receipt"]["phase"], e["receipt"]["step"], e["receipt"]["overlay_index"])
                 for e in trace] == schedule, "trace roster")
        for event in trace:
            rec = event["receipt"]
            path = PHYS / rec["png"]
            require(path.parent == PHYS and path.is_file() and not path.is_symlink()
                    and sha(path) == rec["file_sha256"], "source frame")
            bindings[str(path)] = rec["file_sha256"]
        raw = bytes.fromhex(plan["raw15_float32_hex"])
        require(len(raw) == 120, "literal action length")
        selected = copy.deepcopy(old_triplet[0])
        selected.update(prefix5_float32_hex=raw[:40].hex(), prefix5_sha256=hashlib.sha256(raw[:40]).hexdigest(),
                        checkpoint=next(c for c in row["checkpoints"] if c["step"] == 5),
                        trace=[{k: copy.deepcopy(e[k]) for k in ("receipt", "pre", "post")} for e in trace],
                        source_family="joint_cuda_physical_RUN01")
        for role, anchor in zip(("selected", "zero", "greedy"),
                                (selected, old_triplet[1], old_triplet[2]), strict=True):
            anchor = copy.deepcopy(anchor)
            require(anchor["role"] == role and anchor["logical_pair"] == logical
                    and all(anchor[k] == selected[k] for k in
                            ("root_slot", "condition", "arm", "task_id", "physical_root_sha256", "goal", "root_state")),
                    "triplet binding")
            anchor["index"] = len(anchors)
            anchors.append(anchor)
        banks.append(run.design.bank(raw[40:]))
    require(len(anchors) == 24 and len(banks) == 8 and all(len(x) == 42 for x in banks), "full roster")
    require(all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "anchors": anchors, "probe_banks": banks,
            "bindings": bindings, "fixed_anchors": 24, "fixed_branches": 1008,
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False}


def prepare(output):
    out = Path(output)
    require(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_prepare_"),
            "prepare path")
    return run._write_new(out, build())


def execute(path, digest, output):
    path = Path(path)
    require(sha(path) == digest, "prepare hash")
    p = joint.runner.native.read(path, cap=64*1024**2)
    require(p == build(), "prepare drift")
    _, data = joint.verify(
        joint.runner.native.read(PHYS / "report.json", cap=64*1024**2)["execution_binding"]["prepared"],
        joint.runner.native.read(PHYS / "report.json", cap=64*1024**2)["execution_binding"]["prepare_sha256"],
        JOIN_SHA)
    joint.runner.native.constructor_contract()
    directory = Path(output)
    require(directory.is_absolute() and directory.parent == ROOT / "results"
            and directory.name.startswith(NAME + "_run_") and not directory.exists()
            and shutil.disk_usage(directory.parent).free >= run.FLOOR + 24*run.CAP, "output reservation")
    directory.mkdir()
    run.base.paired.fsync_dir(directory.parent)
    bindings = dict(p["bindings"], **{str(path): digest})
    run._write_new(directory / "run_binding.json", {"prepare": str(path), "prepare_sha256": digest,
                                                     "bindings": bindings, "fixed_branches": 1008})
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
    shards = []
    for anchor in p["anchors"]:
        bank = p["probe_banks"][anchor["logical_pair"]]
        target = directory / f"anchor_{anchor['index']:02d}"
        result = run.execute_shard(anchor, bank, data, target, factory, bindings)
        report_sha = sha(target / "report.json")
        shards.append({"anchor": anchor["index"], "status": result["status"], "sha256": report_sha})
        print(json.dumps(shards[-1]), flush=True)
        if result["status"] != "recorded_not_consumed":
            break
    require(all(sha(k) == v for k, v in bindings.items()), "final source drift")
    status = "recorded_not_consumed" if len(shards) == 24 and all(
        s["status"] == "recorded_not_consumed" for s in shards) else "partial_or_failed"
    result = {"schema": NAME + ".run", "status": status, "shards": shards,
              "fixed_branches": 1008, "P1_admission": False, "independent_confirmation": False}
    return {"status": status, "sha256": run._write_new(directory / "run_report.json", result)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--prepared")
    ap.add_argument("--sha256")
    ap.add_argument("--authorize-physics", action="store_true")
    args = ap.parse_args()
    if args.command == "prepare":
        print(json.dumps({"sha256": prepare(args.output)}))
    else:
        require(args.authorize_physics, "physical authorization")
        print("Execute all fixed joint/zero/greedy H10 branches; no neural inference.", flush=True)
        result = execute(args.prepared, args.sha256, args.output)
        print(json.dumps(result))
        if result["status"] != "recorded_not_consumed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
