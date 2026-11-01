"""Prepare and execute fixed H10 interventions for all uphill-reranked prefixes."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from . import p1_uphill_physical_20261030 as uphill
from . import p1_joint_h10_20261030 as previous

ROOT = uphill.ROOT
NAME = "p1_uphill_h10_20261030"
PHYS = ROOT / "results/p1_uphill_physical_20261030_RUN01"
PHYS_SHA = "b1fbe4b4811cc8d110aca0d4f90edef13382321ecd21dbac66f1f893ddbda825"
JOIN_SHA = "2f8c75e502794248c399385e01db5cc991fb69b1eab7f34e4676becd1fe04e81"
PROTOCOL = ROOT.parent / "docs/D2P_P1_UPHILL_H10_PROTOCOL_20261030_MAIN.md"
run, saved, events = previous.run, previous.saved, previous.events
require, sha = uphill.need, uphill.sha


def build():
    old = run.verify_prepare(previous.OLD, previous.OLD_SHA)
    report_path = PHYS / "report.json"
    require(sha(report_path) == PHYS_SHA, "physical report hash")
    report = uphill.runner.native.read(report_path, cap=64 * 1024**2)
    binding = report["execution_binding"]
    physical, _ = uphill.verify(binding["prepared"], binding["prepare_sha256"], JOIN_SHA)
    uphill.consumer.check(PHYS, report, physical)
    journal = events.events(PHYS / "journal.jsonl", report["journal_chain_tail"])
    bindings = copy.deepcopy(old["bindings"])
    for path in (previous.OLD, report_path, PHYS / "journal.jsonl", PHYS / "execution_binding.json",
                 Path(binding["prepared"]), Path(__file__).resolve(), PROTOCOL,
                 Path(saved.__file__).resolve(), Path(uphill.__file__).resolve()):
        digest = sha(path)
        require(str(path) not in bindings or bindings[str(path)] == digest, "binding collision")
        bindings[str(path)] = digest
    for path, digest in physical["bindings"].items():
        require(path not in bindings or bindings[path] == digest, "physical source collision")
        bindings[path] = digest
    anchors, banks = [], []
    for logical in range(8):
        row, plan = report["rows"][2 * logical + 1], physical["plans"][2 * logical + 1]
        triplet = old["original"]["anchors"][3 * logical:3 * logical + 3]
        require(row["role"] == plan["role"] == "model_selected"
                and all(row[k] == plan[k] == triplet[0][k]
                        for k in ("root_slot", "condition", "arm", "task_id"))
                and row["root_state"] == triplet[0]["root_state"]
                and row["goal"] == triplet[0]["goal"], "source anchor identity")
        trace = [e for e in journal if e["kind"] == "fresh_observation"
                 and e.get("branch") == row["index"]
                 and (e["receipt"]["phase"] == "warmup" or
                      (e["receipt"]["phase"] == "plan" and e["receipt"]["step"] <= 5))]
        schedule = [(phase, step, overlay) for phase, count in (("warmup", 11), ("plan", 6))
                    for step in range(count) for overlay in range(3)]
        require([(e["receipt"]["phase"], e["receipt"]["step"], e["receipt"]["overlay_index"])
                 for e in trace] == schedule, "source trace roster")
        for event in trace:
            rec = event["receipt"]
            path = PHYS / rec["png"]
            require(path.parent == PHYS and path.is_file() and not path.is_symlink()
                    and sha(path) == rec["file_sha256"], "source frame")
            bindings[str(path)] = rec["file_sha256"]
        raw = bytes.fromhex(plan["raw15_float32_hex"])
        require(len(raw) == 120, "action bytes")
        selected = copy.deepcopy(triplet[0])
        selected.update(prefix5_float32_hex=raw[:40].hex(),
                        prefix5_sha256=hashlib.sha256(raw[:40]).hexdigest(),
                        checkpoint=next(c for c in row["checkpoints"] if c["step"] == 5),
                        trace=[{k: copy.deepcopy(e[k]) for k in ("receipt", "pre", "post")} for e in trace],
                        source_family="uphill_cuda_physical_RUN01")
        for role, item in zip(("selected", "zero", "greedy"),
                              (selected, triplet[1], triplet[2]), strict=True):
            anchor = copy.deepcopy(item)
            require(anchor["role"] == role and anchor["logical_pair"] == logical
                    and all(anchor[k] == selected[k] for k in
                            ("root_slot", "condition", "arm", "task_id", "physical_root_sha256",
                             "goal", "root_state")), "triplet binding")
            anchor["index"] = len(anchors)
            anchors.append(anchor)
        banks.append(run.design.bank(raw[40:]))
    require(len(anchors) == 24 and len(banks) == 8 and all(len(bank) == 42 for bank in banks),
            "complete roster")
    require(all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "anchors": anchors, "probe_banks": banks,
            "bindings": bindings, "fixed_anchors": 24, "fixed_branches": 1008,
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False}


def execute(prepared_path, digest, output):
    path = Path(prepared_path)
    require(sha(path) == digest, "prepare hash")
    p = uphill.runner.native.read(path, cap=64 * 1024**2)
    require(p == build(), "prepare/source drift")
    binding = uphill.runner.native.read(PHYS / "report.json", cap=64 * 1024**2)["execution_binding"]
    _, data = uphill.verify(binding["prepared"], binding["prepare_sha256"], JOIN_SHA)
    uphill.runner.native.constructor_contract()
    directory = Path(output)
    require(directory.is_absolute() and directory.parent == ROOT / "results"
            and directory.name.startswith(NAME + "_run_") and not directory.exists()
            and shutil.disk_usage(directory.parent).free >= run.FLOOR + 24 * run.CAP,
            "exclusive output and full disk reservation")
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
        target = directory / f"anchor_{anchor['index']:02d}"
        result = run.execute_shard(anchor, p["probe_banks"][anchor["logical_pair"]],
                                   data, target, factory, bindings)
        shards.append({"anchor": anchor["index"], "status": result["status"],
                       "sha256": sha(target / "report.json")})
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
        out = Path(args.output)
        require(out.is_absolute() and out.parent == ROOT / "results"
                and out.name.startswith(NAME + "_prepare_") and not out.exists(), "exclusive prepare")
        print(json.dumps({"sha256": run._write_new(out, build())}), flush=True)
    else:
        require(args.authorize_physics, "physics authorization")
        print("Execute all fixed uphill/zero/greedy H10 branches; no neural inference.", flush=True)
        result = execute(args.prepared, args.sha256, args.output)
        print(json.dumps(result), flush=True)
        if result["status"] != "recorded_not_consumed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
