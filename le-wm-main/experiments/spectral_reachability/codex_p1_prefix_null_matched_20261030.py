"""Fixed 8-pair prefix-null matched capability exploration; no neural calls."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

from . import codex_p1_prefix_null_saved_consumer_20261030_v2 as source_saved
from . import p1_backtrack_capability_prepare_20261030 as probes
from . import p1_matched_censored_runner_20261030_codex as core
from . import p1_backtrack_physical_consume_20261030 as saved

ROOT = core.ROOT
NAME = "codex_p1_prefix_null_matched_20261030"
SOURCE = ROOT / "results/codex_p1_prefix_null_runner_20261030_v2_run_MAIN01"
SOURCE_SHA = "751344ac69ac078322592707dcd89eb534ee777e63128b3a8cd6298251953551"
PREP = ROOT / "results/codex_p1_prefix_null_runner_20261030_v2_prepare_CODEX02.json"
PREP_SHA = "9344fb56183fbbfeb69520b8161c1d77000c334e241755d1ad92f2810427bef3"
CONSUMER = Path(__file__).with_name("codex_p1_prefix_null_matched_saved_20261030.py")
TEST = ROOT / "tests/test_codex_p1_prefix_null_matched_20261030.py"
DESIGN = ROOT.parent / "docs/D2P_P1_PREFIX_NULL_MATCHED_CODEX_20261030.md"
need = core.require
sha = core.runner.physical.sha
native = core.native


def build():
    verdict = source_saved.consume(SOURCE, SOURCE_SHA)
    need(verdict["status"] == "saved_checks_passed_exploratory_only", "source saved acceptance")
    p, data = source_saved.runner.verify(PREP, PREP_SHA)
    report = native.read(SOURCE / "report.json", cap=64*1024**2)
    need(report["execution_binding"]["prepared"] == str(PREP) and
         report["execution_binding"]["prepare_sha256"] == PREP_SHA, "source prepare")
    events = saved.events(SOURCE / "journal.jsonl", report["journal_chain_tail"])
    anchors = []
    bindings = copy.deepcopy(p["bindings"])
    for path in (PREP, SOURCE / "report.json", SOURCE / "journal.jsonl",
                 SOURCE / "execution_binding.json", Path(__file__).resolve(), CONSUMER, TEST, DESIGN,
                 Path(core.__file__).resolve(), Path(core.proof.__file__).resolve(),
                 Path(probes.__file__).resolve(), Path(saved.__file__).resolve(),
                 Path(source_saved.__file__).resolve(),
                 Path(core.runner.__file__).resolve(), Path(core.prior.__file__).resolve(),
                 Path(__file__).with_name("p1_matched_censored_consumer_20261030_codex.py")):
        need(path.is_file() and not path.is_symlink(), "regular bound source")
        digest = sha(path)
        need(str(path) not in bindings or bindings[str(path)] == digest, "binding collision")
        bindings[str(path)] = digest
    for i, (plan, row) in enumerate(zip(p["plans"], report["rows"], strict=True)):
        need(plan["index"] == row["index"] == i, "source row order")
        trace = [e for e in events if e["kind"] == "fresh_observation" and e.get("branch") == i and
                 (e["receipt"]["phase"] == "warmup" or
                  (e["receipt"]["phase"] == "plan" and e["receipt"]["step"] <= 5))]
        schedule = [(phase, step, overlay) for phase, count in (("warmup", 11), ("plan", 6))
                    for step in range(count) for overlay in range(3)]
        need([(e["receipt"]["phase"], e["receipt"]["step"], e["receipt"]["overlay_index"])
              for e in trace] == schedule, "full source anchor")
        for event in trace:
            receipt = event["receipt"]
            file = SOURCE / receipt["png"]
            need(file.parent == SOURCE and file.resolve() == file and file.is_file()
                 and not file.is_symlink() and sha(file) == receipt["file_sha256"], "anchor PNG")
            bindings[str(file)] = receipt["file_sha256"]
        raw = bytes.fromhex(plan["raw15_float32_hex"])[:40]
        anchors.append(dict(index=i, root_slot=plan["root_slot"], condition=plan["condition"],
                            arm=plan["arm"], role=plan["role"], task_id=plan["task_id"],
                            physical_root_sha256=plan["physical_root_sha256"],
                            prefix5_float32_hex=raw.hex(), prefix5_sha256=hashlib.sha256(raw).hexdigest(),
                            root_state=row["root_state"], goal=row["goal"],
                            checkpoint=next(c for c in row["checkpoints"] if c["step"] == 5),
                            trace=[{k: copy.deepcopy(e[k]) for k in ("receipt", "pre", "post")} for e in trace]))
    bank = probes.probes()
    need(len(anchors) == 16 and len(bank) == 21 and
         [b["index"] for b in bank] == list(range(21)), "fixed roster")
    for i in range(8):
        a, b = anchors[2*i:2*i+2]
        need((a["role"], b["role"]) == ("selected_prefix", "zero_prefix") and
             bytes.fromhex(b["prefix5_float32_hex"]) == bytes(40) and
             all(a[k] == b[k] for k in ("root_slot", "condition", "arm", "task_id",
                                          "physical_root_sha256")), "pair identity")
    need(all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    return dict(schema=NAME+".prepare.v1", anchors=anchors, probes=bank, bindings=bindings,
                source_report_sha256=SOURCE_SHA, fixed_pairs=8, fixed_anchors=16,
                fixed_branches=336, P1_admission=False, independent_confirmation=False,
                equal_budget_certified=False, neural_calls=0), data


def prepare(path):
    path = Path(path)
    need(path.is_absolute() and path.parent == ROOT / "results" and
         path.name.startswith(NAME+"_prepare_"), "prepare namespace")
    p, _ = build()
    return core._write_new(path, p)


def verify(path, digest):
    path = Path(path)
    native.actual_hash(digest)
    need(path.is_absolute() and path.parent == ROOT / "results" and
         path.name.startswith(NAME+"_prepare_") and path.is_file() and not path.is_symlink()
         and sha(path) == digest, "prepare identity")
    value = native.read(path, cap=64*1024**2)
    current, data = build()
    need(value == current and sha(path) == digest, "prepare/source drift")
    return value, data


def run(prepared, digest, output):
    p, data = verify(prepared, digest)
    native.constructor_contract()
    directory = Path(output)
    need(directory.is_absolute() and directory.parent == ROOT / "results" and
         directory.name.startswith(NAME+"_run_") and not directory.exists() and
         not directory.is_symlink() and
         shutil.disk_usage(directory.parent).free >= core.runner.FLOOR+16*core.runner.CAP,
         "exclusive reservation")
    directory.mkdir()
    core.base.paired.fsync_dir(directory.parent)
    bindings = dict(p["bindings"], **{str(Path(prepared)): digest})
    binding_sha = core._write_new(directory / "run_binding.json",
                                  dict(prepared=str(prepared), prepare_sha256=digest, bindings=bindings,
                                       fixed_branches=336, P1_admission=False, independent_confirmation=False))
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
    shards = []
    for anchor in p["anchors"]:
        path = directory / f"anchor_{anchor['index']:02d}"
        report = core.execute_shard(anchor, p["probes"], data, path, factory, bindings)
        shards.append(dict(anchor=anchor["index"], status=report["status"],
                           report_sha256=sha(path / "report.json")))
        if report["status"] != "recorded_not_consumed":
            break
    need(all(sha(file) == h for file, h in bindings.items()), "final source drift")
    result = dict(schema=NAME+".run.v1", fixed_branches=336, fixed_anchors=16,
                  status="recorded_not_consumed" if len(shards) == 16 and
                  all(s["status"] == "recorded_not_consumed" for s in shards) else "partial_or_failed",
                  shards=shards, run_binding_sha256=binding_sha, P1_admission=False,
                  independent_confirmation=False)
    return dict(status=result["status"], sha256=core._write_new(directory / "run_report.json", result))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--prepared")
    ap.add_argument("--prepare-sha")
    a = ap.parse_args()
    if a.command == "prepare":
        need(a.prepared is None and a.prepare_sha is None, "prepare arguments")
        result = dict(status="prepared", sha256=prepare(a.output))
    else:
        need(a.prepared and a.prepare_sha, "run arguments")
        result = run(a.prepared, a.prepare_sha, a.output)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] in ("prepared", "recorded_not_consumed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
