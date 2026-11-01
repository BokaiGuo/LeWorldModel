"""Seal all 16 x 21 known-root P1 common-prefix probe inputs.

Saved-data preparation only. No model, simulator, action selection, physical
dispatch, or scientific admission. Physical outcomes have already been seen,
so this is exploratory, not a prospective independent confirmation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p1_backtrack_physical_consume_20261030 as saved
from experiments.spectral_reachability import p1_backtrack_physical_runner_20261030 as physical

NAME = "p1_backtrack_capability_prepare_20261030"
ROOT = physical.ROOT
DIRECTORY = ROOT / "results/p1_backtrack_physical_runner_20261030_MAIN02"
REPORT_SHA = "26802255a8192e95ae3d67a089ee673ba6e8a7331786481fefde9f4fd1e84de9"
PREP_SHA = "f387cd9cf094c09efe8219b6fe7fc189fdef6fc72491adfe5f30b040f14a8e81"
require = physical.require


def probes():
    values = [("zero", np.zeros(10, dtype=np.float32))]
    for coordinate in range(10):
        for sign, label in ((1, "+"), (-1, "-")):
            logits = np.zeros(10, dtype=np.float32)
            logits[coordinate] = np.float32(sign * .25)
            values.append((f"{label}{coordinate}", logits))
    result = []
    for index, (name, logits) in enumerate(values):
        require(float(np.linalg.norm(logits)) <= .75 * np.sqrt(5), "probe ball")
        raw = np.tanh(logits).astype("<f4").reshape(5, 2)
        result.append(dict(index=index, name=name, logits_float32_hex=logits.astype("<f4").tobytes().hex(),
                           raw5_float32_hex=raw.tobytes().hex(),
                           raw5_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
    return result


def build():
    report_path = DIRECTORY / "report.json"
    require(physical.sha(report_path) == REPORT_SHA, "fixed physical report SHA")
    report = physical.native.read(report_path, cap=64 * 1024**2)
    binding = report["execution_binding"]
    require(binding["prepare_sha256"] == PREP_SHA and
            binding == physical.native.read(DIRECTORY / "execution_binding.json"), "original execution binding")
    prepared, _ = physical.verify(binding["prepared"], PREP_SHA)
    saved.check(DIRECTORY, report, prepared)
    journal = saved.events(DIRECTORY / "journal.jsonl", report["journal_chain_tail"])
    anchors, bindings = [], deepcopy(prepared["bindings"])
    bindings[str(report_path)] = REPORT_SHA
    bindings[str(DIRECTORY / "journal.jsonl")] = report["journal_sha256"]
    bindings[str(DIRECTORY / "execution_binding.json")] = physical.sha(DIRECTORY / "execution_binding.json")
    bindings[binding["prepared"]] = PREP_SHA
    for row, plan in zip(report["rows"], prepared["plans"], strict=True):
        i = row["index"]
        trace = [e for e in journal if e["kind"] == "fresh_observation" and e.get("branch") == i and
                 (e["receipt"]["phase"] == "warmup" or
                  (e["receipt"]["phase"] == "plan" and e["receipt"]["step"] <= 5))]
        expected = [(phase, step, overlay) for phase, count in (("warmup", 11), ("plan", 6))
                    for step in range(count) for overlay in range(3)]
        require([(e["receipt"]["phase"], e["receipt"]["step"], e["receipt"]["overlay_index"])
                 for e in trace] == expected, "fixed 51-frame anchor trace")
        for event in trace:
            rec = event["receipt"]
            require(event["pre"] == event["post"] and rec["snapshot"] == event["post"]["snapshot"] and
                    rec["state"] == rec["snapshot"]["state"], "source trace invariant")
            path = physical.native.paired.flat(DIRECTORY, rec["png"]).resolve()
            require(physical.sha(path) == rec["file_sha256"], "source PNG identity")
            bindings[str(path)] = rec["file_sha256"]
        prefix = bytes.fromhex(plan["raw15_float32_hex"])[:40]
        require(len(prefix) == 40, "literal prefix5 bytes")
        anchors.append(dict(index=i, root_slot=plan["root_slot"], condition=plan["condition"],
                            arm=plan["arm"], role=plan["role"], task_id=plan["task_id"],
                            physical_root_sha256=plan["physical_root_sha256"],
                            prefix5_float32_hex=prefix.hex(), prefix5_sha256=hashlib.sha256(prefix).hexdigest(),
                            root_state=row["root_state"], goal=row["goal"],
                            checkpoint=next(c for c in row["checkpoints"] if c["step"] == 5),
                            trace=[dict(receipt=deepcopy(e["receipt"]), pre=deepcopy(e["pre"]),
                                        post=deepcopy(e["post"])) for e in trace]))
    for file in (Path(__file__), Path(saved.__file__), Path(physical.__file__)):
        bindings[str(file.resolve())] = physical.sha(file)
    for path, digest in bindings.items():
        require(physical.sha(path) == digest, "input binding drift")
    return dict(schema=NAME, source_report=str(report_path), source_report_sha256=REPORT_SHA,
                source_prepare_sha256=PREP_SHA, anchors=anchors, probes=probes(),
                schedule=[dict(anchor=i, probe=j) for i in range(16) for j in range(21)],
                fixed_anchors=16, fixed_probes=21, fixed_branches=336, known_roots=2, fresh_roots=0,
                unique_anchor_dedup=False, common_direction="root goal minus root state first four native coordinates",
                restore="fresh original recipe and exact 51-observation trace; solver cache not independently serialized",
                training=False, neural_calls=0, P1_admission=False, physical_authorized=False, bindings=bindings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    require(path.is_absolute() and path.parent == ROOT / "results" and
            path.name.startswith(NAME + "_") and not path.exists() and not path.is_symlink(), "exclusive output")
    document = build()
    raw = physical.native.raw(document)
    with path.open("xb") as stream:
        require(stream.write(raw) == len(raw), "complete prepare write")
        stream.flush()
        os.fsync(stream.fileno())
    physical.native.paired.fsync_dir(path.parent)
    print(json.dumps(dict(path=str(path), sha256=physical.sha(path), fixed_branches=336,
                          physical_authorized=False)))


if __name__ == "__main__":
    main()
