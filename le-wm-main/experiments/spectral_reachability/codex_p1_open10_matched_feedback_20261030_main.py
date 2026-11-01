"""Replay the original 10-update open plan from each fixed feedback episode.

No new neural inference. Compare only with the already executed 10+10 run;
the first five actions must be verified identical in the subsequent analysis.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_feedback_10x2_20261030_main as feedback
from . import codex_p1_multistart_physical_20261030_main as physical
from . import codex_p1_multistart_physical_saved_20261030_main as consumer

NAME = "codex_p1_open10_matched_feedback_20261030_main"
ROOT = physical.old.ROOT
FB_DIR = ROOT / "results/codex_p1_two_stage_launcher_20261030_v1_run_10x2_MAIN01"
FB_REPORT_SHA = "4c2916154862f8316ef070ea39d59f48e50b97620fe3b8021fa60b60e94fcbd0"
FB_PREP = ROOT / "results/codex_p1_two_stage_launcher_20261030_v1_prepare_10x2_MAIN01.json"
FB_PREP_SHA = "3e31c2c8d16b4fe8c3862985a2c149c559c57311ca68e98d7f2020366c711468"
need = physical.need


def build():
    # The existing consumer authenticates all phase plans against actual
    # transition journal, images, frozen prepare and source-specific ledgers.
    checked = feedback.saved.consume(str(FB_DIR), FB_REPORT_SHA, str(FB_PREP), FB_PREP_SHA)
    prior, _ = physical.gap.read(physical.OLD_PREP, physical.OLD_SHA)
    need(checked["status"] == "saved_checks_passed" and len(checked["rows"]) == 24
         and len(prior["roster"]) == 24, "complete source")
    roster, files = [], [FB_PREP, FB_DIR/"run_report.json", physical.OLD_PREP, Path(__file__).resolve()]
    for i, (row, old) in enumerate(zip(checked["rows"], prior["roster"], strict=True)):
        need(row["index"] == old["index"] == i and row["role"] == old["role"]
             and row["logical_pair"] == old["logical_pair"] and row["root_slot"] == old["root_slot"]
             and row["condition"] == old["condition"] and row["arm"] == old["arm"], "row identity")
        report_path = FB_DIR/f"row_{i:02d}"/"stage_1/report.json"
        report = json.loads(report_path.read_bytes())
        raw = bytes.fromhex(report["result"]["raw_horizon_f32_hex"])
        action = np.frombuffer(raw, dtype="<f4")
        need(report["index"] == i and report["stage"] == 1 and report["status"] == "complete"
             and len(raw) == 80 and np.isfinite(action).all() and (np.abs(action) <= 1).all(),
             "full initial plan")
        item = dict(old)
        item.update(raw10_float32_hex=raw.hex(), raw10_sha256=hashlib.sha256(raw).hexdigest())
        roster.append(item)
        files.append(report_path)
    bindings = dict(prior["bindings"])
    for path in files:
        bindings[str(path)] = physical.old.sha(path)
    return {"schema": NAME+".prepare", "roster": roster, "fixed_rows": 24,
            "bindings": bindings, "caps_per_row": physical.old.CAPS,
            "feedback_report_sha256": FB_REPORT_SHA, "feedback_prepare_sha256": FB_PREP_SHA,
            "known_roots": True, "P1_admission": False, "fairness_proven": False,
            "independent_confirmation": False, "interpretation": "same initial plan, feedback costs more"}


def install():
    feedback.configure()
    physical.NAME = NAME
    physical.build = build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--directory")
    parser.add_argument("--report-sha")
    args = parser.parse_args()
    install()
    output = Path(args.output)
    if args.command == "prepare":
        need(output.is_absolute() and output.parent == ROOT/"results"
             and output.name.startswith(NAME+"_prepare_") and not output.exists() and not output.is_symlink(),
             "exclusive prepare")
        print(json.dumps({"sha256": physical.old.h10._write_new(output, build())}), flush=True)
    elif args.command == "run":
        need(args.prepared and args.prepare_sha, "prepared input")
        prepared, _ = physical.gap.read(Path(args.prepared), args.prepare_sha)
        need(prepared == build(), "prepare drift")
        return physical.run(output)
    else:
        need(args.directory and args.report_sha and output.is_absolute() and output.parent == ROOT/"results"
             and output.name.startswith(NAME+"_saved_") and not output.exists() and not output.is_symlink(),
             "exclusive saved input/output")
        consumer.DIR, consumer.REPORT_SHA = Path(args.directory), args.report_sha
        result = consumer.consume()
        result["schema"] = NAME+".saved"
        result["limitations"] += "; same initial plan comparison requires explicit cp10 replay equivalence"
        print(json.dumps({"sha256": physical.old.h10._write_new(output, result),
                          "rows": len(result["rows"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
