"""Physically replay the fixed same-midpoint one-packet CUDA plans."""
import argparse
import hashlib
import json
from pathlib import Path

from . import codex_p1_open10_matched_feedback_20261030_main as original

NAME = "codex_p1_horizon1_physical_20261030_main"
physical, consumer = original.physical, original.consumer
ROOT = original.ROOT
PLAN = ROOT / "results/codex_p1_horizon1_midpoint_20261030_main_RUN01.json"
PLAN_SHA = "27870d37c665fce4caca5f138c9f5f248d68611dc60bc1b2f6810ef5acfa2397"
need = original.need


def build():
    old = original.build()
    plan, _ = physical.gap.read(PLAN, PLAN_SHA)
    need(plan["status"] == "complete" and len(plan["rows"]) == len(old["roster"]) == 24,
         "fixed plan roster")
    roster = []
    for i, (item, row) in enumerate(zip(old["roster"], plan["rows"], strict=True)):
        need(item["index"] == row["index"] == i, "plan identity")
        first = bytes.fromhex(row["first_packet_hex"])
        second = bytes.fromhex(row["second_packet_hex"])
        need(len(first) == len(second) == 40 and item["raw10_float32_hex"][:80] == first.hex(),
             "authenticated first packet")
        raw = first + second
        updated = dict(item)
        updated.update(raw10_float32_hex=raw.hex(), raw10_sha256=hashlib.sha256(raw).hexdigest())
        roster.append(updated)
    bindings = dict(old["bindings"])
    bindings[str(PLAN)] = PLAN_SHA
    bindings[str(Path(__file__).resolve())] = physical.old.sha(Path(__file__).resolve())
    return {**old, "schema": NAME + ".prepare", "roster": roster, "bindings": bindings,
            "plan_sha256": PLAN_SHA, "interpretation": "known-root exploratory horizon comparison"}


def install():
    original.feedback.configure()
    physical.NAME = NAME
    physical.build = build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--directory")
    parser.add_argument("--report-sha")
    args = parser.parse_args()
    install()
    output = Path(args.output)
    if args.command == "run":
        return physical.run(output)
    need(args.directory and args.report_sha and output.is_absolute()
         and output.parent == ROOT / "results" and output.name.startswith(NAME + "_saved_")
         and not output.exists() and not output.is_symlink(), "exclusive saved output")
    consumer.DIR, consumer.REPORT_SHA = Path(args.directory), args.report_sha
    result = consumer.consume()
    result["schema"] = NAME + ".saved"
    result["limitations"] += "; verify physical midpoint identity before comparing"
    print(json.dumps({"sha256": physical.old.h10._write_new(output, result),
                      "rows": len(result["rows"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
