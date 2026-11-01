"""Execute fixed Adam first packets and real-midpoint reoptimized second packets."""
import argparse
import hashlib
import json
from pathlib import Path

from . import codex_p1_multistart_physical_20261030_main as physical
from . import codex_p1_multistart_physical_saved_20261030_main as consumer
from . import codex_p1_adam_physical_20261030_main as adam

ROOT = physical.old.ROOT
NAME = "codex_p1_adam_feedback_physical_20261030_main"
PLAN = ROOT / "results/codex_p1_adam_feedback_plan_20261030_main_RUN01.json"
PLAN_SHA = "3fe7a35c3b42600a29d74423cdd80cf75a85499a54b31e276dd03261b4ae50c5"
ORIGINAL_BUILD = physical.build
need = physical.need


def build():
    physical.PLAN, physical.PLAN_SHA = adam.PLAN, adam.PLAN_SHA
    original = ORIGINAL_BUILD()
    plan, _ = physical.gap.read(PLAN, PLAN_SHA)
    need(plan["status"] == "complete" and len(plan["rows"]) == len(original["roster"]) == 24, "roster")
    roster = []
    for i, (old, row) in enumerate(zip(original["roster"], plan["rows"], strict=True)):
        need(all(old[k] == row[k] for k in ("index", "logical_pair", "role")) and row["index"] == i, "identity")
        first, second = bytes.fromhex(row["first_packet_hex"]), bytes.fromhex(row["second_packet_hex"])
        need(len(first) == len(second) == 40 and old["raw10_float32_hex"][:80] == first.hex(), "first packet")
        raw = first + second
        roster.append({**old, "raw10_float32_hex": raw.hex(), "raw10_sha256": hashlib.sha256(raw).hexdigest()})
    bindings = dict(original["bindings"])
    bindings.update({str(PLAN): PLAN_SHA, str(Path(__file__).resolve()): physical.old.sha(Path(__file__).resolve())})
    return {**original, "schema": NAME + ".prepare", "roster": roster, "bindings": bindings,
            "plan_sha256": PLAN_SHA, "interpretation": "known-root offline real-midpoint feedback exploration"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("run", "consume"))
    ap.add_argument("--directory", required=True)
    ap.add_argument("--report-sha")
    ap.add_argument("--output")
    args = ap.parse_args()
    physical.NAME, physical.build = NAME, build
    directory = Path(args.directory)
    if args.command == "run":
        return physical.run(directory)
    out = Path(args.output)
    need(args.report_sha and out.is_absolute() and out.parent == ROOT / "results"
         and out.name.startswith(NAME + "_SAVED_") and not out.exists() and not out.is_symlink(), "exclusive output")
    consumer.DIR, consumer.REPORT_SHA = directory, args.report_sha
    result = consumer.consume()
    result["schema"] = NAME + ".saved"
    result["limitations"] += "; midpoint reproducibility needs paired verification; offline feedback"
    print(json.dumps({"rows": len(result["rows"]), "sha256": physical.old.h10._write_new(out, result)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
