"""Execute and consume every fixed best-iterate suffix on the known panel."""
import argparse
import json
from pathlib import Path

from . import codex_p1_adam_feedback_physical_20261030_main as feedback
from . import codex_p1_adam_best_iterate_20261030_main as plan
from . import codex_p1_midpoint_stop_guard_20261030_v1 as guard

ROOT = feedback.ROOT
NAME = "codex_p1_adam_best_iterate_physical_20261030_main"
PLAN = ROOT / "results/codex_p1_adam_best_iterate_20261030_main_RUN01.json"
SHA = "b59e996aa618b6477b97790f63d979c6bc5856b7ced134b2663728176a3489c2"


def build():
    feedback.PLAN, feedback.PLAN_SHA = PLAN, SHA
    result = feedback.build()
    result["schema"] = NAME + ".prepare"
    for module in (plan, guard):
        path = Path(module.__file__).resolve()
        result["bindings"][str(path)] = feedback.physical.old.sha(path)
    result["bindings"][str(Path(__file__).resolve())] = feedback.physical.old.sha(Path(__file__).resolve())
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("run", "consume"))
    ap.add_argument("--directory", required=True)
    ap.add_argument("--report-sha")
    ap.add_argument("--output")
    args = ap.parse_args()
    physical = feedback.physical
    physical.NAME, physical.build = NAME, build
    directory = Path(args.directory)
    if args.command == "run":
        return physical.run(directory)
    if args.output is None:
        raise ValueError("output required")
    out = Path(args.output)
    feedback.need(args.report_sha and out.is_absolute() and out.parent == ROOT / "results"
                  and out.name.startswith(NAME + "_SAVED_") and not out.exists() and not out.is_symlink(), "output")
    feedback.consumer.DIR, feedback.consumer.REPORT_SHA = directory, args.report_sha
    result = feedback.consumer.consume()
    result["schema"] = NAME + ".saved"
    result["limitations"] += "; offline RGB midpoint, paired verification required"
    print(json.dumps({"rows": len(result["rows"]), "sha256": physical.old.h10._write_new(out, result)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
