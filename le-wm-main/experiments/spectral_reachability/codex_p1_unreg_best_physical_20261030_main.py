"""Execute and consume all fixed unregularized best-iterate suffixes."""
import argparse
import json
from pathlib import Path

from . import codex_p1_adam_feedback_physical_20261030_main as feedback
from . import codex_p1_adam_best_iterate_20261030_main as planner
from . import codex_p1_midpoint_stop_guard_20261030_v1 as guard
from . import codex_p1_unreg_best_20261030_main as wrapper

ROOT = feedback.ROOT
NAME = "codex_p1_unreg_best_physical_20261030_main"
PLAN = ROOT / "results/codex_p1_unreg_best_20261030_main_RUN01.json"
SHA = "8cc2ce5e72032dc3966085523a502b2c564a9c3f1cde6acbf0ad7b2bd091c8c8"


def build():
    feedback.PLAN, feedback.PLAN_SHA = PLAN, SHA
    result = feedback.build()
    result["schema"] = NAME + ".prepare"
    for path in (Path(planner.__file__), Path(guard.__file__), Path(wrapper.__file__), Path(__file__)):
        path = path.resolve()
        result["bindings"][str(path)] = feedback.physical.old.sha(path)
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
