"""Execute and consume the fixed trust-regularized midpoint suffix panel."""
import argparse
import json
from pathlib import Path

from . import codex_p1_adam_feedback_physical_20261030_main as feedback

ROOT = feedback.ROOT
NAME = "codex_p1_adam_trust_physical_20261030_main"
PLAN = ROOT / "results/codex_p1_adam_trust_plan_20261030_main_RUN01.json"
SHA = "e42a4e8cd961c2cacc0f41b25d5a09a13b0c4b748a222383bd1f7a58a419bfe4"


def build():
    feedback.PLAN, feedback.PLAN_SHA = PLAN, SHA
    result = feedback.build()
    result["schema"] = NAME + ".prepare"
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
