"""Execute and consume every fixed Adam suffix winner."""
import argparse
import json
from pathlib import Path

from . import codex_p1_multistart_physical_20261030_main as run
from . import codex_p1_multistart_physical_saved_20261030_main as saved

NAME = "codex_p1_adam_physical_20261030_main"
PLAN = run.old.ROOT / "results/codex_p1_adam_cp5_20261030_main_RUN01.json"
PLAN_SHA = "455c93a37f8d1200ffe75290e30e939378ed536b9d81ede802da221f7b6c4253"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "consume"))
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha")
    parser.add_argument("--output")
    args = parser.parse_args()
    run.PLAN, run.PLAN_SHA, run.NAME = PLAN, PLAN_SHA, NAME
    directory = Path(args.directory)
    if args.command == "run":
        return run.run(directory)
    run.need(args.report_sha and args.output and directory.is_absolute(), "consumer inputs")
    output = Path(args.output)
    run.need(output.is_absolute() and output.parent == run.old.ROOT / "results"
             and output.name.startswith(NAME + "_SAVED_") and not output.exists() and not output.is_symlink(),
             "exclusive consumer output")
    saved.DIR, saved.REPORT_SHA = directory, args.report_sha
    result = saved.consume()
    result["schema"] = NAME + ".consume"
    result["wrapper_sha256"] = run.old.sha(Path(__file__).resolve())
    result["limitations"] += "; wrapper and sources not a complete dependency closure"
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]),
                      "sha256": run.old.h10._write_new(output, result)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
