"""Strict saved-only consumer for the v2 exploratory physical adapter."""
import argparse
import json
from pathlib import Path
import sys

from . import codex_p1_prefix_null_runner_20261030_v2 as runner
from . import codex_p1_prefix_null_saved_consumer_20261030_v1 as checks


def consume(directory, report_sha):
    directory = Path(directory)
    runner.need(directory.is_absolute() and directory.parent == runner.ROOT / "results" and
                directory.name.startswith(runner.NAME + "_run_") and directory.is_dir() and
                not directory.is_symlink(), "exclusive run directory")
    runner.v1.base.native.actual_hash(report_sha)
    path = directory / "report.json"
    runner.need(path.is_file() and not path.is_symlink() and runner.v1.base.sha(path) == report_sha,
                "report digest")
    report = runner.v1.base.native.read(path, cap=64 * 1024**2)
    binding = report["execution_binding"]
    prepared, data = runner.verify(binding["prepared"], binding["prepare_sha256"])
    runner.need(binding == runner.v1.base.native.read(directory / "execution_binding.json") and
                binding["join_sha256"] == runner.v1.roster.JOIN_SHA and
                runner.v1.base.sha(path) == report_sha, "execution binding")
    return checks.check(directory, report, prepared, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-sha", required=True)
    args = parser.parse_args()
    try:
        result = consume(args.output, args.report_sha)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "saved_checks_passed_exploratory_only" else 3
    except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
        print(json.dumps({"status": "blocked_invalid", "reason": str(exc), "P1_admission": False}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
