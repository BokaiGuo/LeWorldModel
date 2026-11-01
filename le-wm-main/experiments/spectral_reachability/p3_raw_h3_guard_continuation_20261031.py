"""Continue unattempted H3 batches only after verifying a retained canvas-guard partial."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from . import p3_raw_h3_physical_source_20261031 as source

ROOT, host, require = source.ROOT, source.host, source.require
RUN_NAME = "p3_raw_h3_physical_run_20261031"
PREVIOUS = ROOT / "results/p3_raw_h3_physical_run_20261031_BATCH01_RUN01"
PREVIOUS_SHA = "8d784b16e58fa990a75f1b4f5104177b7f4249070b80cccf58562ec6f796c6dd"
PREPARE_SHA = "7951a82830161ca21d61eeac7350a8e9b4c336e2973060d5e4fac3de480165f3"


def allowed_partial(report, events):
    require(report["status"] == "partial_or_failed" and report["errors"] == []
            and len(report["rows"]) == 6 and report["scientific_pass"] is False,
            "partial report contract")
    failures = [row for row in report["rows"] if row["status"] == "failed"]
    require(bool(failures), "no retained failures")
    for row in report["rows"]:
        if row["status"] != "failed":
            require(row["status"] in ("horizon_completed", "native_success", "native_stopped")
                    and row["first_error"] is None and not row["secondary_errors"], "other row status")
            continue
        error, obs = row["first_error"], row["last_observation"]
        state = obs["state"]
        require(error and error["message"] == "RGB/canvas/state/pure-render guard; failed image retained"
                and not row["secondary_errors"] and len(state) == 7
                and all(math.isfinite(x) for x in state) and obs["observation_state"] == state
                and any(x < 0 or x > 512 for x in state[:4]), "not certified canvas guard")
        matches = [e for e in events if e["kind"] == "guard_receipt"
                   and e.get("branch") == row["branch_index"] and e.get("accepted") is False]
        require(len(matches) == 1 and matches[0]["render_preserved_dynamics"] is True
                and matches[0]["phase"] == "plan" and matches[0]["step"] == row["own_returned"],
                "guard event binding")
    return len(failures)


def authenticate_partial(directory, report_sha, prepare_sha, index):
    require(host.sha(directory / "report.json") == report_sha
            and host.sha(directory / "prepare.json") == prepare_sha, "partial external hashes")
    expected, _ = source.reconstruct(index)
    require(host.helper.read(directory / "prepare.json") == expected, "partial prepare reconstruction")
    report = host.helper.read(directory / "report.json")
    require(host.sha(directory / "journal.jsonl") == report["journal_sha256"], "partial journal hash")
    events = source.batch.kernel.journal(directory / "journal.jsonl", report["journal_chain_tail"])
    return allowed_partial(report, events)


def run():
    authenticate_partial(PREVIOUS, PREVIOUS_SHA, PREPARE_SHA, 1)
    for index in range(2, 12):
        output = ROOT / f"results/{RUN_NAME}_BATCH{index:02d}_RUN01"
        require(not output.exists() and not output.is_symlink(), "batch already attempted")
        command = [sys.executable, "-B", "-m",
                   "experiments.spectral_reachability." + RUN_NAME,
                   "--batch", str(index), "--output", str(output), "--authorize-physics"]
        completed = subprocess.run(command, check=False)
        report, prepare = output / "report.json", output / "prepare.json"
        require(report.is_file() and prepare.is_file(), "missing batch receipt")
        report_sha, prepare_sha = host.sha(report), host.sha(prepare)
        if completed.returncode:
            count = authenticate_partial(output, report_sha, prepare_sha, index)
            print(json.dumps({"batch": index, "status": "canvas_guard_partial",
                              "guard_failures": count, "report_sha256": report_sha}), flush=True)
        else:
            print(json.dumps({"batch": index, "status": "saved_validated",
                              "report_sha256": report_sha}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-physics", action="store_true", required=True)
    parser.parse_args()
    run()
