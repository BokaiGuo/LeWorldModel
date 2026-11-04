"""Verify saved point-30 physical evidence without executing the simulator."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from . import p3_point30_physical_prepare_20261102 as source
from . import p3_preoutcome_winner_physical_saved_20261102_codexexclusive as saved


def consume(plan_path, plan_sha, directory, report_sha):
    prior, moving = source.prior, source.prior.moving
    prior.require(moving.sha(plan_path) == plan_sha, "plan SHA")
    plan = moving.read(plan_path)
    prior.require(plan == source.material(), "plan and source binding")
    directory = Path(directory)
    prior.require(directory.is_absolute() and directory.parent == source.ROOT / "results"
                  and directory.name.startswith(source.NAME + "_RUN_")
                  and directory.is_dir() and not directory.is_symlink(), "run path")
    prior.require(moving.read(directory / "execution_binding.json")
                  == {"prepared": str(Path(plan_path).resolve()), "prepare_sha256": plan_sha},
                  "execution binding")
    prior.require(moving.sha(directory / "report.json") == report_sha, "report SHA")
    report = moving.read(directory / "report.json")
    prior.require(moving.sha(directory / "journal.jsonl") == report["journal_sha256"], "journal SHA")
    events = moving.journal(directory / "journal.jsonl", report["journal_chain_tail"])
    prior.require(events and events[0]["kind"] == "run_started" and events[0]["prepare"] == plan,
                  "exact saved prepare in journal")
    data = moving.load_source(prior.prior.source.SOURCEPACK, prior.prior.source.SOURCEPACK_SHA,
                              prior.prior.source.SOURCE_PREPARE, prior.prior.source.SOURCE_PREPARE_SHA)
    # The reusable validator expects this presentation field in its input plan.
    # Verify the original journal above, then add it only to in-memory copies.
    view, event_view = copy.deepcopy(plan), copy.deepcopy(events)
    view["model_accounting"] = {
        "point30_resources": moving.read(source.SAVED)["points"][1]["resources"],
        "full_cost_certified": False,
    }
    event_view[0]["prepare"] = view
    result = saved.validate(view, data["references"], report, event_view, directory)
    result.update(prepare_sha256=plan_sha, report_sha256=report_sha,
                  journal_sha256=report["journal_sha256"],
                  model_outcomes_already_open=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "sha256", "run", "report-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = consume(args.plan, args.sha256, args.run, args.report_sha256)
    out = Path(args.output)
    source.prior.require(out.is_absolute() and out.parent == source.ROOT / "results"
                         and out.name.startswith(source.NAME + "_SAVED_") and not out.exists(),
                         "exclusive output")
    sha = source.prior.producer.host.write_new(out, result)
    print(json.dumps({"status": result["status"], "sha256": sha}))


if __name__ == "__main__":
    main()
