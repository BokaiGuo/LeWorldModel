"""Saved verifier allowing one explicitly checked empty runtime cache directory."""
from __future__ import annotations

import argparse
import json
import math

from experiments.spectral_reachability import p2_multi_joint_physical_20261102_codexexclusive as run


def verify_inventory(session, required):
    """A known empty torchinductor directory is metadata, not an artifact."""
    actual = {p.name for p in session.iterdir()}
    optional = "torchinductor_gbk"
    run.model.need(actual in (required, required | {optional}), "session inventory")
    if optional in actual:
        path = session / optional
        run.model.need(path.is_dir() and not path.is_symlink() and not any(path.iterdir()),
                       "runtime cache must be an empty regular directory")
    return optional in actual


def consume(plan_path, plan_sha, session_sha):
    plan = run.verify(plan_path, plan_sha)
    session = run.model.read_checked(run.SESSION / "session_report.json", session_sha)
    run.model.need(session["schema"] == run.NAME + ".session_v1" and session["plan_sha256"] == plan_sha
                   and session["status"] == "recorded_development_unconsumed" and session["error"] is None
                   and session["physical_execution"] is True
                   and all(session[k] is False for k in ("equal_total_compute", "independent_confirmation", "P2_admission"))
                   and type(session["wall_seconds"]) in (int, float)
                   and math.isfinite(session["wall_seconds"]) and session["wall_seconds"] >= 0,
                   "session status")
    run.model.need([r["base"] for r in session["rows"]] == list(run.model.CASES), "root order")
    summaries, names = [], {"session_report.json"}
    for row, recipe in zip(session["rows"], plan["recipes"], strict=True):
        base = recipe["case"]["base"]
        folder = run.SESSION / (run.physical.NAME + f"_MULTI_B{base}_01")
        names.add(folder.name)
        run.model.need(row["status"] == "recorded_development_unconsumed" and row["output"] == str(folder)
                       and folder.is_dir() and not folder.is_symlink(), "root folder")
        report = run.model.read_checked(folder / "report.json", row["report_sha256"])
        binding = json.loads((folder / "binding.json").read_bytes())
        run.model.need(binding == {"plan_sha256": plan_sha, "base": base,
                                   "old_plan_sha256": recipe["case"]["old_physical_prep_sha256"]}
                       and report["schema"] == run.NAME + ".root_report_v1"
                       and report["plan_sha256"] == plan_sha
                       and run.model.host.base.file_sha(folder / "journal.jsonl") == report["journal_sha256"],
                       "root report/journal binding")
        raw = (folder / "journal.jsonl").read_bytes()
        run.model.need(raw.endswith(b"\n") and b"\n\n" not in raw and b"\r" not in raw, "journal framing")
        events = [run.physical.native.decode(x) for x in raw.splitlines()]
        summary = run.one_saved.audit_events(events, report, recipe["prepared"], plan_sha, folder)
        run.model.need(run.model.host.base.file_sha(folder / "report.json") == row["report_sha256"]
                       and run.model.host.base.file_sha(folder / "journal.jsonl") == report["journal_sha256"],
                       "root source drift")
        summaries.append({"base": base, "root": recipe["case"]["root"],
                          "native_root_sha256": recipe["case"]["native_root_sha256"],
                          "report_sha256": row["report_sha256"], "summary": summary})
    cache = verify_inventory(run.SESSION, names)
    return {"schema": run.NAME + ".consume_v2", "status": "verified_saved_known_roots_development",
            "plan_sha256": plan_sha, "session_report_sha256": session_sha, "roots": summaries,
            "empty_torchinductor_directory_present": cache,
            "historical_sra_authenticated": False, "equal_total_compute": False,
            "independent_confirmation": False, "P2_admission": False,
            "limitations": ["saved-record consistency, not independent proof of physical execution",
                            "two exposed development roots", "historical environment closure incomplete",
                            "original consumer rejected an empty torchinductor directory"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--session-report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = run.path_new(args.output, "_CONSUME_")
    result = consume(args.plan, args.plan_sha256, args.session_report_sha256)
    print(json.dumps({"receipt": str(output), "sha256": run.model.host.base.write_new(output, result)}))


if __name__ == "__main__":
    main()
