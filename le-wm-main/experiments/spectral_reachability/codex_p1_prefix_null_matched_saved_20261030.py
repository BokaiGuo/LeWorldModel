"""Saved-only verification of every fixed prefix-null matched probe."""
import argparse
import json
from pathlib import Path

from . import codex_p1_prefix_null_matched_20261030 as run
from . import p1_matched_censored_consumer_20261030_codex as checks


def consume(directory, report_sha, prepared, prepare_sha):
    p, _ = run.verify(prepared, prepare_sha)
    d = Path(directory)
    run.need(d.is_absolute() and d.parent == run.ROOT / "results" and
             d.name.startswith(run.NAME+"_run_") and d.resolve() == d and d.is_dir(), "run directory")
    path = d / "run_report.json"
    run.native.actual_hash(report_sha)
    run.need(path.is_file() and not path.is_symlink() and run.sha(path) == report_sha, "run report")
    report = run.native.read(path, cap=4*1024**2)
    run.need(report["schema"] == run.NAME+".run.v1" and report["status"] == "recorded_not_consumed"
             and report["fixed_branches"] == 336 and report["fixed_anchors"] == 16
             and report["P1_admission"] is False and report["independent_confirmation"] is False
             and len(report["shards"]) == 16, "complete fixed run")
    bp = d / "run_binding.json"
    run.need(bp.is_file() and not bp.is_symlink() and run.sha(bp) == report["run_binding_sha256"],
             "run binding digest")
    bindings = dict(p["bindings"], **{str(Path(prepared)): prepare_sha})
    expected = dict(prepared=str(prepared), prepare_sha256=prepare_sha, bindings=bindings,
                    fixed_branches=336, P1_admission=False, independent_confirmation=False)
    run.need(run.native.read(bp, cap=64*1024**2) == expected, "run binding")
    rows = []
    for i, (anchor, shard) in enumerate(zip(p["anchors"], report["shards"], strict=True)):
        run.need(shard["anchor"] == i and shard["status"] == "recorded_not_consumed", "ordered shard")
        rows.append(checks.check_shard(d / f"anchor_{i:02d}", anchor, p["probes"], shard["report_sha256"]))
    run.need(all(run.sha(file) == h for file, h in bindings.items()) and
             run.sha(path) == report_sha and run.sha(bp) == report["run_binding_sha256"], "final drift")
    counts = {s: sum(row["statuses"].count(s) for row in rows) for s in
              ("probe_completed", "native_stopped", "censored_out_of_domain")}
    run.need(sum(counts.values()) == 336, "fixed denominator")
    return dict(status="saved_checks_passed_exploratory_only", fixed_pairs=8, fixed_branches=336,
                counts=counts, rows=rows, P1_admission=False, independent_confirmation=False,
                equal_budget_certified=False, runtime_authenticity_proved=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", required=True)
    ap.add_argument("--report-sha", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--prepare-sha", required=True)
    a = ap.parse_args()
    print(json.dumps(consume(a.directory, a.report_sha, a.prepared, a.prepare_sha),
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
