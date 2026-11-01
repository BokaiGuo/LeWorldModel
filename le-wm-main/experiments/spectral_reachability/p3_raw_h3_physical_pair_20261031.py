"""Pair independently validated H3 old/new trajectories at the real step-15 state."""
import argparse
import hashlib
import json
from pathlib import Path

from . import p3_raw_h3_physical_source_20261031 as source

require = source.require


def pair_rows(report):
    require(len(report["rows"]) == 6, "six fixed rows")
    result = []
    for old, new in zip(report["rows"][::2], report["rows"][1::2], strict=True):
        a, b = old["selected_plan"], new["selected_plan"]
        require(a["variant"] == "old" and b["variant"] == "new"
                and all(a[k] == b[k] for k in ("root_slot", "root_sha256", "task_id",
                    "decision_index", "candidate", "policy", "prefix_float32_sha256")),
                "pair identity")
        cps = []
        for row in (old, new):
            matches = [x for x in row["checkpoints"] if x["phase"] == "plan" and x["step"] == 15]
            require(len(matches) <= 1, "duplicate step-15 checkpoint")
            cps.append(matches[0] if matches else None)
        exact = None if any(x is None for x in cps) else all(
            cps[0][key] == cps[1][key] for key in ("state", "snapshot", "pixel_sha256"))
        # Native stopping exactly at step 20 still has a real horizon endpoint.
        # Earlier stops remain censored; never synthesize unexecuted steps.
        def endpoint(row):
            return (row["final_metrics"] if row["own_returned"] == 20
                    and row["status"] in ("horizon_completed", "native_success", "native_stopped")
                    else None)
        old_end, new_end = endpoint(old), endpoint(new)
        keys = ("object_position_error", "wrapped_object_angle_error", "native_distance")
        delta = None if old_end is None or new_end is None or exact is not True else {
            key: new_end[key] - old_end[key] for key in keys}
        result.append({"root_slot": a["root_slot"], "decision_index": a["decision_index"],
                       "candidate": a["candidate"], "step15_exact": exact,
                       "old_status": old["status"], "new_status": new["status"],
                       "old_returned": old["own_returned"], "new_returned": new["own_returned"],
                       "old_last_success": old["native_success"], "new_last_success": new["native_success"],
                       "old_endpoint": old_end, "new_endpoint": new_end,
                       "new_minus_old_endpoint": delta})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", required=True)
    ap.add_argument("--report-sha", required=True)
    ap.add_argument("--prepare-sha", required=True)
    ap.add_argument("--batch", required=True, type=int)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    output = Path(args.output)
    require(output.is_absolute() and output.parent == source.ROOT / "results"
            and output.name.startswith("p3_raw_h3_physical_pair_20261031_")
            and not output.exists() and not output.is_symlink(), "exclusive output")
    verified = source.consume(args.directory, args.report_sha, args.prepare_sha, args.batch)
    report = source.host.helper.read(Path(args.directory) / "report.json")
    rows = pair_rows(report)
    require(source.host.sha(Path(args.directory) / "report.json") == args.report_sha, "report drift")
    result = {"schema": "p3_raw_h3_physical_pair_20261031",
              "source_report_sha256": args.report_sha, "source_prepare_sha256": args.prepare_sha,
              "batch_index": args.batch, "saved_validation": verified["status"], "pairs": rows,
              "known_roots": True, "independent_confirmation": False,
              "equal_budget_certified": False, "P3_admission": False}
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"sha256": hashlib.sha256(raw).hexdigest(), "pairs": len(rows)}))


if __name__ == "__main__":
    main()
