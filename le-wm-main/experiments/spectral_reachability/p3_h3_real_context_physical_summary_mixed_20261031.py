"""Consume the fixed H3 panel from its explicitly recorded RUN01/RUN02 batches."""
import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from . import p3_h3_real_context_physical_20261031 as source
from . import p3_h3_real_context_physical_summary_20261031 as base
from . import p3_raw_h3_guard_continuation_20261031 as guard
from . import p3_raw_h3_partial_pair_20261031 as partial

ROOT, host, need = source.ROOT, source.host, source.need
NAME = "p3_h3_real_context_physical_summary_mixed_20261031"
MANIFEST = ROOT / "results/p3_h3_real_context_physical_20261031_prepare_RUN01.json"
MANIFEST_SHA = "2912800417ace34400c7f858a7d971d2427b2e094319fefd341d6f9721e1f5ce"

# These are intentionally explicit.  RUN02 is not inferred from timestamps or
# selected by outcome; it is the already recorded replacement for batches 0/1.
BATCH_RUN = {0: "RUN02", 1: "RUN02", **{i: "RUN01" for i in range(2, 18)}}


def _frame_reader(directory):
    def read(name, digest):
        need(Path(name).name == name and host.sha(directory / name) == digest,
             "frame path/hash")
        with Image.open(BytesIO((directory / name).read_bytes())) as image:
            return np.asarray(image.convert("RGB")).copy()
    return read


def validate_partial(p, report, events, data, directory):
    failures = guard.allowed_partial(report, events)
    need(report["status"] == "partial_or_failed" and report["errors"] == []
         and len(report["rows"]) == 6 and report["scientific_pass"] is False,
         "partial report contract")
    for row, plan in zip(report["rows"], p["plans"], strict=True):
        arm = f"d{plan['decision_index']}_c{plan['candidate']}_{plan['variant']}"
        if row["status"] != "failed":
            partial.validate_complete(
                row, plan, data["references"][plan["root_slot"]], events,
                _frame_reader(directory),
            )
        else:
            need(row["branch_index"] >= 0 and row["first_error"] is not None
                 and row["own_returned"] < 20 and row["full_horizon"] is False,
                 "partial failed row")
        need(row["selected_plan"] == {**plan, "arm": arm}, "partial plan binding")
    return failures


def consume_batch(manifest, data, index, run):
    p = source.prepare_batch(manifest, index)
    directory = ROOT / f"results/p3_h3_real_context_physical_20261031_batch{index:02d}_{run}"
    report_path, prepare_path = directory / "report.json", directory / "prepare.json"
    need(directory.is_dir() and report_path.is_file() and prepare_path.is_file(),
         "batch artifacts")
    report_sha, prepare_sha = host.sha(report_path), host.sha(prepare_path)
    report = json.loads(report_path.read_bytes())
    # The normal saved consumer is strict for terminal batches.  Partial
    # batches use the dedicated retained-canvas contract and never get
    # upgraded to a completed result.
    if report["status"] == "recorded_not_admitted":
        source.old.saved.consume(directory, report_sha, prepare_sha, p, data["references"])
        events = source.batch.kernel.journal(directory / "journal.jsonl",
                                             report["journal_chain_tail"])
        base.validate_guards(p, report, events)
        state = "terminal_saved_checked"
        censored = 0
    elif report["status"] == "partial_or_failed":
        events = source.batch.kernel.journal(directory / "journal.jsonl",
                                             report["journal_chain_tail"])
        censored = validate_partial(p, report, events, data, directory)
        state = "partial_saved_checked"
    else:
        raise ValueError(f"unexpected batch status: {report['status']}")
    need(host.sha(report_path) == report_sha and host.sha(prepare_path) == prepare_sha,
         "batch drift")
    return {
        "index": index,
        "run": run,
        "status": report["status"],
        "audit_status": state,
        "censored_rows": censored,
        "report_sha256": report_sha,
        "prepare_sha256": prepare_sha,
        "rows": report["rows"],
        "prepare": p,
    }


def pair_batch(audit):
    p, rows = audit["prepare"], audit["rows"]
    result = []
    for i in range(0, 6, 2):
        old, new = rows[i:i + 2]
        ida, idb = p["identities"][i:i + 2]
        need(ida == idb and old["goal"] == new["goal"], "pair identity")
        endpoint = lambda row: base.physical_metrics(
            row["final_state"], row["goal"]) if row["full_horizon"] else None
        old_metrics, new_metrics = endpoint(old), endpoint(new)
        result.append({
            "identity": ida,
            "old_status": old["status"], "new_status": new["status"],
            "old_returned": old["own_returned"], "new_returned": new["own_returned"],
            "old_native_success": old["native_success"],
            "new_native_success": new["native_success"],
            "old_endpoint": old_metrics, "new_endpoint": new_metrics,
            "new_minus_old": base.paired_difference(new_metrics, old_metrics),
            "old_last_metrics": old["final_metrics"],
            "new_last_metrics": new["final_metrics"],
            "pair_key": hashlib.sha256(json.dumps({
                "root": p["plans"][i]["root_sha256"],
                "task": p["plans"][i]["task_id"],
                "old_actions": p["plans"][i]["raw_float32_sha256"],
                "new_actions": p["plans"][i + 1]["raw_float32_sha256"],
            }, sort_keys=True).encode()).hexdigest(),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.parent == ROOT / "results"
         and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    need(host.sha(MANIFEST) == MANIFEST_SHA, "external manifest hash")
    manifest, data = source.build()
    need(json.loads(MANIFEST.read_bytes()) == manifest, "reconstructed manifest")

    audits, pairs, batch_digests = [], [], {}
    for index in range(18):
        run = BATCH_RUN[index]
        audit = consume_batch(manifest, data, index, run)
        audits.append({k: audit[k] for k in
                       ("index", "run", "status", "audit_status", "censored_rows",
                        "report_sha256", "prepare_sha256")})
        batch_digests[f"{index:02d}_{run}"] = {
            "report": audit["report_sha256"], "prepare": audit["prepare_sha256"]}
        pairs.extend(pair_batch(audit))
        print(json.dumps({"batch": index, "run": run,
                          "status": audit["audit_status"]}), flush=True)

    need(len(pairs) == 54 and sum(x["identity"]["padding"] for x in pairs) == 2,
         "fixed denominator and padding")
    real_pairs = [x for x in pairs if not x["identity"]["padding"]]
    need(len(real_pairs) == 52 and len({
        (x["identity"]["logical_pair"], x["identity"]["source_variant"])
        for x in real_pairs
    }) == 52, "fixed identities")
    complete = [x for x in real_pairs if x["new_minus_old"] is not None]
    counts = {}
    for metric in ("object_position_error", "object_angle_error_wrapped",
                   "pusher_position_error", "native_joint_xy_error"):
        values = [x["new_minus_old"][metric] for x in complete]
        counts[metric] = {
            "lower": sum(v < -1e-9 for v in values),
            "tie": sum(abs(v) <= 1e-9 for v in values),
            "higher": sum(v > 1e-9 for v in values),
        }
    result = {
        "schema": NAME,
        "batch_selection": BATCH_RUN,
        "batches": audits,
        "rows": real_pairs,
        "padding_excluded": 2,
        "fixed_pairs": 52,
        "unique_action_pairs": len({x["pair_key"] for x in real_pairs}),
        "complete_unique_pairs": len({x["pair_key"] for x in complete}),
        "direction_counts_new_minus_old": counts,
        "batch_sha256": batch_digests,
        "manifest_sha256": MANIFEST_SHA,
        "limitations": (
            "Known roots; RUN02 is an explicit recorded batch replacement for "
            "RUN01 batches 0/1; partial canvas-guard rows are censored and not "
            "imputed; deduplicated action pairs are not independent tasks."
        ),
        "P1_admission": False,
        "P3_admission": False,
        "independent_confirmation": False,
        "equal_budget_certified": False,
    }
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sha256": host.sha(out), "pairs": result["unique_action_pairs"],
                      "complete": result["complete_unique_pairs"]}), flush=True)


if __name__ == "__main__":
    main()
