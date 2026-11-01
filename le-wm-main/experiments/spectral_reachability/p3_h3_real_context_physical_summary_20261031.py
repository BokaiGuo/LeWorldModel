"""Consume the entire fixed physical panel and pair actual cp15-replanned outcomes."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from . import p3_h3_real_context_physical_20261031 as source
from .codex_p1_h10_paired_table_20261030_main import paired_difference, physical_metrics

ROOT, need, host = source.ROOT, source.need, source.host
NAME = "p3_h3_real_context_physical_summary_20261031"
MANIFEST = ROOT / "results/p3_h3_real_context_physical_20261031_prepare_RUN01.json"
MANIFEST_SHA = "2912800417ace34400c7f858a7d971d2427b2e094319fefd341d6f9721e1f5ce"


def validate_guards(p, report, events):
    for i, row in enumerate(report["rows"]):
        local = [event for event in events if event.get("branch") == i]
        guards = [(j, e) for j, e in enumerate(local) if e["kind"] == "cp15_guard_passed"]
        calls = [(j, e) for j, e in enumerate(local) if e["kind"] == "operation_attempted"
                 and e.get("resource") == "public_step" and e.get("phase") == "plan"
                 and e.get("step") == 16]
        need(len(guards) == len(calls) == 1 and guards[0][0] < calls[0][0]
             and guards[0][1]["pixel_sha256"] == p["guards"][i]["pixel_sha256"],
             "cp15 guard ordering")
        checkpoints = [cp for cp in row["checkpoints"] if cp["phase"] == "plan" and cp["step"] == 15]
        need(len(checkpoints) == 1 and
             all(checkpoints[0][key] == p["guards"][i][key] for key in source.GUARD_KEYS),
             "saved cp15")


def pair_batch(p, report):
    results = []
    for i in range(0, 6, 2):
        old, new = report["rows"][i:i + 2]
        a, b = p["identities"][i:i + 2]
        need(a == b and old["goal"] == new["goal"], "pair identity")
        endpoint = lambda row: physical_metrics(row["final_state"], row["goal"]) if row["full_horizon"] else None
        old_metrics, new_metrics = endpoint(old), endpoint(new)
        results.append({"identity": a, "old_status": old["status"], "new_status": new["status"],
                        "old_returned": old["own_returned"], "new_returned": new["own_returned"],
                        "old_native_success": old["native_success"], "new_native_success": new["native_success"],
                        "old_endpoint": old_metrics, "new_endpoint": new_metrics,
                        "new_minus_old": paired_difference(new_metrics, old_metrics),
                        "old_last_metrics": old["final_metrics"], "new_last_metrics": new["final_metrics"],
                        "pair_key": hashlib.sha256(json.dumps({
                            "root": p["plans"][i]["root_sha256"], "task": p["plans"][i]["task_id"],
                            "old_actions": p["plans"][i]["raw_float32_sha256"],
                            "new_actions": p["plans"][i + 1]["raw_float32_sha256"],
                        }, sort_keys=True).encode()).hexdigest()})
    return results


def aggregate(rows):
    need(len(rows) == 54 and sum(row["identity"]["padding"] for row in rows) == 2,
         "fixed denominator and padding")
    rows = [row for row in rows if not row["identity"]["padding"]]
    need(len(rows) == 52 and len({(r["identity"]["logical_pair"], r["identity"]["source_variant"])
                                  for r in rows}) == 52, "fixed identities")
    groups = {}
    for row in rows:
        groups.setdefault(row["pair_key"], []).append(row)
    for group in groups.values():
        first = group[0]
        for item in group[1:]:
            need(all(item[k] == first[k] for k in ("old_status", "new_status", "old_returned", "new_returned",
                                                    "old_native_success", "new_native_success",
                                                    "old_endpoint", "new_endpoint")), "duplicate outcome mismatch")
    complete = [group[0] for group in groups.values() if group[0]["new_minus_old"] is not None]
    counts = {}
    for metric in ("object_position_error", "object_angle_error_wrapped",
                   "pusher_position_error", "native_joint_xy_error"):
        values = [row["new_minus_old"][metric] for row in complete]
        counts[metric] = {"lower": sum(v < -1e-9 for v in values),
                          "tie": sum(abs(v) <= 1e-9 for v in values),
                          "higher": sum(v > 1e-9 for v in values)}
    return {"schema": NAME, "rows": rows, "padding_excluded": 2, "fixed_pairs": 52,
            "unique_action_pairs": len(groups), "complete_unique_pairs": len(complete),
            "unique_native_success": {"old": sum(g[0]["old_native_success"] for g in groups.values()),
                                      "new": sum(g[0]["new_native_success"] for g in groups.values())},
            "direction_counts_new_minus_old": counts,
            "limitations": "Known roots; deduplicated action pairs are not independent tasks. "
            "Offline replanning and extra compute; missing endpoints are not imputed.",
            "P1_admission": False, "P3_admission": False, "independent_confirmation": False,
            "equal_budget_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    need(host.sha(MANIFEST) == MANIFEST_SHA, "external manifest hash")
    manifest, data = source.build()
    need(json.loads(MANIFEST.read_bytes()) == manifest, "reconstructed manifest")
    rows, digests = [], {}
    for i in range(18):
        path = ROOT / f"results/p3_h3_real_context_physical_20261031_batch{i:02d}_RUN01"
        p = source.prepare_batch(manifest, i)
        report_path, prepare_path = path / "report.json", path / "prepare.json"
        report_sha, prepare_sha = host.sha(report_path), host.sha(prepare_path)
        source.old.saved.consume(path, report_sha, prepare_sha, p, data["references"])
        report = json.loads(report_path.read_bytes())
        events = source.batch.kernel.journal(path / "journal.jsonl", report["journal_chain_tail"])
        validate_guards(p, report, events)
        rows.extend(pair_batch(p, report))
        need(host.sha(report_path) == report_sha and host.sha(prepare_path) == prepare_sha, "batch drift")
        digests[str(report_path)] = report_sha
        digests[str(prepare_path)] = prepare_sha
        print(json.dumps({"batch": i, "status": "saved_checks_passed"}), flush=True)
    result = aggregate(rows)
    need(host.sha(MANIFEST) == MANIFEST_SHA
         and all(host.sha(path) == digest for path, digest in manifest["bindings"].items()), "source drift")
    result["manifest_sha256"], result["batch_sha256"] = MANIFEST_SHA, digests
    result["analysis_sha256"] = host.sha(__file__)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sha256": host.sha(out), "pairs": result["unique_action_pairs"]}), flush=True)


if __name__ == "__main__":
    main()
