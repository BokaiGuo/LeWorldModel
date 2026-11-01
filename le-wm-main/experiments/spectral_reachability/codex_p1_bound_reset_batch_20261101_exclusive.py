"""Bounded CPU-only validation of explicitly pinned physical reset reports."""
import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import codex_p1_original_reset_inventory_20261101 as core

REACHABLE = ("p1_constructive_reachable_source_20261027_v1_main01",
             "9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9")
MOVING = ("p1_constructive_moving_history_source_20261027_v1_main01",
          "c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6")
PINS = {
    "p1_constructive_gradient_mpc_physical_20261027_v1_main01": (REACHABLE, "134356eb4524998055e46b83d30cfb44369195356b8890ac1a6aea170208a75a"),
    "p1_constructive_mppi_physical_20261027_v1_main01": (REACHABLE, "2f5221e8684100ab8ad78b96ce77e6a17a6b5f0ee908efca3a5fa9fba544ffdc"),
    "p1_constructive_observed_suffix_physical_20261027_v1_main01": (REACHABLE, "f864cc80a88eb289b578d271679d0bc895043a3b7bc798ae141097797a555b5e"),
    "p1_constructive_sampling_physical_20261027_v1_main01": (REACHABLE, "b67710a09b109c2276556d10b5fdbc3bca8e4ce44d31b302cd65ee849950c3ef"),
    "p1_constructive_plan_physical_20261027_v1_main01": (REACHABLE, "d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300"),
    "p1_constructive_matched_ridge_gn_physical_20261027_v1_main01": (REACHABLE, "47d46ffd9664d63e19a1906696b448aa9f06c9259f216d9a343b42ac554752ff"),
    "p1_constructive_matched_sampling_physical_20261027_v1_main01": (REACHABLE, "b735655a98ecb52f10967d2a2f9cbfe53ef72a00ccc064799e0be275fa1a01a7"),
    "p1_constructive_grasp_physical_20261027_v1_main01": (REACHABLE, "bf7f15cf9d5aad4ca7923c4a116f6703c5e559b496c50488de55a4a94394f606"),
    "p1_moving_baseline_physical_20261027_v1_kino_main01": (MOVING, "2db7370214bf62810c21ad8778c1d58bc397dbe8f08c9f5ef1db435468abc655"),
    "p1_moving_baseline_physical_20261027_v1_main01": (MOVING, "6a9f4cf602a328cf5cb0fbe82296fc989a109a6859287c2efb703bebf44511db"),
    "p1_moving_extended_baseline_physical_20261027_v1_main01": (MOVING, "f05ce7e2d773bb046c6c8ba905c9e86c367c7e19ad09174d4509e1823a297038"),
}


def verify(report_raw, journal_raw, source_raw, report_sha, source_sha):
    if core.sha(report_raw) != report_sha or core.sha(source_raw) != source_sha:
        raise ValueError("pinned report or source mismatch")
    if len(journal_raw) > 16 * 1024 * 1024 or not journal_raw.endswith(b"\n"):
        raise ValueError("journal limit or truncation")
    report, source = core.parse(report_raw), core.parse(source_raw)
    if report.get("sourcepack_sha256") != source_sha or report.get("journal_sha256") != core.sha(journal_raw):
        raise ValueError("report source/journal mismatch")
    events, previous = [], None
    for index, line in enumerate(journal_raw.splitlines()):
        event = core.parse(line)
        digest = core.sha(json.dumps({k: v for k, v in event.items() if k != "event_sha256"},
                                     sort_keys=True, allow_nan=False).encode())
        if event.get("index") != index or event.get("previous_sha256") != previous or event.get("event_sha256") != digest:
            raise ValueError("journal chain mismatch")
        events.append(event)
        previous = digest
    if not events or events[0].get("kind") != "run_started" or events[-1].get("kind") != "run_finished":
        raise ValueError("journal boundary mismatch")
    prepare, end = events[0]["prepare"], events[-1]
    if (prepare.get("sourcepack_sha256") != source_sha or prepare.get("contract") != report.get("contract")
            or end.get("counts") != report.get("counts") or end.get("status") != report.get("status")
            or (report.get("journal_chain_tail") is not None and report["journal_chain_tail"] != previous)):
        raise ValueError("prepare, report, finish or chain mismatch")
    slots = {}
    for row in source["rows"]:
        slot = row["slot"]
        if slot["index"] in slots:
            raise ValueError("duplicate source slot")
        slots[slot["index"]] = slot
    branches = {}
    for index, row in enumerate(report["rows"]):
        branch = row.get("branch_index", index)
        key = (branch, row["root_slot"], row["arm"], row["selection_digest"])
        if key in branches or row["root_slot"] not in slots:
            raise ValueError("duplicate branch or source slot absent")
        branches[key] = True
    rows, pending, seen = [], {}, set()
    for event in events:
        if event.get("resource") != "reset":
            continue
        branch = tuple(event.get(k) for k in ("branch", "root_slot", "arm", "selection_digest"))
        kind, seed = event.get("kind_of_reset"), event.get("seed")
        if branch not in branches or kind not in ("original_seed", "placement"):
            raise ValueError("unmatched reset branch")
        if seed != (slots[branch[1]]["reset_seed"] if kind == "original_seed" else 0):
            raise ValueError("seed mismatch")
        key = branch + (kind, seed)
        if event.get("kind") == "operation_attempted":
            if key in seen:
                raise ValueError("duplicate reset attempt")
            seen.add(key)
            row = {"branch": branch[0], "source_slot": branch[1], "arm": branch[2],
                   "selection_digest": branch[3], "kind_of_reset": kind, "seed": seed,
                   "start_line": event["index"] + 1, "start_event_sha256": event["event_sha256"],
                   "completion_line": None, "completion_event_sha256": None,
                   "physical_root_equivalence": "unknown", "independent_root": "unknown"}
            rows.append(row)
            pending[key] = row
        elif event.get("kind") == "operation_completed":
            if key not in pending:
                raise ValueError("unpaired completion")
            row = pending.pop(key)
            row["completion_line"] = event["index"] + 1
            row["completion_event_sha256"] = event["event_sha256"]
        else:
            raise ValueError("unexpected reset kind")
    counts = report["counts"]["reset"]
    if len(rows) != counts["attempted"] or sum(r["completion_line"] is not None for r in rows) != counts["completed"]:
        raise ValueError("reset count mismatch")
    return {"sourcepack_sha256": source_sha, "report_sha256": report_sha,
            "journal_sha256": core.sha(journal_raw), "records": rows}


def build(root=core.ROOT, pins=PINS):
    root = Path(root)
    verified, unknown = [], []
    for name, ((source_name, source_sha), report_sha) in pins.items():
        directory = root / "results" / name
        source = root / "results" / source_name / "sourcepack.json"
        try:
            item = verify((directory / "report.json").read_bytes(), (directory / "journal.jsonl").read_bytes(),
                          source.read_bytes(), report_sha, source_sha)
            item.update({"report_path": str(directory / "report.json"), "journal_path": str(directory / "journal.jsonl"),
                         "sourcepack_path": str(source)})
            verified.append(item)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            unknown.append({"report_path": str(directory / "report.json"), "status": "unknown", "reason": str(exc)})
    return {"schema": "codex_p1_bound_reset_batch_20261101_exclusive.v1",
            "scope": "11 explicitly pinned reports, not a complete historical search",
            "verified": verified, "unknown_sources": unknown,
            "reset_records": sum(len(x["records"]) for x in verified),
            "physical_root_deduplication": "unknown", "global_history_coverage": "unknown",
            "training_exclusion": "unknown", "freshness": "unknown",
            "independent_roots_certified": False, "P1_admitted": False, "new_environment_or_gpu_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build()
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"output": str(args.output), "sha256": core.sha(data), "verified": len(result["verified"]),
                      "unknown": len(result["unknown_sources"]), "reset_records": result["reset_records"]}))


if __name__ == "__main__":
    main()
