"""Reconcile the fixed ratio34 batch against six persisted journals and receipts."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import codex_p3_quota_journal_20261030_v1 as journal

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "results/codex_p3_ratio34_delivery_20261030_v1_RUN_CODEX01"
BATCH_SHA = "f2e800d9eaa75b3353837e661bcfe875f3bdb7e65b2d99d09fcee1096e0ade0f"
SLOTS = [(p, a) for p in (20, 30) for a in ("uniform", "signal", "reverse")]
AXES = ("action_encoder", "predict", "predictor", "pred_proj", "jacobian",
        "vjp_directions", "grad_api", "raw_backward_hook", "svd", "metric", "scheduler")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconcile(events, receipt, row):
    actual = Counter()
    returned = Counter()
    identities, seen_returns = {}, set()
    for event in events:
        if event["kind"] == "resource_attempt":
            entry = event["entry"]
            require(entry["index"] not in identities and entry["scope"][:2] ==
                    [row["point"], row["arm"]] and entry["resource"] in AXES,
                    "attempt identity")
            identities[entry["index"]] = entry
            actual[entry["resource"]] += 1
        elif event["kind"] == "resource_return":
            entry = event["entry"]
            initial = identities.get(entry["index"])
            require(initial is not None and entry["index"] not in seen_returns and
                    entry == dict(initial, completed=True,
                    ended_seconds=entry.get("ended_seconds")), "return identity")
            seen_returns.add(entry["index"])
            returned[entry["resource"]] += 1
    actual = {axis: actual[axis] for axis in AXES}
    returned = {axis: returned[axis] for axis in AXES}
    require(actual == returned == row["actual"] and set(row["caps"]) == set(AXES)
            and all(type(actual[k]) is int and actual[k] <= row["caps"][k] for k in AXES),
            "axis accounting")
    saved = receipt["rows"][0]
    require(receipt["status"] == "semantic_saved_checked_not_scientifically_admitted" and
            len(receipt["rows"]) == 1 and saved["attempted"] == actual and
            (saved["point"], saved["arm"]) == (row["point"], row["arm"]) and
            saved["allocation"] == row["allocation"], "saved summary")
    candidates = [e for e in events if e["kind"] == "candidate_complete"]
    keys = [(e["scope"][3], e["candidate"]["index"]) for e in candidates]
    expected = [(root, i) for root in (0, 1) for i in range(row["allocation"][root])]
    require(keys == expected and len(keys) == actual["metric"], "candidate roster")
    for root in (0, 1):
        records = [e["candidate"] for e in candidates if e["scope"][3] == root]
        require(saved["winners"][root] == min(records, key=lambda x: (x["cost"], x["index"])),
                "winner")
    return {"point": row["point"], "arm": row["arm"], "allocation": row["allocation"],
            "actual": actual, "caps": row["caps"], "wall_seconds": row["attempt"]["wall_seconds"],
            "winner_indices": [w["index"] for w in saved["winners"]],
            "candidates": [list(key) for key in keys]}


def audit():
    require(sha(BATCH / "report.json") == BATCH_SHA, "batch hash")
    report = json.loads((BATCH / "report.json").read_bytes())
    rows = report["rows"]
    require(report["error"] is None and [(x["point"], x["arm"]) for x in rows] == SLOTS,
            "batch roster")
    events = journal.read(BATCH / "events.jsonl", report["journal_tail"], 2 * 1024**2)
    require([e["row"] for e in events if e["kind"] == "slot_result"] == rows and
            events[-1]["kind"] == "batch_terminal" and events[-1]["failure"] is None,
            "batch journal")
    result = []
    for row in rows:
        require(row["delivered_and_saved"] and row["attempt"]["exit_code"] == 0 and
                not row["attempt"]["deadline_exceeded"] and
                0 <= row["attempt"]["wall_seconds"] <= 30, "slot delivery")
        run, receipt = Path(row["run"]), Path(row["receipt"])
        require(sha(run / "report.json") == row["report_sha256"] and
                sha(receipt) == row["receipt_sha256"], "slot hashes")
        report_slot = json.loads((run / "report.json").read_bytes())
        require(report_slot["status"] == "panel_completed_unverified" and
                report_slot["point"] == row["point"] and report_slot["arm"] == row["arm"] and
                report_slot["model_before"] == report_slot["model_after"], "slot report")
        entries = journal.read(run / "events.jsonl", report_slot["journal_tail"], 512 * 1024**2)
        result.append(reconcile(entries, json.loads(receipt.read_bytes()), row))
    difference = {}
    for point in (20, 30):
        s, r = [next(x for x in result if x["point"] == point and x["arm"] == arm)
                for arm in ("signal", "reverse")]
        difference[str(point)] = len({tuple(x) for x in s["candidates"]} ^
                                     {tuple(x) for x in r["candidates"]})
    return {"status": "saved_reconciled_not_scientifically_admitted",
            "batch_sha256": BATCH_SHA, "rows": result, "signal_reverse_symmetric_difference": difference,
            "model_reexecuted": False, "physical_executed": False, "scientific_pass": False,
            "nested_axes_not_additive": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    require(path.is_absolute() and path.parent == ROOT / "results" and
            path.name.startswith("codex_p3_ratio34_audit_20261030_v1_"), "exclusive output")
    data = json.dumps(audit(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    print(hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
