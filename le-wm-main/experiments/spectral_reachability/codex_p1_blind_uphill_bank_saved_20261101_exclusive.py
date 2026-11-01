"""Read-only CPU verification of fixed blind-bank model selection."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_blind_uphill_bank_20261101_exclusive as source


def verify(folder, report_sha):
    folder = Path(folder)
    report_path = folder / "report.json"
    source.host.require(source.host.sha(report_path) == report_sha, "report SHA")
    report = json.loads(report_path.read_bytes())
    event_path = folder / "events.jsonl"
    source.host.require(source.host.sha(event_path) == report["events_sha256"], "events SHA")
    lines = event_path.read_bytes().splitlines()
    events = [json.loads(line) for line in lines]
    source.host.require(len(events) == 96 and report["predict_calls"] == report["action_encode_calls"] == 48
                        and report["model_before"] == report["model_after"] and
                        report["physical_future_used"] is False and report["P1_admission"] is False,
                        "report/call count")
    out = []
    for slot, row in enumerate(report["rows"]):
        task, bank, _, _ = source.load(slot)
        source.host.require(row["slot"] == slot, "slot")
        for i in range(8):
            for k in range(3):
                pos = 2 * (slot * 24 + i * 3 + k)
                expected = {"slot": slot, "candidate": i, "packet": k}
                a, r = events[pos:pos + 2]
                source.host.require(a == {"event": "attempt", **expected}
                                    and {key: r[key] for key in ("slot", "candidate", "packet")} == expected
                                    and r["event"] == "return", "event order")
                if k == 0:
                    source.host.require(r["mse"] == row["prefix_mse"][i], "prefix score")
                if k == 2:
                    source.host.require(r["mse"] == row["terminal_mse"][i], "terminal score")
        index = source.choose(row["root_mse"], row["prefix_mse"], row["terminal_mse"])
        eligible = [i for i in range(8) if row["prefix_mse"][i] >= row["root_mse"]]
        source.host.require(index == row["selected_index"] and eligible == row["eligible"], "selection")
        if index is not None:
            source.host.require(np.array_equal(bank[index], np.asarray(row["selected_raw"], np.float32)),
                                "selected action source")
        out.append({"slot": slot, "selected_index": index, "eligible_count": len(eligible),
                    "root_mse": row["root_mse"],
                    "selected_prefix_mse": None if index is None else row["prefix_mse"][index],
                    "selected_terminal_mse": None if index is None else row["terminal_mse"][index],
                    "terminal_improves_root": False if index is None else row["terminal_mse"][index] < row["root_mse"]})
    return {"status": "saved_selection_consistent_not_independent_inference", "report_sha256": report_sha,
            "rows": out, "P1_admission": False, "physical_execution": False,
            "limitation": "event self-report and action/source consistency; model predictions not recomputed"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    for key in ("directory", "report-sha256", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    value = verify(args.directory, args.report_sha256)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(args.output).open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"sha256": hashlib.sha256(payload).hexdigest(), **value}))


if __name__ == "__main__":
    main()
