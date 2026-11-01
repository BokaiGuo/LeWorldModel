"""Verify durable CUDA journal, preserving all branches; never access future labels."""
import argparse
import math
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_rgb_cuda_runner_20261030_codex_v1 as runner

abi = runner.abi
NAME = "p5_rgb_cuda_consumer_20261030_codex_v1"


def consume(prepared, data):
    check = abi.check
    check(data.endswith(b"\n") and len(data) <= 16 * 1024 * 1024, "complete bounded journal lines")
    lines = data.splitlines()
    check(lines, "empty journal")
    events, prev = [], "0" * 64
    for seq, line in enumerate(lines):
        event = abi.bridge.read_json(line)
        check(isinstance(event, dict) and event.get("seq") == seq and event.get("prev") == prev, "journal chain")
        body = dict(event)
        claimed = body.pop("hash", None)
        check(claimed == abi.bridge.digest(abi.bridge.canonical(body)), "journal hash")
        prev = claimed
        events.append(event)
    start = events[0]
    check(start["kind"] == "start" and start["prepared_sha256"] == runner.PREP_SHA and
          start["runner_source_sha256"] == abi.bridge.digest(Path(runner.__file__).read_bytes()) and
          type(start["max_seconds"]) is int and 1 <= start["max_seconds"] <= 3600, "start")
    backend = start["backend"]
    check(isinstance(backend["device"], str) and backend["device"].startswith("cuda:") and
          backend["strict_load"] is True and backend["frozen"] is True and backend["eval"] is True and
          backend["checkpoint_sha256"] == abi.bridge.FILES["checkpoint"][1], "backend attestation")
    rows = [{"index": r["index"], "root_slot": r["root_slot"], "condition": r["condition"],
             "encode": {"status": "unattempted"},
             "steps": [{"step": s, "status": "unattempted"} for s in range(1, 6)]}
            for r in prepared["rows"]]
    branch, step, pending, ended, failed = 0, 0, False, None, False
    for event in events[1:]:
        kind = event["kind"]
        check(ended is None, "event after end")
        if kind == "end":
            check(event["status"] in ("complete", "partial") and
                  isinstance(event["elapsed_seconds"], (int, float)) and
                  math.isfinite(event["elapsed_seconds"]) and event["elapsed_seconds"] >= 0, "end")
            check(not pending, "ended with pending call")
            check((event["status"] == "complete") == (branch == 164 and not failed), "end completeness")
            ended = event["status"]
            continue
        check(branch < 164 and not failed, "unexpected further operation")
        source, target = prepared["rows"][branch], rows[branch]
        expected = {"branch": branch, "condition": source["condition"],
                    "previous_sha256": source["previous_rgb"]["model_input_chw_float32_sha256"],
                    "image_sha256": source["current_rgb"]["model_input_chw_float32_sha256"],
                    "actions_sha256": source["actions_sha256"],
                    "op": "encode_state" if step == 0 else "predict_next", "step": step}
        check(all(event.get(k) == v for k, v in expected.items()), "operation identity/order")
        slot = target["encode"] if step == 0 else target["steps"][step - 1]
        if kind == "attempt":
            check(not pending, "duplicate attempt")
            slot["status"] = "attempted_unreturned"
            pending = True
        elif kind in ("returned", "failed"):
            check(pending, "return without attempt")
            pending = False
            if kind == "failed":
                check(isinstance(event.get("error_type"), str) and isinstance(event.get("error"), str), "failure")
                slot.update(status="failed", error_type=event["error_type"], error=event["error"])
                failed = True
            else:
                out = event["output"]
                check(isinstance(out, dict) and set(out) == {"values", "sha256"}, "output fields")
                values = np.asarray(out["values"], dtype="<f4")
                check(values.shape == (6,) and np.isfinite(values).all() and
                      abi.bridge.digest(values.tobytes()) == out["sha256"], "output hash")
                slot.update(status="returned", output=out)
                step += 1
                if step == 6:
                    branch, step = branch + 1, 0
        else:
            raise ValueError("unknown event")
    counts = {}
    for condition in ("off", "on"):
        selected = [row for row in rows if row["condition"] == condition]
        counts[condition] = {"denominator": len(selected),
                             "complete": sum(all(s["status"] == "returned" for s in row["steps"])
                                             and row["encode"]["status"] == "returned" for row in selected)}
    check(counts["off"]["denominator"] == counts["on"]["denominator"] == 82, "conditions")
    return {"schema": NAME, "status": "complete_journal_verified" if ended == "complete" else "partial_journal_verified",
            "journal_sha256": abi.bridge.digest(data), "last_event_hash": prev,
            "prepared_sha256": runner.PREP_SHA, "branch_denominator": 164, "pair_denominator": 80,
            "condition_counts": counts, "rows": rows, "backend_self_report": backend,
            "end_status": ended, "P5_admission": False,
            "boundary": "journal consistency only; no future labels, distribution equivalence, accuracy or independent execution proof"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--journal-sha256", required=True)
    ap.add_argument("--output-name", required=True)
    args = ap.parse_args()
    for name, prefix in ((args.run_name, runner.NAME + "_"), (args.output_name, NAME + "_")):
        abi.check(Path(name).name == name and name.startswith(prefix), "exclusive local name")
    directory = abi.bridge.ROOT / "results" / args.run_name
    path = directory / "events.jsonl"
    abi.check(directory.is_dir() and not directory.is_symlink() and path.is_file() and not path.is_symlink(),
              "journal path")
    data = path.read_bytes()
    abi.check(abi.bridge.digest(data) == args.journal_sha256, "external journal SHA")
    result = consume(runner.load_prepared(), data)
    result_bytes = abi.bridge.canonical(result)
    output = abi.bridge.ROOT / "results" / args.output_name
    with output.open("xb") as f:
        f.write(result_bytes)
        f.flush()
        os.fsync(f.fileno())
    runner.sync_dir(output.parent)
    print(f"{result['status']} {abi.bridge.digest(result_bytes)}")


if __name__ == "__main__":
    main()
