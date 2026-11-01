"""Saved-only reconstruction of the unified constrained CEM search."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p1_unified_constrained_cem_20261101_codex as source

ROOT = source.ROOT
NAME = "p1_unified_constrained_cem_consume_20261101_codex"
PREP = ROOT / "results/p1_unified_constrained_cem_20261101_codex_PREP_MAIN01.json"
PREP_SHA = "dc95edfa2f9679d407c887716783801ee7b08ea26c0b43a49ef3756d5996cf58"
RUN = ROOT / "results/p1_unified_constrained_cem_20261101_codex_RUN_MAIN01"
REPORT_SHA = "c575d9ba0fa85dcf347665fee749d2ef435446908cb7d92fc712cdd446be95d0"


def write_new(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def load_events(path):
    rows = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "score":
            rows.append(event)
    return rows


def consume():
    host = source.blind.host
    report_path, events_path = RUN / "report.json", RUN / "events.jsonl"
    host.require(host.sha(PREP) == PREP_SHA and host.sha(report_path) == REPORT_SHA, "pinned artifacts")
    host.require(json.loads(PREP.read_bytes()) == source.protocol(), "protocol/source drift")
    report = json.loads(report_path.read_bytes())
    host.require(report["status"] == "model_only_complete" and not report["physical_execution"]
                 and not report["physical_future_used"] and not report["training"]
                 and report["attempts"] == report["returns"] == report["module_hooks"]
                 == {"predict": source.MAX_CALLS, "action_encoder": source.MAX_CALLS}
                 and report["model_before"] == report["model_after"], "report integrity")
    host.require(host.sha(events_path) == report["events_sha256"], "event hash")
    score_events = load_events(events_path)
    host.require(len(score_events) == 2 * source.ITERATIONS * source.SAMPLES, "score event count")
    output_rows = []
    for slot in (0, 1):
        task, initial, _, _ = source.blind.load(slot)
        root = report["rows"][slot]["root_mse"]
        rng = np.random.default_rng(source.SEED + slot)
        mean = np.zeros((source.HORIZON, 5, 2), np.float32)
        std = np.full_like(mean, source.INITIAL_STD)
        all_rows = []
        for iteration in range(source.ITERATIONS):
            bank = source.make_bank(rng, mean, std, initial, iteration)
            rows = sorted(
                (event for event in score_events
                 if event["slot"] == slot and event["iteration"] == iteration),
                key=lambda event: event["candidate"],
            )
            rows = [{key: value for key, value in row.items() if key not in ("event", "slot")}
                    for row in rows]
            host.require(len(rows) == source.SAMPLES
                         and [row["candidate"] for row in rows] == list(range(source.SAMPLES)),
                         "iteration roster")
            for candidate, row in enumerate(rows):
                actions = np.asarray(row["actions"], np.float32)
                host.require(np.array_equal(actions, bank[candidate])
                             and row["violation"] == max(0.0, root - row["q1"])
                             and np.isfinite([row["q1"], row["q3"], row["violation"]]).all(),
                             "saved bank/score mismatch")
            new_mean, new_std, elites = source.update(bank, rows)
            saved_iteration = report["rows"][slot]["iterations"][iteration]
            host.require(saved_iteration["iteration"] == iteration
                         and saved_iteration["elite_indices"] == elites
                         and np.array_equal(np.asarray(saved_iteration["mean"], np.float32), new_mean)
                         and np.array_equal(np.asarray(saved_iteration["std"], np.float32), new_std),
                         "CEM update mismatch")
            mean, std = new_mean, new_std
            all_rows.extend(rows)
        selected = source.choose(root, all_rows)
        greedy = min(all_rows, key=lambda row: (row["q1"], row["q3"], row["iteration"], row["candidate"]))
        saved = report["rows"][slot]["selected"]
        host.require(selected == saved and selected is not None
                     and report["rows"][slot]["eligible_count"]
                     == sum(row["q1"] >= root and row["q3"] < root for row in all_rows),
                     "selection mismatch")
        selected_actions = np.asarray(selected["actions"], np.float32)
        greedy_actions = np.asarray(greedy["actions"], np.float32)
        zero = selected_actions.copy()
        zero[0] = 0
        output_rows.append({
            "slot": slot,
            "root_mse": root,
            "selected": selected,
            "same_bank_greedy": greedy,
            "candidates": {
                "selected": selected_actions.tolist(),
                "zero_prefix": zero.tolist(),
                "same_bank_greedy": greedy_actions.tolist(),
            },
            "selected_is_distinct_from_greedy": not np.array_equal(selected_actions, greedy_actions),
            "task_sha256": host.digest(task),
        })
    return {
        "schema": NAME + ".v1",
        "source_prepare": str(PREP),
        "source_prepare_sha256": PREP_SHA,
        "source_report": str(report_path),
        "source_report_sha256": REPORT_SHA,
        "source_events_sha256": report["events_sha256"],
        "rows": output_rows,
        "new_neural_calls": 0,
        "new_physical_calls": 0,
        "development_only": True,
        "independent_confirmation": False,
        "P1_admission": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    source.blind.host.require(output.parent == ROOT / "results"
                              and output.name.startswith(NAME + "_")
                              and not output.exists() and not output.is_symlink(),
                              "exclusive output")
    print(json.dumps({"output": str(output), "sha256": write_new(output, consume())}), flush=True)


if __name__ == "__main__":
    main()
