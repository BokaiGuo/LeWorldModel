"""Fail-closed consumer for a completed shared-RGB P2 pair output.

This consumer authenticates the saved output and re-authenticates the
historical CP15 capture.  It does not promote the result to P1/P2 evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import p2_cp15_capture_builder_20261102_codexexclusive as builder
from . import p2_shared_rgb_pair_wiring_20261102_codexexclusive as wire
from . import p1_constructive_blind_encode_cuda_20261027_v1 as checkpoint

NAME = "p2_shared_pair_saved_20261102_codexexclusive"
MODEL_TENSOR_SHA = checkpoint.MODEL_TENSOR_SHA


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def verify_report(path, expected_sha, capture_sha):
    path = Path(path).resolve(strict=True)
    need(sha(path) == expected_sha, "report SHA")
    report = json.loads(path.read_bytes())
    need(report.get("schema") == wire.NAME + ".report.v1", "report schema")
    need(report.get("status") == "completed", "partial or failed output")
    need(report.get("bundle_sha256") == capture_sha, "report capture binding")
    need(report.get("physical_execution") is False
         and report.get("equal_total_compute") is False
         and report.get("independent_confirmation") is False
         and report.get("P1_admission") is False
         and report.get("P2_admission") is False, "claim boundary")
    need(report.get("source_authentication") == "capture_bundle_producer_not_authenticated",
         "unqualified source claim")
    need(report.get("model_tensor_sha256") == MODEL_TENSOR_SHA, "model receipt")
    for key in ("pair_sha256", "journal_sha256", "journal_tail"):
        need(key in report, "missing report closure: " + key)
    return report


def _read_journal(path, expected_sha, expected_tail):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    need(raw.endswith(b"\n") and hashlib.sha256(raw).hexdigest() == expected_sha,
         "journal SHA")
    rows = [json.loads(line) for line in raw.splitlines()]
    need(rows and rows[-1]["sha256"] == expected_tail, "journal tail")
    previous = None
    for index, row in enumerate(rows):
        body = {key: value for key, value in row.items() if key != "sha256"}
        need(row.get("index") == index and row.get("previous") == previous
             and hashlib.sha256(canonical(body)).hexdigest() == row["sha256"],
             "journal chain")
        previous = row["sha256"]
    return rows


def verify(capture_dir, capture_sha, run_dir, report_sha):
    capture_dir = Path(capture_dir).resolve(strict=True)
    run_dir = Path(run_dir).resolve(strict=True)
    capture_path = capture_dir / "capture.json"
    source = builder.verify(capture_dir, capture_sha)
    bundle, frames, goal, actions = wire.load_capture(capture_path, capture_sha)
    report = verify_report(run_dir / "report.json", report_sha, capture_sha)
    pair_path = run_dir / "pair.json"
    need(sha(pair_path) == report["pair_sha256"], "pair SHA")
    pair = json.loads(pair_path.read_bytes())
    need(pair["schema"] == wire.NAME + ".pair.v1"
         and pair["P1_admission"] is False and pair["P2_admission"] is False
         and pair["equal_total_compute"] is False
         and pair["source_authentication"] == "capture_bundle_producer_not_authenticated",
         "pair claim boundary")
    need(pair["context"]["latent_history"] and len(pair["branches"]) == 2,
         "pair structure")
    need(report["journal_sha256"] == sha(run_dir / "journal.jsonl"), "journal file")
    journal = _read_journal(run_dir / "journal.jsonl",
                            report["journal_sha256"], report["journal_tail"])
    events = [row["event"] for row in journal
              if row["kind"] == "pair_event" and "event" in row]
    need(journal[0]["kind"] == "run_started" and journal[-1]["kind"] == "run_finished"
         and journal[-1].get("status") == "completed", "run bookends")
    shared = [event for event in events if event["kind"] == "shared_context"]
    need(len(shared) == 1
         and shared[0]["context_sha256"] == wire.kernel.old.digest(pair["context"])
         and shared[0]["goal_sha256"] == wire.kernel.old.digest(pair["goal"]),
         "shared context receipt")
    attempts = [event for event in events if event["kind"] == "shared_encode_attempt"]
    returns = [event for event in events if event["kind"] == "shared_encode_return"]
    need([e["role"] for e in attempts] == ["control5", "control10", "control15", "goal"]
         and [e["role"] for e in returns] == [e["role"] for e in attempts]
         and [e["pixel_sha256"] for e in returns]
         == [row["pixel_sha256"] for row in bundle["frames"]] + [bundle["goal"]["pixel_sha256"]],
         "shared RGB receipt")
    branch_events = [event["record"] for event in events if event["kind"] == "branch"]
    for arm, branch in zip(wire.kernel.ARMS, pair["branches"], strict=True):
        scoped = [row for row in branch_events if row.get("arm") == arm]
        wire.saved.validate_branch(
            {"base": bundle["base"], "context": pair["context"], "goal": pair["goal"]},
            branch, [{"event": row} for row in scoped],
        )
    need(any(event["kind"] == "pair_complete" for event in events), "pair completion")
    return {
        "status": "verified_saved_pair_known_root_not_admitted",
        "capture_sha256": capture_sha,
        "report_sha256": report_sha,
        "journal_sha256": report["journal_sha256"],
        "source_scope": source["status"],
        "P1_admission": False,
        "P2_admission": False,
    }
