"""Independent structural consumer for a future sealed prefix-feedback archive.

The expected manifest digest is an external trust anchor, not a signature.
No claim of genuine native capture, CUDA execution, or winner deployment is
made. This module does not execute a model or environment.
"""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from . import p2_prefix_feedback_common_budget_20261102_codexexclusive as kernel
from . import p2_cp5_launcher_preflight_20261102_codexexclusive as ledger

SCHEMA = "p2_prefix_feedback_archive_20261102_codexexclusive.v1"
SOURCE_ROLES = ("preparation", "task_bank", "model", "normalization", "code",
                "environment", "device_lock")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def load(root, ref, *, max_bytes=64 * 1024 * 1024):
    need(type(ref) is dict and set(ref) == {"path", "sha256"} and
         isinstance(ref["path"], str) and re.fullmatch(r"[A-Za-z0-9_.-]+", ref["path"])
         and ref["path"] not in (".", "..") and isinstance(ref["sha256"], str)
         and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]), "artifact reference")
    path = root / ref["path"]
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= max_bytes, "artifact missing")
    data = path.read_bytes()
    need(sha(data) == ref["sha256"], "artifact hash mismatch")
    return data


def action(hex_value):
    need(isinstance(hex_value, str) and len(hex_value) == 160 and
         re.fullmatch(r"[0-9a-f]{160}", hex_value), "action bytes")
    x = np.frombuffer(bytes.fromhex(hex_value), dtype="<f4").reshape(10, 2)
    need(np.isfinite(x).all() and (np.abs(x) <= 1).all(), "bounded actions")
    return x


def score_row(row, events, candidate, stage, arm, expected_initial):
    need(type(row) is dict and row["arm"] == arm, "optimizer row arm")
    chosen = action(row["raw_float32_hex"])
    relevant = [e for e in events if e.get("stage") == stage and
                e.get("candidate") == candidate and e.get("arm") == arm]
    need(len(relevant) == 18, "optimizer event count")
    observations = []
    for index in range(9):
        a, b = relevant[2 * index:2 * index + 2]
        need(a["kind"] == "attempt" and b["kind"] == "returned" and
             a["iteration"] == b["iteration"] == index and
             a["gradient_required"] is (index < 8) and b["gradient_required"] is (index < 8) and
             a["action_sha256"] == b["action_sha256"] and
             re.fullmatch(r"[0-9a-f]{64}", a["action_sha256"]) and
             type(b["score"]) in (float, int) and math.isfinite(b["score"]), "optimizer trace")
        observations.append((b["score"], index, b["action_sha256"]))
    need(observations[0][2] == sha(expected_initial), "optimizer initializer")
    best = min(observations, key=lambda v: (v[0], v[1]))
    need((row["model_score"], row["winner_iteration"], sha(chosen.tobytes())) == best and
         row["attempted_score_calls"] == row["returned_score_calls"] == 9 and
         row["attempted_gradients"] == row["returned_gradients"] == 8, "optimizer incumbent")
    return chosen


def audit_model_events(events, result):
    need(type(events) is list and events and events[-1]["kind"] == "root_complete" and
         sum(e.get("kind") == "root_complete" for e in events) == 1 and
         not any(e.get("kind") in ("root_aborted", "failed") for e in events), "model completion")
    active, counts, finished = {}, dict.fromkeys(kernel.LIMITS, 0), dict.fromkeys(kernel.LIMITS, 0)
    for e in events:
        kind = e.get("kind")
        if "stage" in e:
            need(e["stage"] in ("start", "suffix") and e.get("arm") in kernel.ARMS and
                 type(e.get("candidate")) is int and 0 <= e["candidate"] < kernel.N and
                 kind in ("attempt", "returned"), "optimizer event scope")
        if kind == "resource_attempt":
            ident, resource = e["id"], e["resource"]
            need(type(ident) is int and ident > 0 and ident not in active and
                 ident == sum(counts.values()) + 1 and resource in counts, "model meter attempt")
            active[ident] = (resource, {key: e[key] for key in ("arm", "candidate") if key in e})
            counts[resource] += 1
        elif kind == "resource_finish":
            ident = e["id"]
            need(ident in active and e["status"] == "returned" and
                 type(e["elapsed_ns"]) is int and e["elapsed_ns"] >= 0 and e["inclusive"] is True,
                 "model meter finish")
            resource, scope = active.pop(ident)
            need(e["resource"] == resource and
                 {key: e[key] for key in ("arm", "candidate") if key in e} == scope,
                 "model meter scope")
            finished[resource] += 1
    need(not active and counts == finished == kernel.LIMITS and
         result["costs"]["attempted"] == result["costs"]["returned"] == counts and
         events[-1]["costs"] == result["costs"] and events[-1]["candidate_denominator"] == kernel.N,
         "model resource totals")


def audit_budget(path, spec):
    data = path.read_bytes().splitlines()
    info = ledger.check_chain(path)
    need(info["tail_sha256"] == spec["tail_sha256"], "budget tail")
    events = [decode(line) for line in data]
    need(events and events[0]["kind"] == "contract", "budget contract")
    caps, counts, active = events[0]["caps"], {}, {}
    per_session = {}
    need(type(caps) is dict and caps and set(spec["expected_counts"]) == set(caps), "budget caps")
    counts = dict.fromkeys(caps, 0)
    for e in events[1:]:
        kind, resource, session = e["kind"], e["resource"], e["session"]
        need(resource in counts and isinstance(session, str) and session, "budget event")
        key = (resource, session)
        if kind == "attempt":
            counts[resource] += 1
            if resource == "public_step":
                per_session[session] = per_session.get(session, 0) + 1
            active[key] = active.get(key, 0) + 1
            need(type(caps[resource]) is int and counts[resource] <= caps[resource], "budget cap")
        elif kind == "returned":
            need(active.get(key, 0) > 0 and type(e["elapsed_ns"]) is int and e["elapsed_ns"] >= 0,
                 "budget return")
            active[key] -= 1
        else:
            raise ValueError("budget contains failure, rejection or unknown event")
    expected_sessions = {f"capture_{i:02d}_{arm}": 30 for i in range(16) for arm in kernel.ARMS}
    need(not any(active.values()) and counts == spec["expected_counts"] and
         per_session == expected_sessions, "budget completeness")
    return counts


def audit_native_journal(data, manifest, prefix, receipt):
    need(data.endswith(b"\n"), "native journal truncated")
    events = [decode(line) for line in data.splitlines()]
    need(not any(e.get("kind") in ("segment_native_stop", "segment_step_failed", "reconstruction_failed")
                 for e in events), "native failure or stop")
    relevant = [e for e in events if e.get("kind") in
                ("segment_step_intent", "segment_step_returned", "root_guard_accepted",
                 "shared_prefix_guard_accepted", "cp5_capture")]
    warm, shared = bytes.fromhex(manifest["warmup_hex"]), bytes.fromhex(manifest["shared_prefix_hex"])
    need(len(warm) == 80 and len(shared) == 120, "warmup/shared prefix actions")
    expected = []
    for phase, sequence in (("warmup", warm), ("plan", shared + prefix)):
        if phase == "plan":
            expected.append({"kind": "root_guard_accepted", "root_sha256": manifest["root_sha256"],
                             "task_id": manifest["task_id"]})
        for step in range(1, len(sequence) // 8 + 1):
            if phase == "plan" and step == 16:
                expected.append({"kind": "shared_prefix_guard_accepted",
                                 "root_sha256": manifest["root_sha256"], "task_id": manifest["task_id"],
                                 "prefix_float32_hex": shared.hex()})
            expected.extend(({"kind": "segment_step_intent", "phase": phase, "step": step,
                              "raw_float32_hex": sequence[(step-1)*8:step*8].hex()},
                             {"kind": "segment_step_returned", "phase": phase, "step": step,
                              "terminated": False, "truncated": False}))
    expected.append({"kind": "cp5_capture", "step": 20, "snapshot": receipt["snapshot"],
                     "rgb_sha256": receipt["rgb_sha256"][2]})
    need(len(relevant) == len(expected) and
         all(all(actual.get(key) == value for key, value in reference.items())
             for actual, reference in zip(relevant, expected, strict=True)), "native journal binding")


def consume(folder, manifest_sha256):
    root = Path(folder)
    need(root.is_dir() and not root.is_symlink() and isinstance(manifest_sha256, str)
         and re.fullmatch(r"[0-9a-f]{64}", manifest_sha256), "archive anchor")
    manifest = decode(load(root, {"path": "manifest.json", "sha256": manifest_sha256}))
    need(manifest["schema"] == SCHEMA and set(manifest["sources"]) == set(SOURCE_ROLES), "source closure")
    for role in SOURCE_ROLES:
        need(len(load(root, manifest["sources"][role])) > 0, "empty source")
    root_sha, task = manifest["root_sha256"], manifest["task_id"]
    need(isinstance(root_sha, str) and re.fullmatch(r"[0-9a-f]{64}", root_sha) and
         isinstance(task, str) and task, "root/task")
    base = np.asarray(manifest["base_history"], dtype="<f4")
    need(base.shape == (2, 10) and np.isfinite(base).all() and len(manifest["base_rgb_sha256"]) == 2,
         "base context")
    bank = load(root, manifest["bank"])
    need(len(bank) == 16 * 80, "bank length")
    result, events = decode(load(root, manifest["result"])), decode(load(root, manifest["events"]))
    need(result["schema"] == kernel.NAME + ".result_v1" and result["candidate_denominator"] == 16 and
         all(result[k] is False for k in ("physical_adapter_authenticated", "equal_total_compute",
                                           "independent_confirmation", "P2_admission")), "result schema")
    audit_model_events(events, result)
    spec = manifest["budget"]
    load(root, spec["file"])
    budget = audit_budget(root / spec["file"]["path"], spec)
    rows = result["rows"]
    need(len(rows) == 16 and [r["candidate_index"] for r in rows] == list(range(16)) and
         set(result["winners"]) == set(kernel.ARMS), "candidate census")
    need(set(manifest["captures"]) == {f"{i:02d}_{arm}" for i in range(16) for arm in kernel.ARMS},
         "capture census")
    for i, row in enumerate(rows):
        raw = bank[i * 80:(i + 1) * 80]
        action(raw.hex())
        need(row["bank_sha256"] == sha(raw) and
             [x["arm"] for x in row["start"]] == list(kernel.ARMS) and
             [x["arm"] for x in row["suffix"]] == list(kernel.ARMS) and
             set(row["feedback"]) == set(kernel.ARMS), "bank/arms")
        for arm, initial_row, suffix_row in zip(kernel.ARMS, row["start"], row["suffix"], strict=True):
            start = score_row(initial_row, events, i, "start", arm, raw)
            if arm == kernel.ARMS[0]:
                need(start[:5].tobytes() == raw[:40], "conditional fixed prefix")
            suffix = score_row(suffix_row, events, i, "suffix", arm, start.tobytes())
            need(suffix[:5].tobytes() == start[:5].tobytes(), "suffix prefix")
            refs = manifest["captures"][f"{i:02d}_{arm}"]
            need(set(refs) == {"receipt", "rgb", "context", "native_journal"}, "capture sources")
            receipt_bytes = load(root, refs["receipt"])
            receipt = decode(receipt_bytes)
            need(receipt["terminated"] is False and receipt["truncated"] is False,
                 "capture stop evidence")
            pixels = np.frombuffer(load(root, refs["rgb"]), dtype=np.uint8)
            need(pixels.size == 3 * 224 * 224 * 3, "capture RGB")
            pixels = pixels.reshape(3, 224, 224, 3)
            history_bytes = bytes.fromhex(receipt["action_history_hex"])
            need(len(history_bytes) == 80, "capture history")
            history = np.frombuffer(history_bytes, dtype="<f4").reshape(2, 10)
            value = {"arm": receipt["arm"], "candidate_index": receipt["candidate_index"],
                     "root_sha256": receipt["root_sha256"], "task_id": receipt["task_id"],
                     "prefix_hex": receipt["prefix_hex"], "rgb_history": pixels,
                     "action_history": history, "capture_receipt_sha256": sha(receipt_bytes),
                     "terminated": False, "truncated": False}
            obs = kernel.validate_capture(value, arm, i, start[:5], root_sha, task, base,
                                          manifest["base_rgb_sha256"])
            need(receipt["rgb_sha256"] == obs["rgb_sha256"], "receipt RGB")
            audit_native_journal(load(root, refs["native_journal"]), manifest, start[:5].tobytes(), receipt)
            context = decode(load(root, refs["context"]))
            kernel.validate_context(context, history)
            expected = {"prefix_sha256": sha(start[:5].tobytes()), "rgb_sha256": obs["rgb_sha256"],
                        "action_history_sha256": sha(history.tobytes()),
                        "capture_receipt_sha256": sha(receipt_bytes),
                        "context_sha256": sha(kernel.canonical(context))}
            need(row["feedback"][arm] == expected, "feedback/context binding")
            witness = [e for e in events if e.get("kind") == "feedback_bound" and
                       e.get("candidate") == i and e.get("arm") == arm]
            need(len(witness) == 1 and {key: witness[0][key] for key in expected} == expected,
                 "feedback event")
    for arm in kernel.ARMS:
        choices = [(e["model_score"], r["candidate_index"], e["winner_iteration"], e)
                   for r in rows for e in r["suffix"] if e["arm"] == arm]
        score, index, iteration, entry = min(choices, key=lambda x: x[:3])
        need(result["winners"][arm] == {"candidate_index": index, "iteration": iteration,
             "model_score": score, "raw_float32_hex": entry["raw_float32_hex"],
             "feedback": rows[index]["feedback"][arm]}, "winner action/selection binding")
    return {"schema": SCHEMA + ".consumer_receipt", "manifest_sha256": manifest_sha256,
            "winners": result["winners"], "budget_counts": budget, "physical_authorization": False,
            "source_semantics_authenticated": False, "physical_capture_authenticated": False,
            "gradient_updates_recomputed": False, "equal_total_compute": False}
