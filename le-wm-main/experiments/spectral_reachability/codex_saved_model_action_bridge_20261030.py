"""Pinned saved planner ABI -> paired journal selection, diagnostic only.

This never runs either planner, model or environment; neither input is Full D2P.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
JOURNAL = HERE.with_name("codex_paired_durable_journal_20261030.py")
JOURNAL_SHA = "79546cca63a80b6c63ec913cb79d64ea29f77af3fd2c9947938a87aacdfa86db"
SOURCES = {
    "sra_h2_saved": ("results/p2_no_composition_sra_20261027_v2_prepared01/report.json", "522afa5a68b23a6ff829c5c6d820d858953dccc0f778548e747ba594ff73937a"),
    "p4_suffix_calibration": ("results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1/runtime.events.jsonl", "150bcecb0f03c0ebe9f5b006738e35bac19443d3f7fa40fb1ece9e573066d594"),
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def old_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_journal_module():
    need(hashlib.sha256(JOURNAL.read_bytes()).hexdigest() == JOURNAL_SHA, "journal source drift")
    spec = importlib.util.spec_from_file_location("codex_pinned_journal_bridge", JOURNAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def packet(value):
    need(isinstance(value, list) and len(value) == 5 and all(
        isinstance(v, list) and len(v) == 2 and all(
            type(x) in (int, float) and math.isfinite(x) and -1 <= x <= 1 for x in v) for v in value),
        "invalid raw five-step packet")
    return copy.deepcopy(value)


def context(value):
    need(isinstance(value, dict) and value.get("layout") == "states_with_between_state_actions_v1"
         and value.get("context_length") == 3 and value.get("latent_dim") == 192 and value.get("action_dim") == 10,
         "context ABI mismatch")
    for key, n, d in (("latent_history", 3, 192), ("action_history", 2, 10)):
        a = value.get(key)
        need(isinstance(a, list) and len(a) == n and all(isinstance(row, list) and len(row) == d and
             all(type(x) in (int, float) and math.isfinite(x) for x in row) for row in a), "bad context")
    return value


def normalize_sra(row):
    r = row["result"]
    need(row["complete"] is True and r["arm"] == "joint_h2_no_composition_sra"
         and r["fixture_only"] is False and r["status"] == "search_complete"
         and r["candidate_and_replay_complete"] is True and r["replay_error"] is None, "incomplete SRA")
    attempts = r["attempts"]
    need(len(attempts) == 40 and len({a["candidate_id"] for a in attempts}) == 40
         and all(a["completed"] is True and a["metric_attempted"] is True
                 and type(a["cost"]) in (int, float) and math.isfinite(a["cost"]) for a in attempts), "SRA candidate census")
    winner = min(attempts, key=lambda x: x["cost"])
    need(winner["candidate_id"] == r["selected_candidate_id"] and winner["cost"] == r["selected_model_cost"], "SRA winner mismatch")
    witness = r["witness"]
    need(witness["independent_full_replay_exact"] is True and len(witness["edges"]) == 1, "SRA witness")
    edge = witness["edges"][0]
    root = row["root_sha256"]
    need(row["root"]["root_id"] == edge["root_id"] == root and edge["program_id"] == winner["candidate_id"]
         and context(edge["input_context"]) == context(row["root"]["context"])
         and edge["actions"] == winner["actions"] == witness["receipt"]["actions"], "SRA binding")
    domain = edge["action_domain"]
    need(domain["name"] == "raw" and domain["low"] == [-1., -1.] and domain["high"] == [1., 1.]
         and domain["action_block"] == 5 and len(edge["actions"]) == 10, "SRA domain")
    first, second = packet(edge["actions"][:5]), packet(edge["actions"][5:])
    return {"kind": "sra_h2_saved", "root_sha256": root, "step": 0,
            "input_context_sha256": digest(edge["input_context"]), "candidate_id": winner["candidate_id"],
            "commands": first, "remaining_saved_commands": second, "source_action_count": 10,
            "meaning": "first packet of saved open-loop H2 winner; not live recompilation"}


def normalize_p4(event):
    p = event["compiler_plan"]
    need(event["kind"] == "selection" and event["arm"] == "actual" and event["boundary"] in (5, 10, 15),
         "P4 actual selection required")
    need(p["arm"] == event["arm"] and p["root_sha256"] == event["root_sha256"] and
         p["role"] == "selected" and p["independent_replay_exact"] is True, "P4 binding")
    need(old_digest(context(p["context"])) == event["context_sha256"]
         and old_digest([p["goal"], p["q"]]) == event["metric_sha256"], "P4 context or metric")
    candidates = p["candidates"]
    need(len(candidates) == 10 and len({x["candidate_id"] for x in candidates}) == 10
         and all(type(x["nonlinear_terminal_cost"]) in (float, int)
                 and math.isfinite(x["nonlinear_terminal_cost"]) for x in candidates), "P4 candidate census")
    winner = min(candidates, key=lambda x: x["nonlinear_terminal_cost"])
    need(winner["candidate_id"] == p["selected_candidate_id"] and len(winner["commands"]) == 10
         and winner["commands"][:5] == p["prefix"] and winner["commands"][5:] == event["commands"], "P4 winner mismatch")
    payload = {k: event[k] for k in ("root_sha256", "arm", "boundary", "context_sha256", "metric_sha256",
                                      "incoming_between_raw", "compiler_plan", "commands", "successor")}
    need(old_digest(payload) == event["selection_sha256"], "P4 selection digest")
    return {"kind": "p4_suffix_calibration", "root_sha256": event["root_sha256"], "step": event["boundary"],
            "input_context_sha256": digest(p["context"]), "candidate_id": winner["candidate_id"],
            "commands": packet(event["commands"]), "remaining_saved_commands": None, "source_action_count": 5,
            "meaning": "saved actual-feedback H1 suffix calibration; not Full D2P"}


def load_selection(kind, root, step):
    need(kind in SOURCES, "unknown source")
    path, expected = SOURCES[kind]
    raw = (ROOT / path).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == expected, "saved source drift")
    if kind == "sra_h2_saved":
        need(step == 0, "SRA saved selection is root-only")
        matches = [r for r in json.loads(raw)["rows"] if r["root_sha256"] == root]
        need(len(matches) == 1, "missing or duplicate SRA root")
        return normalize_sra(matches[0])
    events = [json.loads(line) for line in raw.splitlines()]
    matches = [r for r in events
               if r.get("kind") == "selection" and r.get("arm") == "actual"
               and r.get("root_sha256") == root and r.get("boundary") == step]
    need(len(matches) == 1, "missing or duplicate P4 selection")
    boundaries = [r for r in events if r.get("kind") == "boundary_begin" and r.get("arm") == "actual"
                  and r.get("boundary") == step]
    need(len(boundaries) == 1 and boundaries[0]["between_raw"] == matches[0]["incoming_between_raw"],
         "P4 observed boundary mismatch")
    value = normalize_p4(matches[0])
    value["observed_frames"] = boundaries[0]["frames_pixel_sha256"]
    value["observed_actions"] = boundaries[0]["between_raw"]
    return value


def bind_saved(journal, arm, request, saved, observed_context, receipt_path):
    """Attach saved selection to journal; use only for non-dispatching diagnostics."""
    need(saved == load_selection(saved["kind"], saved["root_sha256"], saved["step"]), "saved selection mismatch")
    need(request["arm"] == arm and request["identity"]["root_sha256"] == saved["root_sha256"]
         and request["step"] == saved["step"] and digest(context(observed_context)) == saved["input_context_sha256"],
         "root, clock or planner context mismatch")
    if saved["kind"] == "p4_suffix_calibration":
        need(request["frames"] == saved["observed_frames"] and request["actions"] == saved["observed_actions"],
             "P4 observed history mismatch")
    need(journal.config["packet_steps"] == 5, "incompatible physical packet")
    need((saved["kind"], arm) in (("sra_h2_saved", "sra_no_composition"),
                                 ("p4_suffix_calibration", "full_d2p")), "slot mismatch")
    receipt = {"schema": "codex_saved_model_action_bridge_20261030_v1",
               "scope": "saved-only diagnostic; p4 slot is NOT Full D2P, no deployment authorization",
               "source": copy.deepcopy(saved), "request": copy.deepcopy(request),
               "bridge_source_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
               "journal_source_sha256": JOURNAL_SHA}
    raw = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    receipt_sha = hashlib.sha256(raw).hexdigest()
    with Path(receipt_path).open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    load_journal_module().sync_dir(Path(receipt_path).parent)
    token = journal.call("select", arm, request, saved["commands"], receipt_sha)
    return {"selection_sha256": token, "receipt_sha256": receipt_sha}


def consume_bound(journal_path, journal_sha, receipts):
    j = load_journal_module()
    out = j.consume(journal_path, journal_sha)
    raw = Path(journal_path).read_bytes()
    selects = [r["payload"] for r in (json.loads(line) for line in raw.splitlines())
               if r["kind"] == "begin" and r["payload"]["operation"] == "select"]
    need(len(selects) == len(receipts), "receipt census mismatch")
    seen = set()
    for select, (path, expected) in zip(selects, receipts):
        data = Path(path).read_bytes()
        need(hashlib.sha256(data).hexdigest() == expected and expected not in seen, "receipt hash or duplicate")
        seen.add(expected)
        r = json.loads(data, object_pairs_hook=j.no_duplicates)
        need(set(r) == {"schema", "scope", "source", "request", "bridge_source_sha256", "journal_source_sha256"}
             and r["schema"] == "codex_saved_model_action_bridge_20261030_v1"
             and r["scope"] == "saved-only diagnostic; p4 slot is NOT Full D2P, no deployment authorization"
             and r["bridge_source_sha256"] == hashlib.sha256(HERE.read_bytes()).hexdigest()
             and r["journal_source_sha256"] == JOURNAL_SHA, "bridge receipt source")
        s = r["source"]
        need(s == load_selection(s["kind"], s["root_sha256"], s["step"]), "saved selection changed")
        args = select["args"]
        need(len(args) == 4 and not select["kwargs"] and args[0] == r["request"]["arm"]
             and args[1] == r["request"] and args[2] == s["commands"] and args[3] == expected
             and (s["kind"], args[0]) in (("sra_h2_saved", "sra_no_composition"),
                                         ("p4_suffix_calibration", "full_d2p"))
             and args[1]["identity"]["root_sha256"] == s["root_sha256"] and args[1]["step"] == s["step"],
             "journal receipt binding")
        if s["kind"] == "p4_suffix_calibration":
            need(args[1]["frames"] == s["observed_frames"] and args[1]["actions"] == s["observed_actions"],
                 "P4 journal history mismatch")
    return {"schema": "codex_saved_model_action_bridge_consume_20261030_v1",
            "journal_consume": out, "receipts_verified": len(receipts),
            "full_d2p_verified": False, "online_sra_verified": False,
            "equal_total_compute": False, "production_authorized": False}
