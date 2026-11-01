"""Pinned metadata projection for ten historical calls; no outcome or candidate extraction."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import zlib

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "results/codex_d2p_historical_exclusion_index_20261029_LOCAL01.json"
DELTA = ROOT / "results/agent_freshroots_resume_20261027_next_delta.json"
COVERAGE = ROOT / "results/agent_freshroots_resume_20261027_coverage.json"
PINS = {INDEX: "ac2544512d033ec14fd07808c3abe2dfd96676bb6b82ec89569f498f58256f98",
        DELTA: "5ef07ab7815fff0c392feef59309c0116596e95d541eb95413d1d966a8b2a29b",
        COVERAGE: "00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb"}
LOG = Path("/home/gbk/.codex/sessions/2026/10/23/rollout-2026-10-23T21-46-56-01a167e2-eca3-7f63-bb9b-93030b5d366c_01a16903-c194-71a3-972b-7a268203b373.jsonl")
LOG_SHA = "0a2c097548ef4bd9818affcf82ecb53102ddfa965498b966c13aa7067efdf70c"
LOG_BYTES = 508228945


def pinned(path):
    if path.is_symlink() or path.resolve() != path or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("invalid metadata path or size")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINS[path]:
        raise ValueError("metadata SHA mismatch")
    return raw


def field(raw, name, cap=1024 * 1024):
    """Decode one preselected JSON value, never parse surrounding values."""
    match = re.search(rb'"' + name.encode("ascii") + rb'"\s*:\s*', raw)
    if not match:
        raise ValueError("missing projection field " + name)
    value, _ = json.JSONDecoder().raw_decode(raw[match.end():match.end() + cap].decode("utf-8"))
    return value


def delta_receipts(raw):
    env = json.loads(raw)
    if set(env) != {"schema", "semantic_bytes", "semantic_sha256", "gzip_base64"}:
        raise ValueError("invalid delta envelope")
    if env["schema"] != "agent_freshroots_resume_20261027_next.lossless-envelope.v1":
        raise ValueError("unexpected delta schema")
    if type(env["semantic_bytes"]) is not int or not 0 < env["semantic_bytes"] <= 64 * 1024 * 1024:
        raise ValueError("invalid semantic length")
    dec = zlib.decompressobj(31)
    payload = dec.decompress(base64.b64decode(env["gzip_base64"], validate=True), 64 * 1024 * 1024 + 1)
    if (not dec.eof or dec.unused_data or dec.unconsumed_tail or
            len(payload) != env["semantic_bytes"] or hashlib.sha256(payload).hexdigest() != env["semantic_sha256"]):
        raise ValueError("delta semantic mismatch")
    if not payload.startswith(b'{"dispatch_evidence":{"dispatch_receipts":'):
        raise ValueError("unexpected canonical projection prefix")
    # Decode only the allowlisted receipt array, never metadata_delta or candidates.
    return field(payload[:1024 * 1024], "dispatch_receipts")


def merge(dispatch_log, receipts, sources):
    dispatch = dispatch_log["dispatches"]
    if len(dispatch) != 10 or len({d["call_id"] for d in dispatch}) != 10:
        raise ValueError("dispatch denominator or identity mismatch")
    if dispatch_log["path"] != str(LOG) or dispatch_log["sha256"] != LOG_SHA or dispatch_log["bytes"] != LOG_BYTES:
        raise ValueError("wrong original log binding")
    source_map = {x["id"]: (i, x) for i, x in enumerate(sources)}
    if set(source_map) != set("ABCDEFGH") or len(sources) != 8:
        raise ValueError("source denominator")
    if len(receipts) != 13:
        raise ValueError("receipt denominator")
    groups = {d["call_id"]: [] for d in dispatch}
    positions = {}
    for index, rec in enumerate(receipts):
        cid = rec["dispatch_call_id"]
        if cid not in groups:
            raise ValueError("orphan receipt")
        positions[id(rec)] = index
        groups[cid].append(rec)
    rows = []
    for d in dispatch:
        rs = groups[d["call_id"]]
        if not rs or any(r["source_id"] != d["source_id"] or r["dispatch_line"] != d["line"]
                         or r["log"] != str(LOG) for r in rs):
            raise ValueError("receipt dispatch join mismatch")
        starts = [r for r in rs if r["operation"] == "exec_command"]
        polls = [r for r in rs if r["operation"] == "write_stdin"]
        if len(starts) != 1 or len(polls) > 1 or len(rs) != len(starts) + len(polls):
            raise ValueError("unexpected receipt operations")
        if polls and (starts[0]["session_id"] is None or polls[0]["polled_session"] != starts[0]["session_id"]):
            raise ValueError("session join mismatch")
        statuses = [r["exit_code"] for r in rs if r["exit_code"] is not None]
        errors = [e.split(":", 1)[0] for r in rs for e in r["exceptions"]]
        status = ("exit_recorded" if statuses else "traceback_recorded" if errors else "stdout_without_exit")
        source_index, src = source_map[d["source_id"]]
        evidence = [{**{k: r[k] for k in ("call_id", "operation", "operation_index", "line",
                    "line_sha256", "session_id", "polled_session", "exit_code")},
                    "delta_semantic_pointer": "/dispatch_evidence/dispatch_receipts/" + str(positions[id(r)]),
                    "producer_traceback_frames": r["producer_traceback_frames"]} for r in rs]
        rows.append({
            "source_id": d["source_id"], "call_id": d["call_id"],
            "dispatch": {k: d[k] for k in ("line", "line_sha256", "timestamp")},
            "reported_family_source": {"path": src["path"], "sha256": src["sha256"],
                "coverage_pointer": "/sources/" + str(source_index),
                "binding": "family join only; not proof this report existed at this call"},
            "return_receipts": evidence, "process_status": status,
            "exit_code": statuses[-1] if statuses else None, "traceback_types": errors,
            "reset_attempt_ledger": "not_established", "loaded_closure_at_run": "not_established",
            "exhaustive_attempt_bound": "not_established", "restart_and_scope_complete": "not_established",
            "missing_evidence": ["original complete reset ledger OR run-bound loaded closure and exhaustive envelope",
                                 "failure/unselected/explicit-state/replay attempts and restarts",
                                 "independent origin and historical scope review"] +
                (["original shell launcher for this call"] if d["source_id"] == "B" else []) +
                (["original P0 dependency for this call"] if d["source_id"] in "DEFGH" else []),
        })
    if sum(r["source_id"] == "B" for r in rows) != 3:
        raise ValueError("B's three calls must be retained")
    return rows


def verify_original_lines(log, dispatch_log, rows):
    expected = {d["line"]: d["line_sha256"] for d in dispatch_log["dispatches"]}
    for row in rows:
        for r in row["return_receipts"]:
            if r["line"] in expected and expected[r["line"]] != r["line_sha256"]:
                raise ValueError("conflicting line hash")
            expected[r["line"]] = r["line_sha256"]
    h, size, seen = hashlib.sha256(), 0, set()
    with log.open("rb") as f:
        n = 0
        while size < LOG_BYTES:
            line = f.readline(min(8 * 1024 * 1024 + 1, LOG_BYTES - size))
            n += 1
            if not line.endswith(b"\n") or len(line) > 8 * 1024 * 1024 or size + len(line) > LOG_BYTES:
                raise ValueError("invalid bounded log line")
            size += len(line)
            h.update(line)
            if n in expected:
                if hashlib.sha256(line).hexdigest() != expected[n]:
                    raise ValueError("original line hash mismatch")
                seen.add(n)
    if size != LOG_BYTES or h.hexdigest() != LOG_SHA or seen != set(expected):
        raise ValueError("original log prefix or line denominator mismatch")
    return len(seen)


def build():
    index_raw, delta_raw, coverage_raw = pinned(INDEX), pinned(DELTA), pinned(COVERAGE)
    dispatch_log = field(index_raw, "original_dispatch_log", 32768)
    sources = field(coverage_raw, "sources", 32768)
    receipts = delta_receipts(delta_raw)
    rows = merge(dispatch_log, receipts, sources)
    verified = verify_original_lines(LOG, dispatch_log, rows)
    return {
        "schema": "codex_reset_ledger_recovery_map_20261030.v1",
        "input_sha256": {str(p): h for p, h in PINS.items()},
        "original_log": {"path": str(LOG), "prefix_bytes": LOG_BYTES, "prefix_sha256": LOG_SHA,
                         "distinct_lines_sha_verified": verified},
        "calls": rows, "historical_sources": 8, "inherited_references": 16,
        "known_calls": 10, "return_receipts": len(receipts),
        "source_binding_limits": "Receipt semantics inherited from pinned projection; original line bytes rehashed, not re-parsed.",
        "ledger_completeness_certified": False, "history_complete": False,
        "freshness_certified": False, "source_run_authorized": False, "p1_admitted": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    payload = (json.dumps(build(), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"output": str(args.output.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}))


if __name__ == "__main__":
    main()
