"""Exact async session/poll evidence in two sealed log prefixes; no execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

from experiments.spectral_reachability import agent_p4_original_evidence_extract_20261028_v1 as b

NAME = "agent_p4_original_evidence_extract_20261028_v2"
ROOT = b.ROOT
BASE_SHA = "5f2288bafdc03e4717dc3734a3e8970fc62649ec667276874a9480a0810aaea6"
TOKEN = re.compile(r'''\s+|//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|[A-Za-z_$][\w$]*|[0-9]+(?:\.[0-9]+)?|[^\s]''')


def tokens(source):
    return [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(source)
            if not m.group().isspace() and not m.group().startswith(("//", "/*"))]


def literal(token):
    if token.startswith('"'):
        return json.loads(token)
    if token.startswith("'") and "\\" not in token:
        return token[1:-1]
    if re.fullmatch(r"0|[1-9][0-9]*", token):
        return int(token)
    if token in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[token]
    raise ValueError("not a supported JS literal")


def object_literal(ts):
    b.need(ts[0][0] == "{" and ts[-1][0] == "}", "literal argument object")
    result = {}
    i = 1
    while i < len(ts) - 1:
        key = ts[i][0]
        if key.startswith(('"', "'")):
            key = literal(key)
        b.need(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) and key not in result, "unique literal key")
        b.need(i + 2 < len(ts) - 1 and ts[i + 1][0] == ":", "literal key:value")
        result[key] = literal(ts[i + 2][0])
        i += 3
        if i < len(ts) - 1:
            b.need(ts[i][0] == ",", "literal argument separator")
            i += 1
    return result


def poll_inputs(payload):
    """Parse only unconditional top-level literal calls, never eval JS.

    Other calls/strings/comments/loops/variable session IDs do not become poll
    evidence.  Multiple literal polls remain a batch, not a positional mapping.
    """
    if payload.get("type") == "function_call" and payload.get("name") in {"write_stdin", "functions.write_stdin"}:
        args = b.parse(payload["arguments"].encode())
        b.need(type(args.get("session_id")) is int and args["session_id"] > 0, "literal session ID")
        return {"polls": [{"session_id": args["session_id"], "pointer": "/payload/arguments", "input_span": None}], "pure_single": True, "unsupported": 0}
    if payload.get("type") != "custom_tool_call" or payload.get("name") != "exec":
        return {"polls": [], "pure_single": False, "unsupported": 0}
    source = payload.get("input", "")
    ts = tokens(source)
    statements, start, depth = [], 0, 0
    for i, (token, _, _) in enumerate(ts):
        if token in {"(", "{", "["}:
            depth += 1
        elif token in {")", "}", "]"}:
            depth -= 1
        if token == ";" and depth == 0:
            if i > start:
                statements.append(ts[start:i])
            start = i + 1
    if start < len(ts):
        statements.append(ts[start:])
    polls, unsupported = [], 0
    for statement in statements:
        values = [item[0] for item in statement]
        if "write_stdin" not in values:
            continue
        try:
            b.need(values[:7] == ["text", "(", "await", "tools", ".", "write_stdin", "("] and values[-2:] == [")", ")"], "direct top-level poll only")
            args = object_literal(statement[7:-2])
            b.need(type(args.get("session_id")) is int and args["session_id"] > 0, "literal positive session ID")
            polls.append({"session_id": args["session_id"], "pointer": "/payload/input", "input_span": [statement[0][1], statement[-1][2]]})
        except (ValueError, IndexError, TypeError):
            unsupported += 1
    return {"polls": polls, "pure_single": len(statements) == len(polls) == 1 and not unsupported, "unsupported": unsupported}


def engine_results(payload):
    """Only explicit tool-result JSON, never search numbers in stdout arrays."""
    if payload.get("type") == "function_call_output":
        blocks = [("/payload/output", payload.get("output"))]
    elif payload.get("type") == "custom_tool_call_output" and isinstance(payload.get("output"), list):
        blocks = [(f"/payload/output/{i}/text", item.get("text")) for i, item in enumerate(payload["output"])
                  if isinstance(item, dict) and item.get("type") == "input_text"]
    else:
        blocks = []
    out = []
    for pointer, text in blocks:
        if not isinstance(text, str):
            continue
        try:
            value = b.parse(text.encode())
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict) or not isinstance(value.get("chunk_id"), str) or not isinstance(value.get("output"), str) or type(value.get("wall_time_seconds")) not in (int, float):
            continue
        if "session_id" in value and not (type(value["session_id"]) is int and value["session_id"] > 0):
            continue
        if "exit_code" in value and type(value["exit_code"]) is not int:
            continue
        out.append({"pointer": pointer, "json_text_sha256": b.digest(text.encode()), "value": value})
    return out


def sanitized_evidence(result):
    value = result["value"]
    text = value["output"]
    trace = b.frames({"type": "custom_tool_call_output", "output": [{"type": "input_text", "text": text}]})
    for frame in trace:
        frame.pop("pointer")
        frame["decoded_pointer"] = "/output"
    ledgers = []
    offset = 0
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        try:
            ledger = b.parse(line.encode())
        except ValueError:
            ledger = None
        if isinstance(ledger, dict) and set(ledger) == {"call_id", "attempts"} and isinstance(ledger["call_id"], str) and isinstance(ledger["attempts"], list):
            valid = all(isinstance(row, dict) and set(row) == {"index", "reset_kind", "disposition", "identity_sha256"}
                        and type(row["index"]) is int and row["index"] >= 0
                        and row["reset_kind"] in {"random", "explicit_state", "goal_render", "replay"}
                        and row["disposition"] in {"selected", "unselected", "rejected", "failed", "replay"}
                        and (row["identity_sha256"] is None or isinstance(row["identity_sha256"], str) and re.fullmatch("[a-f0-9]{64}", row["identity_sha256"]))
                        for row in ledger["attempts"])
            if valid:
                ledgers.append({"decoded_pointer": "/output", "text_line": line_number,
                                "character_span": [offset, offset + len(line.rstrip("\r\n"))],
                                "record": ledger, "exhaustive": False})
        offset += len(line)
    return {"traceback_frames": trace, "reset_ledger_candidates": ledgers}


def origin(binding, line, offset, raw, call):
    return {"path": binding["path"], "prefix_bytes": binding["bytes"], "prefix_sha256": binding["sha256"],
            "jsonl_line": line, "record_byte_span": [offset, offset + len(raw)], "record_sha256": b.digest(raw), "call_id": call}


class Tracker:
    def __init__(self, dispatches):
        self.parents = {row["call_id"]: {"dispatch": row, "origin": None, "return_origin": None} for row in dispatches if row["source_id"] in {"A", "E", "H"}}
        self.sessions = []
        self.batches = []
        self.by_call = {}

    def receive(self, record, where):
        if record.get("type") != "response_item":
            return
        payload = record.get("payload", {})
        if not isinstance(payload, dict):
            return
        call, kind = payload.get("call_id"), payload.get("type")
        if call in self.parents:
            parent = self.parents[call]
            dispatch = parent["dispatch"]
            if kind == "custom_tool_call":
                b.need(parent["origin"] is None and where["path"] == dispatch["log"] and where["jsonl_line"] == dispatch["line"] and dispatch["command"] in payload.get("input", ""), "parent dispatch join")
                parent["origin"] = where
            elif kind == "custom_tool_call_output":
                b.need(parent["origin"] is not None and parent["return_origin"] is None
                       and where["path"] == parent["origin"]["path"]
                       and where["jsonl_line"] > parent["origin"]["jsonl_line"], "parent return join")
                parent["return_origin"] = where
                for result in engine_results(payload):
                    sid = result["value"].get("session_id")
                    if sid is not None:
                        self.sessions.append({"session_id": sid, "parent_call_id": call, "parent_source_label": dispatch["source_id"],
                                              "birth_origin": {**where, "pointer": result["pointer"], "decoded_pointer": "/session_id", "json_text_sha256": result["json_text_sha256"]},
                                              "subprocess_source_assignment": None, "poll_calls": [], "terminal_observations": [], "reset_attempts": None})
            return
        if kind in {"custom_tool_call", "function_call"}:
            parsed = poll_inputs(payload)
            matched = [s for s in self.sessions if any(p["session_id"] == s["session_id"] for p in parsed["polls"])]
            if not matched:
                return
            b.need(call not in self.by_call, "unique poll call ID")
            batch = {"call_id": call, "input_origin": where, "poll_inputs": parsed["polls"], "pure_single": parsed["pure_single"],
                     "unsupported_poll_statements": parsed["unsupported"], "return_origin": None, "result_observations": []}
            self.batches.append(batch)
            self.by_call[call] = batch
            for session in matched:
                if session["birth_origin"]["path"] == where["path"]:
                    session["poll_calls"].append(call)
            return
        if kind not in {"custom_tool_call_output", "function_call_output"} or call not in self.by_call:
            return
        batch = self.by_call[call]
        b.need(batch["return_origin"] is None and where["path"] == batch["input_origin"]["path"] and where["jsonl_line"] > batch["input_origin"]["jsonl_line"], "poll return call-ID/order join")
        batch["return_origin"] = where
        results = engine_results(payload)
        for result in results:
            value = result["value"]
            sid = value.get("session_id")
            targets = [s for s in self.sessions if call in s["poll_calls"] and s["session_id"] == sid] if sid is not None else []
            join = "explicit_return_session_id"
            if sid is None and batch["pure_single"] and len(results) == 1:
                targets = [s for s in self.sessions if call in s["poll_calls"]]
                join = "single_literal_poll_and_same_call_id"
            if len(targets) != 1:
                targets = []
                join = "unassigned_batch_return_no_unique_session_join"
            observation = {"origin": {**where, "pointer": result["pointer"], "json_text_sha256": result["json_text_sha256"]},
                           "session_id_field_present": "session_id" in value, "observed_session_id": sid,
                           "exit_code": value.get("exit_code"), "exit_code_field_present": "exit_code" in value,
                           "exit_decoded_pointer": "/exit_code" if "exit_code" in value else None,
                           "join": join, "attributed_session_id": targets[0]["session_id"] if targets else None,
                           **sanitized_evidence(result)}
            batch["result_observations"].append(observation)
            if targets and "exit_code" in value:
                targets[0]["terminal_observations"].append(observation)


def scan(stream, binding, tracker):
    import hashlib
    remaining, line, offset = binding["bytes"], 0, 0
    b.need(type(remaining) is int and 0 < remaining <= b.LOG_CAP, "bounded prefix")
    h = hashlib.sha256()
    while remaining:
        raw = stream.readline(min(remaining, b.LINE_CAP + 1))
        b.need(raw and len(raw) <= b.LINE_CAP, "truncated prefix or oversized line")
        h.update(raw)
        remaining -= len(raw)
        line += 1
        # Parse only explicit parent/pending call IDs or actual poll input candidates.
        known = [key.encode() for key in (*tracker.parents, *tracker.by_call)]
        if b"write_stdin" in raw or any(key in raw for key in known):
            record = b.parse(raw)
            payload = record.get("payload", {})
            call = payload.get("call_id") if isinstance(payload, dict) else None
            tracker.receive(record, origin(binding, line, offset, raw, call))
        offset += len(raw)
    b.need(h.hexdigest() == binding["sha256"], "prefix SHA mismatch")


def closure():
    paths = [Path(__file__).resolve(), ROOT / "tests" / ("test_" + NAME + ".py"), ROOT / "innerdocs" / (NAME + ".md")]
    return {str(path): b.digest(b.small_read(path)) for path in paths}


def extract():
    b.need(b.digest(b.small_read(Path(b.__file__).resolve())) == BASE_SHA, "frozen v1 dependency SHA")
    data = b.catalog()
    tracker = Tracker(data["recovery"]["dispatches"])
    b.need(len(tracker.parents) == 3 and sum(row["bytes"] for row in data["log_bindings"]) <= b.LOG_CAP, "fixed parent/log scope")
    for binding in data["log_bindings"]:
        path = b.regular(binding["path"])
        b.need(path.stat().st_size >= binding["bytes"], "bound prefix missing")
        with path.open("rb") as stream:
            scan(stream, binding, tracker)
    b.need(all(p["origin"] is not None and p["return_origin"] is not None for p in tracker.parents.values()), "known parent origins complete")
    for session in tracker.sessions:
        session["status"] = "terminal_session_joined_not_task_assigned" if session["terminal_observations"] else "polled_but_terminal_unattributed" if session["poll_calls"] else "no_literal_poll_join_in_prefix"
        session["unknown"] = ["specific subprocess/source assignment within parent", "full reset attempt ledger and identities", "original loaded bytes", "exhaustive/restart coverage"]
        if not session["terminal_observations"]:
            session["unknown"].append("no unique terminal session join; batch exit must not be positionally assigned")
        session["unclosed_poll_reasons"] = []
        for call in session["poll_calls"]:
            batch = tracker.by_call[call]
            if batch["return_origin"] is None:
                reason = "same_call_id_return_not_in_bound_prefix"
            elif not batch["result_observations"]:
                reason = "no_recognized_engine_result_in_call_return"
            elif not any(row["attributed_session_id"] == session["session_id"] and row["exit_code_field_present"] for row in batch["result_observations"]):
                reason = "no_unique_terminal_result_session_join"
            else:
                continue
            session["unclosed_poll_reasons"].append({"call_id": call, "reason": reason})
    observations = [r for batch in tracker.batches for r in batch["result_observations"]]
    return {"schema": NAME + ".report", "status": "partial_async_evidence_not_authority", "implementation_closure": closure(),
            "base_source": {"path": str(Path(b.__file__).resolve()), "sha256": BASE_SHA},
            "coverage": {"path": str(b.CATALOG), "sha256": b.CATALOG_SHA}, "bound_logs": data["log_bindings"],
            "parents": list(tracker.parents.values()), "sessions": tracker.sessions, "poll_batches": tracker.batches,
            "summary": {"parents": len(tracker.parents), "sessions": len(tracker.sessions), "poll_batches": len(tracker.batches),
                        "sessions_with_literal_poll": sum(bool(s["poll_calls"]) for s in tracker.sessions),
                        "sessions_with_joined_terminal": sum(bool(s["terminal_observations"]) for s in tracker.sessions),
                        "batch_terminal_observations": sum(r["exit_code_field_present"] for r in observations),
                        "unassigned_terminal_observations": sum(r["exit_code_field_present"] and r["attributed_session_id"] is None for r in observations),
                        "traceback_frames": sum(len(r["traceback_frames"]) for r in observations),
                        "reset_ledger_candidates": sum(len(r["reset_ledger_candidates"]) for r in observations)},
            "limitations": ["same-log session namespace only; cross-log IDs are not silently equivalent", "no return-order/nearby-text mapping",
                            "only literal unconditional supported poll syntax is joined; dynamic syntax remains unproved", "stdout score/outcome arrays are not copied",
                            "traceback paths and ledger candidates are partial observations, not loaded-byte or exhaustive certificates"],
            "source_run_authorized": False, "certificate_created": False, "approval_created": False, "reset_blockers_cleared": 0,
            "reserved_roots_opened": False, "new_neural_calls": 0, "new_physics_calls": 0}


def write(path, value):
    path = Path(path)
    b.need(path.is_absolute() and path.parent.resolve() == path.parent and path.name.startswith(NAME + "_") and path.suffix == ".json", "own output only")
    raw = b.canonical(value)
    b.need(len(raw) <= b.JSON_CAP, "report cap")
    with path.open("xb") as stream:
        b.need(stream.write(raw) == len(raw), "short report write")


def consume(path, expected):
    path = Path(path)
    b.need(path.name.startswith(NAME + "_") and path.suffix == ".json", "own saved report only")
    raw = b.small_read(path)
    b.need(b.digest(raw) == expected and b.canonical(b.parse(raw)) == b.canonical(extract()), "saved SHA/reconstruction")
    return {"status": "partial_async_evidence_verified_not_authority", "report_sha256": expected, "source_run_authorized": False, "reset_blockers_cleared": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    option = subs.add_parser("extract")
    option.add_argument("--output", type=Path)
    option = subs.add_parser("consume")
    option.add_argument("--report", type=Path, required=True)
    option.add_argument("--sha", required=True)
    args = parser.parse_args()
    if args.command == "consume":
        value = consume(args.report, args.sha)
    else:
        value = extract()
        if args.output:
            write(args.output, value)
            value = {"output": str(args.output), "sha256": b.digest(b.canonical(value)), "summary": value["summary"], "source_run_authorized": False}
    print(b.canonical(value).decode())


if __name__ == "__main__":
    main()
