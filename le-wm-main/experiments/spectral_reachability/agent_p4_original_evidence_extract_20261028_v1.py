"""Extract partial original runtime path evidence, not historical authority.

Only coverage-pinned log byte prefixes and exact A-H call IDs are searched.
No path appearing in a traceback, command, or tool return is opened/executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

NAME = "agent_p4_original_evidence_extract_20261028_v1"
ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "results/agent_freshroots_resume_20261027_coverage.json"
CATALOG_SHA = "00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb"
LINE_CAP = 64 * 1024**2
JSON_CAP = 4 * 1024**2
LOG_CAP = 1024**3
FRAME = re.compile(r'^\s*(?:>\s*)?File "(/[^"\n]+)", line ([1-9][0-9]*), in (.+)$')


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def unique(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    out = json.loads(raw, object_pairs_hook=unique)
    canonical(out)
    return out


def regular(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(), "canonical regular explicit file")
    return path


def small_read(path):
    path = regular(path)
    need(path.stat().st_size <= JSON_CAP, "small JSON cap")
    raw = path.read_bytes()
    need(len(raw) <= JSON_CAP, "small JSON grew")
    return raw


def catalog():
    raw = small_read(CATALOG)
    need(digest(raw) == CATALOG_SHA, "coverage SHA")
    out = parse(raw)
    need([item["id"] for item in out["sources"]] == list("ABCDEFGH"), "all eight source IDs")
    need(len(out["recovery"]["dispatches"]) == 10 and len(out["log_bindings"]) == 2, "fixed dispatch/log scope")
    need(len({row["call_id"] for row in out["recovery"]["dispatches"]}) == 10, "unique known calls")
    return out


def frames(payload):
    """Extract only literal absolute frames in returned traceback text.

    Strings in source edits/commands are never passed here; non-traceback
    warnings, paths, model scores, root arrays and process-exit summaries are
    deliberately not promoted to provenance.  No unescaping or eval is used.
    """
    if payload.get("type") != "custom_tool_call_output":
        return []
    blocks = payload.get("output")
    if not isinstance(blocks, list):
        return []
    result = []
    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict) or block.get("type") != "input_text" or not isinstance(block.get("text"), str):
            continue
        text = block["text"]
        active = False
        offset = 0
        for text_line, raw_line in enumerate(text.splitlines(keepends=True), 1):
            line = raw_line.rstrip("\r\n")
            if line == "Traceback (most recent call last):":
                active = True
            match = FRAME.fullmatch(line) if active else None
            if match:
                result.append({"observed_path": match[1], "source_line": int(match[2]), "function": match[3],
                               "pointer": f"/payload/output/{block_index}/text", "text_line": text_line,
                               "character_span": [offset, offset + len(line)], "literal_frame": line,
                               "loaded_bytes_sha256": None, "evidence_kind": "returned_traceback_path_only"})
            offset += len(raw_line)
    return result


def scan_prefix(stream, binding, dispatches):
    """Bounded scan; never read even one byte of the unbound appended suffix."""
    size = binding["bytes"]
    need(type(size) is int and 0 < size <= LOG_CAP, "bound log byte cap")
    ids = {row["call_id"]: row for row in dispatches}
    needles = [call.encode() for call in ids]
    remaining, line_number, offset = size, 0, 0
    hasher = hashlib.sha256()
    selected = []
    while remaining:
        raw = stream.readline(min(remaining, LINE_CAP + 1))
        need(raw and len(raw) <= LINE_CAP, "truncated prefix or oversized log line")
        hasher.update(raw)
        remaining -= len(raw)
        line_number += 1
        if any(needle in raw for needle in needles):
            record = parse(raw)
            payload = record.get("payload", {})
            call = payload.get("call_id") if isinstance(payload, dict) else None
            if record.get("type") == "response_item" and call in ids:
                kind = payload.get("type")
                if kind in {"custom_tool_call", "custom_tool_call_output"}:
                    origin = {"path": binding["path"], "prefix_sha256": binding["sha256"], "prefix_bytes": size,
                              "jsonl_line": line_number, "record_byte_span": [offset, offset + len(raw)],
                              "record_sha256": digest(raw), "call_id": call}
                    if kind == "custom_tool_call":
                        need(binding["path"] == ids[call]["log"] and line_number == ids[call]["line"], "dispatch origin join")
                        origin["pointer"] = "/payload/input"
                        need(ids[call]["command"] in payload.get("input", ""), "declared dispatch command join")
                    else:
                        origin["pointer"] = "/payload/output"
                    selected.append({"kind": kind, "origin": origin, "frames": frames(payload)})
        offset += len(raw)
    need(hasher.hexdigest() == binding["sha256"], "bound prefix SHA mismatch")
    return selected


def own_closure():
    paths = [Path(__file__).resolve(), ROOT / "tests" / ("test_" + NAME + ".py"), ROOT / "innerdocs" / (NAME + ".md")]
    return {str(path): digest(small_read(path)) for path in paths}


def extract():
    data = catalog()
    dispatches = data["recovery"]["dispatches"]
    bindings = data["log_bindings"]
    need(sum(row["bytes"] for row in bindings) <= LOG_CAP, "total bound bytes cap")
    known_logs = {item["path"] for item in bindings}
    need(all(row["log"] in known_logs for row in dispatches), "dispatch must reference bound log")
    records = []
    for binding in bindings:
        path = regular(binding["path"])
        need(path.stat().st_size >= binding["bytes"], "bound prefix not available")
        with path.open("rb") as stream:
            records.extend(scan_prefix(stream, binding, dispatches))
    rows = []
    for dispatch in dispatches:
        linked = [item for item in records if item["origin"]["call_id"] == dispatch["call_id"]]
        calls = [item for item in linked if item["kind"] == "custom_tool_call"]
        returns = [item for item in linked if item["kind"] == "custom_tool_call_output"]
        need(len(calls) == 1 and len(returns) <= 1, "unambiguous original dispatch and direct return")
        need(not returns or returns[0]["origin"]["jsonl_line"] > calls[0]["origin"]["jsonl_line"], "return precedes dispatch")
        observations = [{**frame, "origin": item["origin"]} for item in returns for frame in item["frames"]]
        rows.append({"source_id": dispatch["source_id"], "call_id": dispatch["call_id"],
                     "dispatch_origin": calls[0]["origin"], "direct_return_origins": [item["origin"] for item in returns],
                     "runtime_path_observations": observations, "reset_attempts": None, "reset_ledger": None,
                     "exhaustive": False, "loaded_closure_complete": False, "loaded_bytes_provenance": None,
                     "unknown": ["reset attempts and identities", "original loaded byte hashes and full factory/environment closure",
                                 "attempt/restart completeness", "async child process/session mapping not reconstructed"],
                     "status": "partial_runtime_path_evidence" if observations else "no_traceback_path_in_direct_return"})
    return {"schema": NAME + ".report", "status": "partial_original_runtime_paths_extracted_not_authority",
            "coverage": {"path": str(CATALOG), "sha256": CATALOG_SHA}, "implementation_closure": own_closure(),
            "bound_logs": bindings, "rows": rows,
            "summary": {"known_dispatches": len(rows), "dispatches_with_runtime_paths": sum(bool(row["runtime_path_observations"]) for row in rows),
                        "runtime_path_frames": sum(len(row["runtime_path_observations"]) for row in rows)},
            "scope": "only exact known call IDs in original coverage-bound prefixes; no appended suffix",
            "limitations": ["traceback path is partial runtime evidence, not a loaded-byte seal", "no traceback does not imply no resets or execution",
                            "direct tool return may contain several subprocesses; no async session attribution is inferred",
                            "historical first-capture SHA is not execution-time sealing; origin review remains MAIN's responsibility"],
            "origin_review_required": True, "reset_blockers_cleared": 0,
            "certificate_created": False, "approval_created": False, "source_run_authorized": False,
            "reserved_roots_opened": False, "new_neural_calls": 0, "new_physics_calls": 0}


def write(path, value):
    path = Path(path)
    need(path.is_absolute() and path.parent.resolve() == path.parent and path.name.startswith(NAME + "_") and path.suffix == ".json", "exclusive own output path")
    raw = canonical(value)
    need(len(raw) <= JSON_CAP, "output byte cap")
    with path.open("xb") as stream:
        need(stream.write(raw) == len(raw), "short output write")


def consume(path, expected):
    path = Path(path)
    need(path.name.startswith(NAME + "_") and path.suffix == ".json", "own saved report only")
    raw = small_read(path)
    need(digest(raw) == expected, "saved report SHA")
    need(canonical(parse(raw)) == canonical(extract()), "saved reconstruction/input closure")
    return {"status": "partial_evidence_verified_not_authority", "report_sha256": expected,
            "source_run_authorized": False, "reset_blockers_cleared": 0, "origin_review_required": True}


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
        out = consume(args.report, args.sha)
    else:
        out = extract()
        if args.output:
            write(args.output, out)
            out = {"path": str(args.output), "sha256": digest(canonical(out)), "summary": out["summary"], "source_run_authorized": False}
    print(canonical(out).decode())


if __name__ == "__main__":
    main()
