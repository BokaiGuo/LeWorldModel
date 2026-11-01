"""Validate a proposed historical attempt ledger; never certify completeness or freshness.

This is an intake contract, not a reconstruction of missing historical records.
Input is JSON on stdin, output is JSON on stdout, and exit 2 means blocked.
"""
import hashlib
import json
import re
import sys

SCHEMA = "p1_reset_ledger_intake.v1"
SHA = re.compile(r"^[0-9a-f]{64}$")
LIMIT = 4 * 1024 * 1024
MAX_ATTEMPTS = 100000


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def text(value):
    return type(value) is str and bool(value.strip()) and len(value) <= 512


def digest(value):
    return type(value) is str and SHA.fullmatch(value) is not None


def inspect(data):
    if type(data) is not dict or set(data) != {"schema", "calls"} or data["schema"] != SCHEMA:
        raise ValueError("invalid input schema")
    calls = data["calls"]
    if type(calls) is not list or len(calls) != 10:
        raise ValueError("exactly ten calls required")
    expected = {"A:1", "B:1", "B:2", "B:3", "C:1", "D:1", "E:1", "F:1", "G:1", "H:1"}
    seen, keys, collisions, missing = set(), {}, [], []
    count = 0
    for call in calls:
        if type(call) is not dict or set(call) != {"call_id", "ledger_sha256", "coverage", "attempts"}:
            raise ValueError("invalid call")
        ident = call["call_id"]
        if not text(ident) or ident not in expected or ident in seen:
            raise ValueError("unexpected or duplicate call")
        seen.add(ident)
        if call["ledger_sha256"] is not None and not digest(call["ledger_sha256"]):
            raise ValueError("invalid ledger digest")
        coverage = call["coverage"]
        if coverage not in ("unknown", "declared_complete"):
            raise ValueError("invalid coverage")
        attempts = call["attempts"]
        if type(attempts) is not list:
            raise ValueError("attempts must be a list")
        count += len(attempts)
        if count > MAX_ATTEMPTS:
            raise ValueError("too many attempts")
        if coverage == "unknown" or call["ledger_sha256"] is None:
            missing.append(ident)
        if attempts and call["ledger_sha256"] is None:
            raise ValueError("attempts require ledger digest")
        ids = set()
        for attempt in attempts:
            if type(attempt) is not dict or set(attempt) != {"attempt_id", "record_pointer", "state_sha256"}:
                raise ValueError("invalid attempt")
            aid, pointer, state = attempt["attempt_id"], attempt["record_pointer"], attempt["state_sha256"]
            if not text(aid) or aid in ids or not text(pointer) or not pointer.startswith("/"):
                raise ValueError("invalid attempt identity or pointer")
            if state is not None and not digest(state):
                raise ValueError("invalid state digest")
            ids.add(aid)
            ref = {"call_id": ident, "attempt_id": aid, "record_pointer": pointer}
            if state is None:
                missing.append(ident + "/" + aid + ":state")
            else:
                keys.setdefault(state, []).append(ref)
    if seen != expected:
        raise ValueError("missing call")
    for state, refs in sorted(keys.items()):
        if len(refs) > 1:
            collisions.append({"state_sha256": state, "records": refs})
    return {
        "schema": SCHEMA + ".report",
        "input_sha256": None,
        "calls": len(calls),
        "attempts": count,
        "missing_or_unknown": missing,
        "exact_state_digest_collisions": collisions,
        "interpretation": "Declared identities only; digests and pointers are not verified against original records. Exact byte equality is not physical pose deduplication.",
        "historical_coverage_certified": False,
        "training_disjoint": "unknown",
        "freshness_certified": False,
        "source_run_authorized": False,
        "P1_admission": False,
    }


def main():
    try:
        raw = sys.stdin.buffer.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError("input too large")
        data = json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        report = inspect(data)
        report["input_sha256"] = hashlib.sha256(raw).hexdigest()
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        print(json.dumps({"error": str(exc), "P1_admission": False}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
