"""Bind a saved state7 source to the bounded historical prefix index.

Only the explicit JSON pointer is used for identity. Exact state7 byte matches
are not original-reset, episode, training-disjoint, or freshness certificates.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

from experiments.spectral_reachability import d2p_bounded_root_exclusion_20261102_exclusive as bounded

SHA = re.compile(r"^[0-9a-f]{64}$")
MAX = 16 * 1024 * 1024


def read_source(path, expected):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not SHA.fullmatch(expected):
        raise ValueError("absolute canonical source and SHA256 required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(MAX + 1)
    if len(raw) > MAX or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("source byte hash or size mismatch")
    return json.loads(raw, object_pairs_hook=bounded.unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def pointer(value, ptr):
    if type(ptr) is not str or not ptr.startswith("/") or len(ptr) > 1024:
        raise ValueError("nonempty JSON pointer required")
    for token in ptr[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ValueError("invalid pointer escape")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ValueError("invalid array index")
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise ValueError("pointer traversal through scalar")
    return value


def state_digest(value):
    if not isinstance(value, list) or len(value) != 7:
        raise ValueError("state must be exactly 7 values")
    if any(type(x) not in (int, float) for x in value):
        raise ValueError("state requires JSON numbers")
    try:
        converted = [float(x) for x in value]
        if not all(math.isfinite(x) for x in converted):
            raise ValueError("state must be finite")
        return hashlib.sha256(struct.pack("<7d", *converted)).hexdigest()
    except OverflowError as exc:
        raise ValueError("state number out of range") from exc


def audit(source_path, source_sha, ptr):
    source = read_source(source_path, source_sha)
    digest = state_digest(pointer(source, ptr))
    index = bounded.read_json(bounded.INDEX, bounded.INDEX_SHA)
    candidate = {"schema": bounded.SCHEMA, **{k: None for k in bounded.FIELDS}}
    candidate["root_state_sha256"] = digest
    comparison = bounded.compare(index, candidate)
    verified_matches = []
    for row in comparison["matched_identity_fields"]:
        if row["kind"] != "same_declared_prefix_state_sha256":
            raise ValueError("unexpected match type")
        ref = row["source"]
        original = read_source(ref["path"], ref["sha256"])
        actual = state_digest(pointer(original, ref["pointer"]))
        if actual != digest:
            raise ValueError("indexed source state mismatch")
        verified_matches.append({"source": ref, "state_sha256": actual, "unresolved": row["unresolved"]})
    return {
        "schema": "d2p_root_state_source_binding_20261102.v1",
        "verdict": "FAIL_CLOSED",
        "candidate_source": {"path": str(source_path), "sha256": source_sha, "pointer": ptr,
                             "derived_state_sha256": digest, "encoding": "7 IEEE-754 float64 little-endian"},
        "historical_index": {"path": str(bounded.INDEX), "sha256": bounded.INDEX_SHA,
                             "coverage_complete": False, "prefix_records": 80},
        "verified_exact_prefix_matches": verified_matches,
        "historical_match_count": len(verified_matches),
        "freshness_certified": False, "training_disjoint_certified": False,
        "independent_confirmation_authorized": False,
        "limits": ["candidate source bytes are not proof of original reset or execution",
                   "observed prefix state is not necessarily sampling root",
                   "unmatched indexed records and outside-index history are not verified here",
                   "no semantic near-duplicate, goal, action history, seed, producer or training-split verification",
                   "missing or nonmatching identity never implies independent or fresh"]
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--pointer", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        report = audit(args.source, args.source_sha256, args.pointer)
        raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        print(json.dumps({"verdict": report["verdict"], "output": str(args.output),
                          "sha256": hashlib.sha256(raw).hexdigest(),
                          "historical_match_count": report["historical_match_count"]}))
    except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"verdict": "FAIL_CLOSED", "error": str(exc)}))
    return 2


if __name__ == "__main__":
    sys.exit(main())
