"""Bounded historical identity comparison; never grants admission or freshness.

Candidate fields are declared digests, not proof of how they were generated.
The fixed historical index includes prefix states and selected records only.
Run --self-test for contract tests; no models or simulator are imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "results/codex_d2p_historical_exclusion_index_20261029_LOCAL01.json"
INDEX_SHA = "ac2544512d033ec14fd07808c3abe2dfd96676bb6b82ec89569f498f58256f98"
SCHEMA = "d2p_bounded_candidate_identity.v1"
FIELDS = {"root_state_sha256", "root_sha256", "goal_rgb_sha256",
          "causal_history_action_sha256", "reset_seed"}
DIGEST = re.compile(r"^[0-9a-f]{64}$")


def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: " + key)
        out[key] = value
    return out


def read_json(path, expected, limit=16 * 1024 * 1024):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink():
        raise ValueError("absolute canonical nonsymlink path required")
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("input size or SHA mismatch")
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def candidate_validate(candidate):
    if type(candidate) is not dict or set(candidate) != FIELDS | {"schema"} or candidate["schema"] != SCHEMA:
        raise ValueError("candidate schema or fields mismatch")
    for key in FIELDS - {"reset_seed"}:
        value = candidate[key]
        if value is not None and (type(value) is not str or DIGEST.fullmatch(value) is None):
            raise ValueError(f"invalid {key}")
    seed = candidate["reset_seed"]
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("invalid reset_seed")
    if all(candidate[k] is None for k in FIELDS):
        raise ValueError("empty candidate")


def compare(index, candidate):
    candidate_validate(candidate)
    if index.get("schema") != "codex_d2p_historical_exclusion_index_20261029.v1":
        raise ValueError("index schema mismatch")
    if index.get("history_complete") is not False or index.get("freshness_certified") is not False:
        raise ValueError("unexpected index authority")
    prefixes, selected, envelopes = (index[k] for k in
                                     ("observed_prefix_records", "selected_declared_records", "conditional_seed_envelopes"))
    if (len(prefixes), len(selected), len(envelopes)) != (80, 52, 11):
        raise ValueError("index denominator mismatch")
    matches = []
    root_state = candidate["root_state_sha256"]
    if root_state is not None:
        for row in prefixes:
            if row["root_state_sha256"] == root_state:
                matches.append({"kind": "same_declared_prefix_state_sha256",
                                "source": row["evidence"], "unresolved": row["unresolved"]})
    selected_fields = ("root_sha256", "goal_rgb_sha256", "causal_history_action_sha256")
    for row in selected:
        for field in row["fields"]:
            key = field["pointer"].rsplit("/", 1)[-1]
            if key in selected_fields and candidate[key] is not None and field["value"] == candidate[key]:
                matches.append({"kind": "same_selected_declared_" + key,
                                "source_id": row["source_id"], "source": field})
    possible = []
    if candidate["reset_seed"] is not None:
        for env in envelopes:
            if env["start_inclusive"] <= candidate["reset_seed"] <= env["end_inclusive"]:
                possible.append({"source_id": env["source_id"], "kind": env["kind"],
                                 "start_inclusive": env["start_inclusive"], "end_inclusive": env["end_inclusive"],
                                 "certified": False, "assumptions": env["assumptions"], "unknown": env["unknown"]})
    return {"schema": "d2p_bounded_root_exclusion_report.v1",
            "verdict": "FAIL_CLOSED", "historical_index_sha256": INDEX_SHA,
            "candidate": candidate, "matched_identity_fields": matches,
            "conditional_seed_envelopes": possible,
            "index_scope": {"prefix_records": len(prefixes), "selected_records": len(selected),
                            "conditional_envelopes": len(envelopes), "history_complete": False},
            "not_checked": ["unindexed runs, failures, restarts, other machines", "semantic or approximate state/goal/history duplication",
                            "candidate digest derivation and actual reset binding", "checkpoint training data and split membership"],
            "freshness_certified": False, "training_disjoint_certified": False,
            "independent_confirmation_authorized": False,
            "interpretation": "Individual field matches are not a joint root identity. Absence of matches is unknown, not disjoint. "
                              "Prefix states need not be original reset states. Seed envelopes are conditional, not observed attempts."}


class Tests(unittest.TestCase):
    def setUp(self):
        self.index = read_json(INDEX, INDEX_SHA)
        self.c = {"schema": SCHEMA, **{k: None for k in FIELDS}}

    def test_known_prefix_match(self):
        self.c["root_state_sha256"] = self.index["observed_prefix_records"][0]["root_state_sha256"]
        result = compare(self.index, self.c)
        self.assertGreater(len(result["matched_identity_fields"]), 0)
        self.assertFalse(result["freshness_certified"])

    def test_no_match_not_fresh(self):
        self.c["root_state_sha256"] = "0" * 64
        result = compare(self.index, self.c)
        self.assertEqual(result["matched_identity_fields"], [])
        self.assertEqual(result["verdict"], "FAIL_CLOSED")

    def test_selected_identity_and_conditional_seed(self):
        item = next(f for f in self.index["selected_declared_records"][0]["fields"]
                    if f["pointer"].endswith("/goal_rgb_sha256"))
        self.c["goal_rgb_sha256"] = item["value"]
        self.c["reset_seed"] = self.index["conditional_seed_envelopes"][0]["start_inclusive"]
        result = compare(self.index, self.c)
        self.assertTrue(result["matched_identity_fields"])
        self.assertTrue(result["conditional_seed_envelopes"])
        self.assertFalse(result["conditional_seed_envelopes"][0]["certified"])

    def test_bad_candidate(self):
        with self.assertRaises(ValueError):
            compare(self.index, self.c)
        self.c["reset_seed"] = True
        with self.assertRaises(ValueError):
            compare(self.index, self.c)

    def test_index_tamper(self):
        with self.assertRaises(ValueError):
            read_json(INDEX, "0" * 64)

    def test_duplicate_json_key(self):
        with self.assertRaises(ValueError):
            json.loads('{"a":1,"a":2}', object_pairs_hook=unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate", type=Path, help="absolute path to bounded identity JSON")
    group.add_argument("--known-prefix-index", type=int, help="demonstration using an already-indexed prefix")
    group.add_argument("--self-test", action="store_true")
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--output", type=Path, help="optional new report file, exclusive create")
    args = parser.parse_args()
    if args.self_test:
        if args.output or args.candidate_sha256:
            parser.error("self-test takes no output or candidate hash")
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    try:
        index = read_json(INDEX, INDEX_SHA)
        if args.candidate:
            if not args.candidate_sha256 or not DIGEST.fullmatch(args.candidate_sha256):
                raise ValueError("candidate-sha256 required")
            candidate = read_json(args.candidate, args.candidate_sha256, 1024 * 1024)
            source = {"path": str(args.candidate), "sha256": args.candidate_sha256, "authority": "caller_declaration"}
        else:
            if args.candidate_sha256 or not 0 <= args.known_prefix_index < len(index["observed_prefix_records"]):
                raise ValueError("invalid known-prefix-index")
            candidate = {"schema": SCHEMA, **{k: None for k in FIELDS}}
            candidate["root_state_sha256"] = index["observed_prefix_records"][args.known_prefix_index]["root_state_sha256"]
            source = {"kind": "known_historical_prefix_demonstration", "index": args.known_prefix_index}
        result = compare(index, candidate)
        result["candidate_source"] = source
        data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if args.output:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        sys.stdout.buffer.write(data)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"verdict": "FAIL_CLOSED", "error": str(exc)}))
    return 2  # No outcome from this bounded check certifies independent admission.


if __name__ == "__main__":
    sys.exit(main())
