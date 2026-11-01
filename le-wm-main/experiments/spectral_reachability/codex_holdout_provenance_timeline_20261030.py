"""Metadata-only future holdout chain lint; never authenticates or admits a source."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import sys

CHECKPOINT = "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
SCHEMA = "codex.holdout.provenance_timeline.v2"
MAX_BYTES = 262144
ROLES = ("checkpoint_freeze", "protocol_commitment", "beacon_publication",
         "acquisition", "dataset_seal")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fields(value, expected):
    require(type(value) is dict and set(value) == set(expected), "unexpected or missing fields")


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "invalid sha256")
    return value


def timestamp(value):
    require(type(value) is str and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value) is not None,
            "invalid UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("invalid UTC timestamp") from exc


def check(document, now):
    """Check asserted links only. No file, network, signature, or timestamp verification."""
    fields(document, ("schema", "checkpoint_sha256", "protocol_sha256", "source_sha256",
                      "dataset_manifest_sha256", "events"))
    require(document["schema"] == SCHEMA, "schema mismatch")
    for key in ("checkpoint_sha256", "protocol_sha256", "source_sha256", "dataset_manifest_sha256"):
        digest(document[key])
    require(document["checkpoint_sha256"] == CHECKPOINT, "checkpoint mismatch")
    require(len({document["checkpoint_sha256"], document["protocol_sha256"],
                 document["source_sha256"], document["dataset_manifest_sha256"]}) == 4,
            "distinct object identities required")
    require(type(now) is datetime and now.tzinfo is not None and now.utcoffset().total_seconds() == 0,
            "UTC clock required")
    events = document["events"]
    require(type(events) is list and len(events) == len(ROLES), "five events required")
    expected_subjects = (document["checkpoint_sha256"], document["protocol_sha256"],
                         document["protocol_sha256"], document["source_sha256"],
                         document["dataset_manifest_sha256"])
    previous_upper = None
    previous_evidence = None
    evidence = set()
    for role, subject, event in zip(ROLES, expected_subjects, events):
        fields(event, ("role", "subject_sha256", "evidence_sha256", "parent_evidence_sha256",
                       "not_before_utc", "not_after_utc"))
        require(event["role"] == role, "event order or role mismatch")
        require(digest(event["subject_sha256"]) == subject, "event subject mismatch")
        require(event["parent_evidence_sha256"] == previous_evidence, "evidence chain mismatch")
        previous_evidence = digest(event["evidence_sha256"])
        evidence.add(previous_evidence)
        lower, upper = timestamp(event["not_before_utc"]), timestamp(event["not_after_utc"])
        require(lower <= upper <= now, "invalid or future event interval")
        if previous_upper is not None:
            require(previous_upper < lower, "event intervals overlap or are unordered")
        previous_upper = upper
    require(len(evidence) == len(ROLES), "distinct evidence references required")
    return {
        "schema": SCHEMA + ".report",
        "status": "blocked",
        "declarations": "structurally_consistent_unverified",
        "clock": "local_untrusted",
        "evidence_bytes_checked": False,
        "blocking": [
            "checkpoint_freeze_and_immutability_not_verified",
            "commitment_timestamp_and_signature_not_verified",
            "beacon_identity_round_value_and_signature_not_verified",
            "acquisition_time_custody_and_no_copy_not_verified",
            "manifest_bytes_membership_and_source_binding_not_verified",
            "training_lineage_or_future_exclusion_not_verified",
            "historical_physical_state_disjointness_not_verified",
            "protocol_abi_exclusions_and_authorization_not_verified",
        ],
        "training_exclusion": "unknown",
        "source_independence": "unknown",
        "freshness_certified": False,
        "source_run_authorized": False,
        "p1_admitted": False,
    }


def parse(raw):
    require(len(raw) <= MAX_BYTES, "input too large")

    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, "duplicate JSON key")
            obj[key] = value
        return obj

    def nonfinite(_):
        raise ValueError("nonfinite JSON")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sha256", required=True, help="expected SHA256 of stdin bytes")
    args = parser.parse_args()
    try:
        digest(args.input_sha256)
        raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "input too large")
        require(hashlib.sha256(raw).hexdigest() == args.input_sha256, "input digest mismatch")
        report = check(parse(raw), datetime.now(timezone.utc))
    except (ValueError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc),
                          "freshness_certified": False, "source_run_authorized": False,
                          "p1_admitted": False}), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
