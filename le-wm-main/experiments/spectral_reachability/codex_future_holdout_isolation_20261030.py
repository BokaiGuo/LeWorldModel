"""Future-only holdout metadata contract. No reset, external trust or admission."""
import argparse
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parents[2]
NAME = "codex_future_holdout_isolation_20261030"
DOMAIN = b"p1-future-holdout-v1/root/"
MAX_SEED = 2**32


def need(condition, message):
    if not condition:
        raise ValueError(message)


def fields(obj, names):
    need(type(obj) is dict and set(obj) == set(names.split()), "exact fields: " + names)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def hash_value(value):
    need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "SHA256 required")


def identifier(value):
    need(type(value) is str, "UUID string required")
    try:
        u = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError("invalid UUID") from exc
    need(str(u) == value and u.int != 0, "canonical nonzero UUID required")


def instant(value):
    need(type(value) is str and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", value) is not None,
         "UTC second timestamp required")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("invalid UTC timestamp") from exc


def seed_int(v, upper=MAX_SEED):
    need(type(v) is int and 0 <= v < upper, "seed integer out of range")


def validate_protocol(p):
    fields(p, "schema batch_uuid source_uuid experiment_sha256 contract_code_sha256 environment_sha256 checkpoint_sha256 "
           "normalizer_sha256 exclusion_sha256 beacon_provider beacon_round beacon_scheduled_utc "
           "not_before_utc deadline_utc slot_count")
    need(p["schema"] == NAME + ".protocol.v1", "protocol schema")
    identifier(p["batch_uuid"]); identifier(p["source_uuid"])
    need(p["batch_uuid"] != p["source_uuid"], "namespace UUID collision")
    for k in ("experiment_sha256", "contract_code_sha256", "environment_sha256", "checkpoint_sha256", "normalizer_sha256", "exclusion_sha256"):
        hash_value(p[k])
    need(p["contract_code_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "contract code binding mismatch")
    for k in ("beacon_provider", "beacon_round"):
        need(type(p[k]) is str and re.fullmatch(r"[A-Za-z0-9._:/-]{1,128}", p[k]) is not None, "beacon identity")
    need(type(p["slot_count"]) is int and p["slot_count"] == 4, "fixed four slots")
    need(instant(p["beacon_scheduled_utc"]) < instant(p["not_before_utc"]) < instant(p["deadline_utc"]),
         "scheduled beacon before future window")
    return sha(p)


def validate_exclusions(x, expected):
    fields(x, "schema scope_sha256 seeds spans")
    need(x["schema"] == NAME + ".known_exclusions.v1" and sha(x) == expected, "exclusion binding")
    hash_value(x["scope_sha256"])
    need(type(x["seeds"]) is list and len(x["seeds"]) <= 100000, "bounded seed list")
    for s in x["seeds"]:
        seed_int(s)
    need(x["seeds"] == sorted(set(x["seeds"])), "seeds unique and sorted")
    need(type(x["spans"]) is list and len(x["spans"]) <= 100000, "bounded span list")
    previous_end = -1
    for span in x["spans"]:
        need(type(span) is list and len(span) == 2, "half-open span")
        a, b = span
        seed_int(a); need(type(b) is int and a < b <= MAX_SEED, "span end")
        need(a >= previous_end, "sorted nonoverlapping spans")
        previous_end = b
    return x


def derive(value_hex):
    hash_value(value_hex)
    key = bytes.fromhex(value_hex)
    return [int.from_bytes(hmac.new(key, DOMAIN + i.to_bytes(4, "big"), hashlib.sha256).digest()[:4], "big")
            for i in range(4)]


def prepare(protocol, exclusions, now):
    digest = validate_protocol(protocol)
    validate_exclusions(exclusions, protocol["exclusion_sha256"])
    need(now.tzinfo is not None and now.utcoffset().total_seconds() == 0, "UTC clock required")
    need(now < instant(protocol["beacon_scheduled_utc"]), "beacon must still be in the future")
    return {"protocol_sha256": digest, "status": "blocked_external_commitment_and_beacon_required",
            "source_run_authorized": False, "freshness_certified": False, "p1_admitted": False}


def review(protocol, exclusions, receipt, now):
    protocol_sha = validate_protocol(protocol)
    x = validate_exclusions(exclusions, protocol["exclusion_sha256"])
    fields(receipt, "schema protocol_sha256 commit_utc commit_attestation_sha256 beacon_provider beacon_round "
           "beacon_value_hex beacon_published_utc beacon_receipt_sha256 source_uuid batch_uuid instance_uuid "
           "instance_created_utc custody_attestation_sha256 no_copy_attestation_sha256")
    need(receipt["schema"] == NAME + ".receipt.v1" and receipt["protocol_sha256"] == protocol_sha,
         "receipt protocol binding")
    for k in ("commit_attestation_sha256", "beacon_receipt_sha256", "custody_attestation_sha256", "no_copy_attestation_sha256"):
        hash_value(receipt[k])
    for k in ("source_uuid", "batch_uuid", "instance_uuid"):
        identifier(receipt[k])
    need(receipt["source_uuid"] == protocol["source_uuid"] and receipt["batch_uuid"] == protocol["batch_uuid"],
         "source and batch binding")
    need(len({receipt["source_uuid"], receipt["batch_uuid"], receipt["instance_uuid"]}) == 3,
         "distinct namespace identifiers")
    need(receipt["beacon_provider"] == protocol["beacon_provider"] and receipt["beacon_round"] == protocol["beacon_round"],
         "beacon round mismatch")
    commit, scheduled = instant(receipt["commit_utc"]), instant(protocol["beacon_scheduled_utc"])
    published, created = instant(receipt["beacon_published_utc"]), instant(receipt["instance_created_utc"])
    need(commit < scheduled <= published < created, "commit/beacon/instance order")
    need(now.tzinfo is not None and now.utcoffset().total_seconds() == 0 and created <= now,
         "instance timestamp is not yet observable")
    need(instant(protocol["not_before_utc"]) <= created < instant(protocol["deadline_utc"]),
         "instance outside future window")
    seeds = derive(receipt["beacon_value_hex"])
    need(len(set(seeds)) == 4, "seed collision: whole batch blocked, no redraw")
    for seed in seeds:
        need(seed not in x["seeds"] and not any(a <= seed < b for a, b in x["spans"]),
             "known seed or span collision: whole batch blocked, no redraw")
    return {
        "schema": NAME + ".review.v1", "protocol_sha256": protocol_sha,
        "seed_derivation": "HMAC-SHA256(first four bytes, big endian); four fixed indexes, no redraw",
        "slots": [{"index": i, "seed": seed, "attempts_allowed": 1, "status": "collection_unobserved"}
                  for i, seed in enumerate(seeds)],
        "syntactic_checks": "consistent_declarations_only",
        "blocking_unverified": ["trusted_commitment_time_and_content", "beacon_signature_and_round",
                                "custodian_and_new_instance", "no_copy_and_append_only_custody",
                                "exclusion_scope_and_delta", "runtime_abi_and_actual_attempt_journal",
                                "explicit_future_collection_authority"],
        "training_disjoint": "unknown", "historical_physical_state_disjoint": "unknown",
        "sampling_independence": "unknown", "source_event_independence": "unknown",
        "source_run_authorized": False, "freshness_certified": False, "p1_admitted": False,
    }


def read_input(path):
    need(path.is_absolute() and path.parent == ROOT / "results" and path.resolve() == path
         and path.name.startswith("codex_future_holdout_input_") and path.suffix == ".json"
         and path.is_file() and not path.is_symlink(), "own explicit input file required")
    need(path.stat().st_size <= 4 * 1024 * 1024, "input cap")
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, "duplicate JSON key")
            result[k] = v
        return result
    raw = path.read_bytes()
    need(len(raw) <= 4 * 1024 * 1024, "input cap")
    return raw, json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda _: need(False, "nonfinite JSON"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "review"))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--input-sha256", required=True)
    args = p.parse_args()
    hash_value(args.input_sha256)
    raw, obj = read_input(args.input)
    # Bind input bytes, not reserialized semantics; a changed file fails closed.
    need(hashlib.sha256(raw).hexdigest() == args.input_sha256, "input SHA mismatch")
    fields(obj, "protocol exclusions receipt")
    if args.mode == "prepare":
        need(obj["receipt"] is None, "no beacon before commitment")
        result = prepare(obj["protocol"], obj["exclusions"], datetime.now(timezone.utc))
    else:
        result = review(obj["protocol"], obj["exclusions"], obj["receipt"], datetime.now(timezone.utc))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
