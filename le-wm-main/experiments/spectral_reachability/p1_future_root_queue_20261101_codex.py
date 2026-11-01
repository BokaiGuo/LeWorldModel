"""Freeze an explicitly supplied root queue and audit its prospective reset ledger.

No environment, model, source generation or confirmation authorization.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat

SCHEMA = "p1_future_root_queue_20261101_codex"
HASH = re.compile(r"^[0-9a-f]{64}$")
SLOT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_BYTES = 4 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "results/p1_future_root_history_inventory_20261101_codex_LOCAL01.json"
HISTORY_SHA = "70ab1a9b0a26ec16c9851c377cd275de4b4ada91c40ad6543e5382bd429ab8e4"
QUARANTINE = ROOT / "results/moving_historical_reset_quarantine_20261027_v1_saved01.json"
QUARANTINE_SHA = "fe27d07b749c8a518faee2667f2c256c97fd94ea80d5ae25347560ec72bf3b70"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    raw = Path(path).read_bytes()
    require(len(raw) <= MAX_BYTES, "input size")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def reject_known_seed_hits(seeds):
    for path, expected in ((HISTORY, HISTORY_SHA), (QUARANTINE, QUARANTINE_SHA)):
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "historical inventory drift")
    history, quarantine = read_json(HISTORY), read_json(QUARANTINE)
    require(not history["unknown_sources"] and quarantine["source_count"] == 8, "historical inventory unavailable")
    observed = set()
    for row in history["baseline"]["events"]:
        seed = row.get("seed", row.get("start_event", {}).get("seed"))
        if type(seed) is int:
            observed.add(seed)
    observed.update(row["seed"] for row in history["additional_overlay"]["events"] if type(row["seed"]) is int)
    spans = [(e["start"], e["end"]) for row in quarantine["rows"]
             for e in row["conditional_envelopes"]]
    # Seed 0 is also used for explicit-state placement; rejection is a
    # conservative label quarantine, not evidence that states are identical.
    observed.add(0)
    require(not any(seed in observed or any(a <= seed <= b for a, b in spans) for seed in seeds),
            "seed overlaps listed observed or conditional historical quarantine")


def freeze(spec):
    require(type(spec) is dict and set(spec) == {
        "schema", "experiment_id", "producer_sha256", "environment_sha256",
        "protocol_sha256", "model_sha256", "slots"}, "spec fields")
    require(spec["schema"] == SCHEMA + ".spec.v1"
            and type(spec["experiment_id"]) is str and SLOT.fullmatch(spec["experiment_id"]), "spec identity")
    for field in ("producer_sha256", "environment_sha256", "protocol_sha256", "model_sha256"):
        require(type(spec[field]) is str and HASH.fullmatch(spec[field]), field)
    slots = spec["slots"]
    require(type(slots) is list and 2 <= len(slots) <= 10000, "slot count")
    ids, seeds, roles = set(), set(), set()
    for slot in slots:
        require(type(slot) is dict and set(slot) == {"slot_id", "role", "reset_seeds"}, "slot fields")
        sid, role, numbers = slot["slot_id"], slot["role"], slot["reset_seeds"]
        require(type(sid) is str and SLOT.fullmatch(sid) and sid not in ids, "slot id")
        require(role in ("development", "confirmation"), "role")
        require(type(numbers) is list and 1 <= len(numbers) <= 1024, "attempt roster")
        for seed in numbers:
            require(type(seed) is int and 0 <= seed <= 2**63-1 and seed not in seeds, "seed uniqueness")
            seeds.add(seed)
        ids.add(sid)
        roles.add(role)
    require(roles == {"development", "confirmation"}, "both partitions required")
    reject_known_seed_hits(seeds)
    return {"schema": SCHEMA + ".frozen.v1", "spec": spec, "spec_sha256": digest(spec),
            "queue_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "isolation": "declared_seed_and_slot_disjoint_not_physical_identity_or_access_control",
            "bounded_history_inventory_sha256": HISTORY_SHA,
            "conditional_quarantine_sha256": QUARANTINE_SHA,
            "known_seed_quarantine": "no_listed_seed_or_conditional_span_hit_not_global_exclusion",
            "training_membership": "unknown", "historical_coverage": "unknown",
            "freshness": "unknown", "P1_admission": False,
            "external_timestamp_verified": False, "execution_authorized": False}


def validate(queue, events):
    require(type(queue) is dict and queue == freeze(queue.get("spec")), "queue integrity")
    slots = {s["slot_id"]: s for s in queue["spec"]["slots"]}
    previous, pending, seen, states, collisions = None, {}, set(), {}, []
    results = []
    for index, event in enumerate(events):
        require(type(event) is dict and set(event) == {"index", "previous", "queue_sha256", "kind", "slot_id",
                                                      "attempt", "seed", "payload", "sha256"}, "event fields")
        body = {k: v for k, v in event.items() if k != "sha256"}
        require(event["index"] == index and type(event["index"]) is int and event["previous"] == previous
                and event["queue_sha256"] == digest(queue) and event["sha256"] == digest(body), "event hash chain")
        sid = event["slot_id"]
        require(sid in slots and type(event["attempt"]) is int and type(event["seed"]) is int, "event identity")
        attempt = event["attempt"]
        roster = slots[sid]["reset_seeds"]
        require(0 <= attempt < len(roster) and event["seed"] == roster[attempt], "attempt outside frozen roster")
        key = (sid, attempt)
        payload = event["payload"]
        require(type(payload) is dict, "payload")
        if event["kind"] == "reset_attempt":
            require(not payload and key not in seen and sid not in pending, "attempt duplicated or pending")
            require(attempt == sum(1 for s, _ in seen if s == sid), "attempt order")
            seen.add(key)
            pending[sid] = attempt
        elif event["kind"] in ("reset_return", "reset_failed"):
            require(pending.get(sid) == attempt, "unpaired return")
            del pending[sid]
            if event["kind"] == "reset_return":
                require(set(payload) == {"state_sha256", "goal_sha256", "history_sha256", "snapshot_sha256"}
                        and all(type(v) is str and HASH.fullmatch(v) for v in payload.values()), "identity hashes")
                # Exact state hash reuse across roles is a hard collision; distinct
                # hashes do not prove physical or distributional independence.
                state = payload["state_sha256"]
                if state in states:
                    for earlier in states[state]:
                        if earlier["role"] != slots[sid]["role"]:
                            collisions.append({"state_sha256": state, "earlier": earlier["slot_id"],
                                               "later": sid})
                states.setdefault(state, []).append({"slot_id": sid, "role": slots[sid]["role"]})
            else:
                require(set(payload) == {"error_class"} and type(payload["error_class"]) is str
                        and SLOT.fullmatch(payload["error_class"]), "failure identity")
            results.append({"slot_id": sid, "attempt": attempt, "kind": event["kind"]})
        else:
            raise ValueError("unknown event kind")
        previous = event["sha256"]
    return {"schema": SCHEMA + ".audit.v1", "queue_sha256": digest(queue), "events": len(events),
            "ledger_tail": previous, "pending": pending, "attempts_recorded": len(seen), "returns": results,
            "planned_attempts": sum(len(s["reset_seeds"]) for s in slots.values()),
            "cross_partition_state_collisions": collisions,
            "all_attempts_accounted": len(seen) == sum(len(s["reset_seeds"]) for s in slots.values()) and not pending,
            "scope": "supplied ledger only; no proof of logging completeness or original byte provenance",
            "access_isolation_verified": False, "external_timestamp_verified": False,
            "training_membership": "unknown", "historical_coverage": "unknown",
            "freshness": "unknown", "independent_confirmation": False, "P1_admission": False}


def write_new(path, value):
    data = canonical(value) + b"\n"
    require(len(data) <= MAX_BYTES, "output size")
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def append(queue_path, ledger_path, event):
    queue = read_json(queue_path)
    # Cooperating writers serialize on this file; the chain binds every event
    # to the frozen queue. This is not a device-wide or access-control lock.
    fd = os.open(ledger_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        st = os.fstat(fd)
        require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_size <= MAX_BYTES, "ledger file")
        with os.fdopen(fd, "r+b", closefd=False) as stream:
            raw = stream.read(MAX_BYTES + 1)
            require(not raw or raw.endswith(b"\n"), "incomplete ledger")
            events = [json.loads(line) for line in raw.splitlines()]
            require(type(event) is dict and set(event) == {"kind", "slot_id", "attempt", "seed", "payload"},
                    "append fields")
            body = {"index": len(events), "previous": events[-1]["sha256"] if events else None,
                    "queue_sha256": digest(queue), **event}
            row = {**body, "sha256": digest(body)}
            validate(queue, events + [row])
            encoded = canonical(row) + b"\n"
            require(len(raw) + len(encoded) <= MAX_BYTES, "ledger cap")
            stream.seek(0, os.SEEK_END)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            return row["sha256"]
    finally:
        os.close(fd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--spec", required=True)
    f.add_argument("--output", required=True)
    a = sub.add_parser("append")
    a.add_argument("--queue", required=True)
    a.add_argument("--ledger", required=True)
    a.add_argument("--event", required=True)
    v = sub.add_parser("audit")
    v.add_argument("--queue", required=True)
    v.add_argument("--ledger", required=True)
    v.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "freeze":
        print(write_new(args.output, freeze(read_json(args.spec))))
    elif args.command == "append":
        print(append(args.queue, args.ledger, read_json(args.event)))
    else:
        raw = Path(args.ledger).read_bytes()
        require(len(raw) <= MAX_BYTES and (not raw or raw.endswith(b"\n")), "ledger bytes")
        result = validate(read_json(args.queue), [json.loads(line) for line in raw.splitlines()])
        print(write_new(args.output, result))


if __name__ == "__main__":
    main()
