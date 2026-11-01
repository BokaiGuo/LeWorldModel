import copy
import json

import pytest

from experiments.spectral_reachability import p1_future_root_queue_20261101_codex as q


def spec():
    return dict(schema=q.SCHEMA + ".spec.v1", experiment_id="fixture",
                producer_sha256="1"*64, environment_sha256="2"*64,
                protocol_sha256="3"*64, model_sha256="4"*64,
                slots=[dict(slot_id="d", role="development", reset_seeds=[1, 2]),
                       dict(slot_id="c", role="confirmation", reset_seeds=[3])])


def event(kind, sid, attempt, seed, payload):
    return dict(kind=kind, slot_id=sid, attempt=attempt, seed=seed, payload=payload)


def test_freeze_rejects_overlap_and_unknown_fields():
    value = spec()
    value["slots"][1]["reset_seeds"] = [1]
    with pytest.raises(ValueError):
        q.freeze(value)
    value = spec()
    value["slots"][0]["reset_seeds"] = [202753000]
    with pytest.raises(ValueError, match="historical quarantine"):
        q.freeze(value)
    value = spec()
    value["success"] = True
    with pytest.raises(ValueError):
        q.freeze(value)


def test_append_and_incomplete_failed_attempt(tmp_path):
    queue = q.freeze(spec())
    path, ledger = tmp_path / "queue.json", tmp_path / "ledger.jsonl"
    q.write_new(path, queue)
    q.append(path, ledger, event("reset_attempt", "d", 0, 1, {}))
    events = [json.loads(line) for line in ledger.read_bytes().splitlines()]
    result = q.validate(queue, events)
    assert result["pending"] == {"d": 0}
    assert result["training_membership"] == "unknown" and not result["P1_admission"]
    q.append(path, ledger, event("reset_failed", "d", 0, 1, {"error_class": "ResetError"}))
    with pytest.raises(ValueError):
        q.append(path, ledger, event("reset_attempt", "d", 0, 1, {}))
    with pytest.raises(ValueError):
        q.append(path, ledger, event("reset_attempt", "d", 1, 99, {}))


def test_state_collision_and_queue_rebinding(tmp_path):
    queue = q.freeze(spec())
    path, ledger = tmp_path / "queue.json", tmp_path / "ledger.jsonl"
    q.write_new(path, queue)
    hashes = {key: "a"*64 for key in ("state_sha256", "goal_sha256", "history_sha256", "snapshot_sha256")}
    q.append(path, ledger, event("reset_attempt", "d", 0, 1, {}))
    q.append(path, ledger, event("reset_return", "d", 0, 1, hashes))
    q.append(path, ledger, event("reset_attempt", "c", 0, 3, {}))
    q.append(path, ledger, event("reset_return", "c", 0, 3, hashes))
    events = [json.loads(line) for line in ledger.read_bytes().splitlines()]
    assert len(q.validate(queue, events)["cross_partition_state_collisions"]) == 1
    changed = copy.deepcopy(spec())
    changed["experiment_id"] = "other"
    with pytest.raises(ValueError):
        q.validate(q.freeze(changed), events)


def test_exclusive_and_tamper(tmp_path):
    path = tmp_path / "queue.json"
    q.write_new(path, q.freeze(spec()))
    with pytest.raises(FileExistsError):
        q.write_new(path, q.freeze(spec()))
    ledger = tmp_path / "events.jsonl"
    q.append(path, ledger, event("reset_attempt", "d", 0, 1, {}))
    rows = [json.loads(line) for line in ledger.read_bytes().splitlines()]
    rows[0]["seed"] = 7
    with pytest.raises(ValueError):
        q.validate(q.freeze(spec()), rows)
