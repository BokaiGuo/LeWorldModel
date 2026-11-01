import copy
from datetime import datetime, timezone

import pytest

from experiments.spectral_reachability import codex_future_holdout_isolation_20261030 as m


@pytest.fixture
def fixture():
    exclusions = {"schema": m.NAME + ".known_exclusions.v1", "scope_sha256": "1" * 64,
                  "seeds": [], "spans": []}
    protocol = {
        "schema": m.NAME + ".protocol.v1",
        "batch_uuid": "12345678-1234-4234-8234-123456789abc",
        "source_uuid": "abcdefab-1234-4234-8234-123456789abc",
        "contract_code_sha256": m.hashlib.sha256(m.Path(m.__file__).read_bytes()).hexdigest(),
        "experiment_sha256": "2" * 64, "environment_sha256": "3" * 64,
        "checkpoint_sha256": "4" * 64, "normalizer_sha256": "5" * 64,
        "exclusion_sha256": m.sha(exclusions), "beacon_provider": "example",
        "beacon_round": "42", "beacon_scheduled_utc": "2031-01-01T00:00:00Z",
        "not_before_utc": "2031-01-01T00:10:00Z", "deadline_utc": "2031-01-02T00:00:00Z",
        "slot_count": 4,
    }
    receipt = {
        "schema": m.NAME + ".receipt.v1", "protocol_sha256": m.sha(protocol),
        "commit_utc": "2030-12-31T23:59:00Z", "commit_attestation_sha256": "6" * 64,
        "beacon_provider": "example", "beacon_round": "42", "beacon_value_hex": "a" * 64,
        "beacon_published_utc": "2031-01-01T00:01:00Z", "beacon_receipt_sha256": "7" * 64,
        "source_uuid": protocol["source_uuid"], "batch_uuid": protocol["batch_uuid"],
        "instance_uuid": "fedcba98-1234-4234-8234-123456789abc",
        "instance_created_utc": "2031-01-01T00:11:00Z",
        "custody_attestation_sha256": "8" * 64, "no_copy_attestation_sha256": "9" * 64,
    }
    return protocol, exclusions, receipt


def test_future_prepare_and_review_are_never_authority(fixture):
    p, x, r = fixture
    before = datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert not m.prepare(p, x, before)["source_run_authorized"]
    out = m.review(p, x, r, datetime(2031, 1, 1, 0, 12, tzinfo=timezone.utc))
    assert [row["seed"] for row in out["slots"]] == m.derive(r["beacon_value_hex"])
    assert all(row["attempts_allowed"] == 1 for row in out["slots"])
    assert out["training_disjoint"] == "unknown"
    assert out["historical_physical_state_disjoint"] == "unknown"
    assert out["source_event_independence"] == "unknown"
    assert not out["source_run_authorized"] and not out["freshness_certified"]


@pytest.mark.parametrize("case", ["past", "tampered_protocol", "round", "before_commit",
                                  "instance_early", "instance_late", "same_namespace",
                                  "extra_outcome", "nonhex", "bool_count"])
def test_fail_closed(fixture, case):
    p, x, r = copy.deepcopy(fixture)
    if case == "past":
        with pytest.raises(ValueError):
            m.prepare(p, x, datetime(2031, 1, 1, tzinfo=timezone.utc))
        return
    if case == "tampered_protocol":
        p["checkpoint_sha256"] = "0" * 64
    elif case == "round":
        r["beacon_round"] = "43"
    elif case == "before_commit":
        r["commit_utc"] = "2031-01-01T00:00:00Z"
    elif case == "instance_early":
        r["instance_created_utc"] = "2031-01-01T00:09:00Z"
    elif case == "instance_late":
        r["instance_created_utc"] = "2031-01-02T00:00:00Z"
    elif case == "same_namespace":
        r["instance_uuid"] = r["source_uuid"]
    elif case == "extra_outcome":
        r["outcome"] = "success"
    elif case == "nonhex":
        r["beacon_value_hex"] = "x" * 64
    else:
        p["slot_count"] = True
    with pytest.raises(ValueError):
        m.review(p, x, r, datetime(2031, 1, 1, 0, 12, tzinfo=timezone.utc))


@pytest.mark.parametrize("kind", ["exact", "span", "duplicate_batch"])
def test_collision_blocks_whole_batch(fixture, kind, monkeypatch):
    p, x, r = copy.deepcopy(fixture)
    seeds = m.derive(r["beacon_value_hex"])
    if kind == "duplicate_batch":
        monkeypatch.setattr(m, "derive", lambda _: [1, 2, 2, 3])
    else:
        if kind == "exact":
            x["seeds"] = [seeds[0]]
        else:
            x["spans"] = [[seeds[0], seeds[0] + 1]]
        p["exclusion_sha256"] = m.sha(x)
        r["protocol_sha256"] = m.sha(p)
    with pytest.raises(ValueError, match="collision"):
        m.review(p, x, r, datetime(2031, 1, 1, 0, 12, tzinfo=timezone.utc))


def test_no_followup_collision_at_exclusive_span_end(fixture):
    p, x, r = copy.deepcopy(fixture)
    seed = m.derive(r["beacon_value_hex"])[0]
    assert seed > 0
    x["spans"] = [[seed - 1, seed]]
    p["exclusion_sha256"] = m.sha(x)
    r["protocol_sha256"] = m.sha(p)
    m.review(p, x, r, datetime(2031, 1, 1, 0, 12, tzinfo=timezone.utc))


def test_future_receipt_not_yet_observable(fixture):
    p, x, r = fixture
    with pytest.raises(ValueError, match="not yet observable"):
        m.review(p, x, r, datetime(2030, 1, 1, tzinfo=timezone.utc))


def test_stable_seed_vectors():
    assert m.derive("00" * 32) == [513355489, 2947742943, 3559069454, 813717503]
