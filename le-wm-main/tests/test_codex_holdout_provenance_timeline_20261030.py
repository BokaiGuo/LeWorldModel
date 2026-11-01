"""Synthetic metadata only; no source events or timestamps are asserted as real."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

import pytest

from experiments.spectral_reachability import codex_holdout_provenance_timeline_20261030 as m


def fixture():
    doc = {
        "schema": m.SCHEMA,
        "checkpoint_sha256": m.CHECKPOINT,
        "protocol_sha256": "1" * 64,
        "source_sha256": "2" * 64,
        "dataset_manifest_sha256": "3" * 64,
        "events": [],
    }
    subjects = [m.CHECKPOINT, doc["protocol_sha256"], doc["protocol_sha256"],
                doc["source_sha256"], doc["dataset_manifest_sha256"]]
    for i, (role, subject) in enumerate(zip(m.ROLES, subjects)):
        doc["events"].append({
            "role": role, "subject_sha256": subject, "evidence_sha256": str(i + 4) * 64,
            "parent_evidence_sha256": None if i == 0 else str(i + 3) * 64,
            "not_before_utc": f"2000-01-0{i + 1}T00:00:00Z",
            "not_after_utc": f"2000-01-0{i + 1}T00:00:01Z",
        })
    return doc


NOW = datetime(2000, 2, 1, tzinfo=timezone.utc)


def test_consistency_never_grants_authority():
    report = m.check(fixture(), NOW)
    assert report["declarations"] == "structurally_consistent_unverified"
    assert report["training_exclusion"] == "unknown"
    assert report["status"] == "blocked" and report["evidence_bytes_checked"] is False
    assert all(report[k] is False for k in ("freshness_certified", "source_run_authorized", "p1_admitted"))


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(checkpoint_sha256="0" * 64),
    lambda d: d["events"][0].update(subject_sha256="0" * 64),
    lambda d: d["events"][1].update(role="acquisition"),
    lambda d: d["events"][1].update(not_before_utc=d["events"][0]["not_after_utc"]),
    lambda d: d["events"][1].update(not_after_utc="1999-01-01T00:00:00Z"),
    lambda d: d["events"][4].update(not_after_utc="2001-01-01T00:00:00Z"),
    lambda d: d["events"][1].update(evidence_sha256=d["events"][0]["evidence_sha256"]),
    lambda d: d["events"][1].update(evidence_sha256="G" * 64),
    lambda d: d["events"][3].update(parent_evidence_sha256="f" * 64),
    lambda d: d["events"][0].update(parent_evidence_sha256="f" * 64),
    lambda d: d.update(source_sha256=d["protocol_sha256"]),
    lambda d: d["events"][2].pop("parent_evidence_sha256"),
    lambda d: d.update(schema="codex.holdout.provenance_timeline.v1"),
    lambda d: d["events"][0].update(outcome="forbidden"),
    lambda d: d["events"].pop(),
    lambda d: d.update(extra=True),
])
def test_bad_evidence_or_order_fails_closed(mutation):
    doc = deepcopy(fixture())
    mutation(doc)
    with pytest.raises(ValueError):
        m.check(doc, NOW)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b"[" * 5000, b"x" * (m.MAX_BYTES + 1)])
def test_parse_fails_closed(raw):
    with pytest.raises((ValueError, RecursionError)):
        m.parse(raw)


def test_cli_binding_and_no_candidate_output():
    raw = json.dumps(fixture()).encode()
    cmd = [sys.executable, "-B", "-m", m.__name__, "--input-sha256"]
    good = subprocess.run(cmd + [hashlib.sha256(raw).hexdigest()], input=raw, capture_output=True, check=False)
    assert good.returncode == 0
    assert b"seed" not in good.stdout and b"candidate" not in good.stdout
    bad = subprocess.run(cmd + ["0" * 64], input=raw, capture_output=True, check=False)
    assert bad.returncode == 2
    assert json.loads(bad.stderr)["p1_admitted"] is False


def test_valid_declared_chain_is_not_cryptographic_proof():
    doc = fixture()
    doc["events"][2]["evidence_sha256"] = "f" * 64
    doc["events"][3]["parent_evidence_sha256"] = "f" * 64
    report = m.check(doc, NOW)
    assert report["evidence_bytes_checked"] is False
    assert report["source_independence"] == "unknown"
