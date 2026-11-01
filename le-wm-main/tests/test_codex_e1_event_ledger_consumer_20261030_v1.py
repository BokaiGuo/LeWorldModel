import copy
import json
import subprocess
import sys

import pytest
from experiments.spectral_reachability import codex_e1_event_ledger_consumer_20261030_v1 as m


@pytest.fixture(scope="module")
def actual():
    a = m.audit
    targets = a.parse(a.read_bound(m.TARGETS, m.TARGETS_SHA, 4 * 1024 * 1024))
    prepare = a.parse(a.read_bound(m.PREPARE, m.PREPARE_SHA, 4 * 1024 * 1024))
    report = a.parse(a.read_bound(a.REPORT, a.REPORT_SHA, 4 * 1024 * 1024))
    receipt = a.parse(a.read_bound(a.RECEIPT, a.RECEIPT_SHA, 4 * 1024 * 1024))
    records, terminal = a.decode(a.read_bound(a.EVENTS, a.EVENT_SHA, 16 * 1024 * 1024))
    return targets, prepare, report, a.audit(records, terminal, report, receipt)


def test_real_sources_into_incomplete_ledger(actual):
    out = m.consume(*actual)
    assert len(out["slots"]) == 6 and not out["equal_budget_certified"]
    for item in out["slots"]:
        phases = item["ledger_input"]["phases"]
        assert item["ledger_preflight"]["structural_status"] == "incomplete"
        assert phases["proposal"]["scope"] == phases["inherited"]["scope"] == "unknown"
        assert all(x is None for x in phases["proposal"]["measurements"].values())
        assert all(x is None for x in phases["inherited"]["measurements"].values())
        for phase, n in (("controller", 144), ("replay", 4)):
            assert phases[phase]["scope"] == "per_arm"
            assert phases[phase]["receipt_sha256"] == m.audit.RECEIPT_SHA
            assert sum(x is not None for x in phases[phase]["measurements"].values()) == 4
            for axis in m.AXES:
                assert phases[phase]["measurements"][axis]["attempted"] == n
    assert out["slots"][0]["ledger_input"]["phases"]["controller"]["measurements"]["predict_api_calls"]["event_set_sha256"] == "9004ef70ea41ef0cda8dc0f6c474194d354bd4ee2eca71bd362fb10dd4ff8f73"


@pytest.mark.parametrize("mutate", [
    lambda x: x[0]["rows"][0].update(target_data_sha256="0" * 64),
    lambda x: x[0]["rows"][0].update(root_slot=1),
    lambda x: x[1].update(six_sha256="0" * 64),
    lambda x: x[2].update(prepare_sha256="0" * 64),
    lambda x: x[3]["groups"][0].update(arm="wrong"),
    lambda x: x[3]["groups"].__setitem__(0, copy.deepcopy(x[3]["groups"][1])),
    lambda x: x[3]["groups"][0]["measurements"].pop("predictor_samples"),
    lambda x: x[3]["groups"][0]["measurements"]["predict_api_calls"].update(completed=9999),
])
def test_bad_joins_or_measurements_rejected(actual, mutate):
    data = copy.deepcopy(actual)
    mutate(data)
    with pytest.raises((ValueError, KeyError)):
        m.consume(*data)


def test_cli_real_sources():
    result = subprocess.run([sys.executable, "-B", "-m", m.__name__], capture_output=True, check=False)
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["status"] == "blocked_incomplete" and not report["scientific_admission"]


def test_pin_rejected():
    with pytest.raises(ValueError):
        m.audit.read_bound(m.PREPARE, "0" * 64, 4 * 1024 * 1024)
