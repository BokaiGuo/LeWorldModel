import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_four_arm_hooked_saved_check_20261030_main as check


DIRECTORY = check.ROOT / "results/p1_four_arm_hooked_cuda_20261030_main_RUN01"
REPORT_SHA = "599407de034bed59d3c424fe5284f5317b524053e4af889ecc2c7bc5f432acb6"


@pytest.fixture
def sealed():
    if not (DIRECTORY / "report.json").is_file():
        pytest.skip("sealed exploratory run not available")
    return DIRECTORY


def test_saved_receipts_accept_sealed_run(sealed):
    result = check.check(sealed, REPORT_SHA)
    assert result["nodes"] == 16
    assert result["fairness_proven"] is False
    assert result["physical_event_chain_independently_verified"] is False


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("s0_structured_b00_hook_audit.json", "hook receipt"),
        ("s0_structured_b00_common/ledger/events.jsonl", "ledger path or hash"),
        ("s0_structured_b00_common/candidates.jsonl", "trace hash"),
    ],
)
def test_saved_tampering_rejected(sealed, monkeypatch, relative, message):
    target = sealed / relative
    original = Path.read_bytes

    def tampered(path):
        data = original(path)
        if path == target:
            if relative.endswith("hook_audit.json"):
                value = json.loads(data)
                value["boundary"] = 99
                return json.dumps(value).encode()
            return data + b" "
        return data

    monkeypatch.setattr(Path, "read_bytes", tampered)
    with pytest.raises(ValueError, match=message):
        check.check(sealed, REPORT_SHA)


def test_recorded_action_tampering_rejected_even_with_new_report_hash(sealed, monkeypatch):
    target = sealed / "report.json"
    original = Path.read_bytes
    report = json.loads(original(target))
    report["rows"][0]["executed_raw"][0][0] += 1.0
    replacement = json.dumps(report).encode()
    expected = hashlib.sha256(replacement).hexdigest()

    def tampered(path):
        return replacement if path == target else original(path)

    monkeypatch.setattr(Path, "read_bytes", tampered)
    with pytest.raises(ValueError, match="executed action binding"):
        check.check(sealed, expected)
