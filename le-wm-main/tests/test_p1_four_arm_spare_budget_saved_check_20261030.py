import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_four_arm_spare_budget_saved_check_20261030 as module

DIRECTORY = module.ROOT / "results/p1_four_arm_spare_budget_cuda_20261030_RUN01"
REPORT_SHA = "07476d922fcfbe6413061b2d7d2d57c2abbf955d6401e9b42f937b8ba685c7a4"


@pytest.fixture
def sealed():
    if not (DIRECTORY / "report.json").is_file():
        pytest.skip("sealed known-root experiment not available")
    return DIRECTORY


def test_saved_run(sealed):
    result = module.check(sealed, REPORT_SHA)
    assert result["nodes"] == 16
    assert not result["fairness_proven"]


@pytest.mark.parametrize(("relative", "message"), [
    ("s0_random_b00_hook_audit.json", "hook receipt"),
    ("s0_random_b00_spare/ledger/events.jsonl", "ledger hash/path"),
    ("s0_random_b00_spare/candidates.jsonl", "trace hash/path"),
])
def test_tampered_receipts_are_rejected(sealed, monkeypatch, relative, message):
    original = Path.read_bytes
    target = sealed / relative

    def tampered(path):
        content = original(path)
        if path == target:
            if relative.endswith("hook_audit.json"):
                value = json.loads(content)
                value["boundary"] = 99
                return json.dumps(value).encode()
            return content + b" "
        return content

    monkeypatch.setattr(Path, "read_bytes", tampered)
    with pytest.raises(ValueError, match=message):
        module.check(sealed, REPORT_SHA)


def test_changed_action_rejected_even_with_new_report_hash(sealed, monkeypatch):
    original = Path.read_bytes
    target = sealed / "report.json"
    report = json.loads(original(target))
    report["rows"][0]["executed_raw"][0][0] += 1.0
    replacement = json.dumps(report).encode()
    digest = hashlib.sha256(replacement).hexdigest()
    monkeypatch.setattr(Path, "read_bytes", lambda path: replacement if path == target else original(path))
    with pytest.raises(ValueError, match="action binding"):
        module.check(sealed, digest)
