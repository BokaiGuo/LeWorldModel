import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_joint_prefix_suffix_saved_check_20261030 as checker

REPORT = checker.ROOT / "results/p1_joint_prefix_suffix_cuda_20261030_RUN01.json"


def test_actual_sealed_run():
    if not REPORT.is_file():
        pytest.skip("sealed CUDA result unavailable")
    result = checker.check(REPORT)
    assert result["candidates"] == result["initial_rollout_exact"] == 160
    assert not result["P1_admission"]


def test_changed_action_is_rejected_with_updated_report_hash(monkeypatch):
    if not REPORT.is_file():
        pytest.skip("sealed CUDA result unavailable")
    original = Path.read_bytes
    report = json.loads(original(REPORT))
    report["rows"][0]["trajectory"][0]["actions"][0][0][0] += 0.1
    replacement = json.dumps(report).encode()
    monkeypatch.setattr(Path, "read_bytes", lambda path: replacement if path == REPORT else original(path))
    monkeypatch.setattr(checker, "REPORT_SHA", hashlib.sha256(replacement).hexdigest())
    with pytest.raises(ValueError, match="source rollout binding"):
        checker.check(REPORT)
