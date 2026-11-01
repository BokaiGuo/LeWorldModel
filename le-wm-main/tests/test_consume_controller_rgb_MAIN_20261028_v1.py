"""Persistence-only regressions; no model or native environment."""
import json
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import consume_controller_rgb_MAIN_20261028_v1 as writer


@pytest.fixture
def receipt(tmp_path, monkeypatch):
    (tmp_path / 'results').mkdir()
    monkeypatch.setattr(writer.rgb, 'ROOT', tmp_path)
    result = dict(rows=[dict(root_slot=0, arm='test', actual_tracking_cost=1.,
                            checkpoints=[dict(Q=2., model_Q=3.)], final_metrics=dict(native_success=False),
                            full_trace=['preserve', {'all': [1, 2]}])], scientific_admission=False)
    monkeypatch.setattr(writer.rgb, 'consume', lambda *args: result)
    return result, tmp_path / 'results/agent_experiment1_common_controller_rgb_20261028_v1_consume_MAIN01.json'


def test_persists_complete_receipt(receipt):
    expected, path = receipt
    writer.main()
    assert json.loads(path.read_bytes()) == expected


def test_existing_receipt_untouched(receipt):
    expected, path = receipt
    writer.main()
    previous = path.read_bytes()
    with pytest.raises(ValueError, match='NEW receipt'):
        writer.main()
    assert path.read_bytes() == previous


def test_validation_failure_does_not_write(receipt, monkeypatch):
    _, path = receipt
    def reject(*args):
        raise ValueError('original validator rejected')
    monkeypatch.setattr(writer.rgb, 'consume', reject)
    with pytest.raises(ValueError, match='original validator'):
        writer.main()
    assert not path.exists()


def test_disk_floor_failure_does_not_write(receipt, monkeypatch):
    _, path = receipt
    monkeypatch.setattr(writer.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match='cap/floor'):
        writer.main()
    assert not path.exists()
