import json

import pytest

from experiments.spectral_reachability import consume_child_signal_MAIN_20261028_v1 as wrapper


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(wrapper.producer, 'ROOT', tmp_path)
    folder = tmp_path/'results'
    folder.mkdir()
    path = folder/'consume_child_signal_MAIN_20261028_v1_RAM.json'
    result = dict(status='RAM_full_result_not_scientific_evidence', fixed_denominator=48,
                  budget_refusals_by_mode={}, rows=[dict(index=i, full='unchanged') for i in range(48)])
    return path, result


def test_exact_full_result_and_no_overwrite(tmp_path, monkeypatch):
    path, result = fixture(tmp_path, monkeypatch)
    receipt = wrapper.persist(path, result)
    assert json.loads(path.read_bytes()) == result
    assert receipt['bytes'] == path.stat().st_size
    assert receipt['sha256'] == wrapper.producer.sha(path)
    assert receipt['fixed_denominator'] == 48
    with pytest.raises(ValueError):
        wrapper.persist(path, result)
    assert json.loads(path.read_bytes()) == result


def test_own_engineering_cap_rejects_without_trimming(tmp_path, monkeypatch):
    path, result = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(wrapper, 'RECEIPT_CAP', 10)
    with pytest.raises(ValueError, match='cap'):
        wrapper.persist(path, result)
    assert not path.exists() and len(result['rows']) == 48


def test_symlink_rejected(tmp_path, monkeypatch):
    path, result = fixture(tmp_path, monkeypatch)
    other = path.parent/'other.json'
    path.symlink_to(other)
    with pytest.raises(ValueError):
        wrapper.persist(path, result)
    assert not other.exists()
