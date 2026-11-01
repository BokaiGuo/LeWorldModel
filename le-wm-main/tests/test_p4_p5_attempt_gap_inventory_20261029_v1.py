from pathlib import Path
import hashlib
import pytest

from experiments.spectral_reachability import p4_p5_attempt_gap_inventory_20261029_v1 as m


def test_inventory_is_shallow_sorted_and_no_contents(tmp_path):
    (tmp_path/'report.json').write_text('outcome content not parsed')
    (tmp_path/'other').mkdir()
    (tmp_path/'other'/'ledger.json').write_text('not scanned')
    assert m.inventory(tmp_path)==[dict(name='other',kind='directory'),dict(name='report.json',kind='file')]


def test_symlink_rejected(tmp_path):
    (tmp_path/'target').write_text('x')
    (tmp_path/'link').symlink_to(tmp_path/'target')
    with pytest.raises(Exception):m.inventory(tmp_path)


def test_real_bounded_metadata():
    v=m.build()
    assert (v['source_count'],v['dispatch_count'],v['selected_rows'],v['bridge_records'])==(8,10,52,4)
    assert v['original_attempts'] is None and not v['original_load_attested'] and not v['freshness_certified']
    assert len(next(x for x in v['sources'] if x['source_id']=='B')['calls'])==3
    assert all(x['scope'].startswith('prior saved bridge') for x in v['known_root_bridges'])


def test_drift_is_detected(tmp_path,monkeypatch):
    folder=tmp_path/'source';folder.mkdir();(folder/'report.json').write_text('x')
    first=m.inventory(folder)
    (folder/'new-ledger.json').write_text('x')
    assert m.inventory(folder)!=first


def test_noncanonical_inventory_path(tmp_path):
    with pytest.raises(Exception):m.inventory(Path('relative'))


def test_verify_rejects_changed_reconstruction(tmp_path,monkeypatch):
    p=tmp_path/'inventory.json'; data=m.raw({'entries': ['report.json']}); p.write_bytes(data)
    sha=hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(m,'build',lambda:{'entries':['report.json','ledger.json']})
    with pytest.raises(Exception):m.verify(str(p),sha)
