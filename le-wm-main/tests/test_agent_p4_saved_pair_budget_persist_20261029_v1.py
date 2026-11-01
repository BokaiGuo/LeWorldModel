import hashlib
import json
import os
import pytest
from experiments.spectral_reachability import agent_p4_saved_pair_budget_persist_20261029_v1 as m


def test_real_build_source_closure():
    x=m.build()
    assert x['analysis']['denominator']['episodes']==8
    assert len(x['source_closure'])==6 and not x['P1_admission']
    assert x['analysis']['pairs'][1]['checkpoints'][4]['R'] is None


def test_exclusive_write_and_verify(tmp_path):
    p=tmp_path/'result.json'; raw=m.base.canonical(m.build()).encode()
    m.write_exclusive(p,raw)
    sha=hashlib.sha256(raw).hexdigest()
    assert m.verify(p,sha)['status']=='exact_saved_analysis_verified'
    with pytest.raises(FileExistsError): m.write_exclusive(p,b'other')
    assert p.read_bytes()==raw
    p.write_bytes(raw.replace(b'"episodes":8',b'"episodes":9'))
    with pytest.raises(ValueError): m.verify(p,sha)
    with pytest.raises(ValueError): m.verify(p,hashlib.sha256(p.read_bytes()).hexdigest())


def test_output_limits_and_symlink(tmp_path, monkeypatch):
    p=tmp_path/'x'; p.symlink_to(tmp_path/'unwritten')
    with pytest.raises(FileExistsError): m.write_exclusive(p,b'x')
    with pytest.raises(ValueError): m.write_exclusive(tmp_path/'too-big', b'x'*(m.CAP+1))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _: type('D',(),{'free':0})())
    with pytest.raises(ValueError): m.write_exclusive(tmp_path/'no-space',b'x')


def test_target_namespace():
    with pytest.raises(ValueError): m.target('/tmp/elsewhere.json')
    with pytest.raises(ValueError): m.target(str(m.ROOT/'results'/'existing.json'))


def test_sealed_source_tampering(monkeypatch):
    original=m.SEALED.copy()
    monkeypatch.setattr(m,'SEALED',{**original,next(iter(original)):'0'*64})
    with pytest.raises(ValueError): m.build()


def test_reject_duplicate_json():
    with pytest.raises(ValueError): m.base.decode('{"x":1,"x":2}')
