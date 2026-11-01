import json
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_p2_stagec_composition_saved_receipt_20261028_v1 as m

def fake_result():
    return dict(schema='agent_p2_stagec_composition_saved_consume_20261028_v1.consume',
        status='saved_contract_verified_not_admitted',composition_benefit=None,
        P1_admission=False,P2_admission=False,branch_denominator=8,artifact_count=116)

def test_writer_persists_full_boundary(monkeypatch,tmp_path):
    monkeypatch.setattr(m.consumer,'consume',lambda *args:fake_result())
    out=tmp_path/'receipt.json';r=m.write_receipt('/saved','r'*64,'p'*64,'f'*64,out)
    data=json.loads(out.read_text());assert r['status']=='saved_receipt_persisted'
    assert data['consumer_result']==fake_result() and data['composition_benefit'] is None
    assert data['P1_admission'] is data['P2_admission'] is False

def test_writer_never_overwrites(monkeypatch,tmp_path):
    monkeypatch.setattr(m.consumer,'consume',lambda *args:fake_result())
    out=tmp_path/'receipt.json';out.write_text('sealed')
    with pytest.raises(ValueError):m.write_receipt('/saved','r'*64,'p'*64,'f'*64,out)

def test_writer_rejects_science_boundary_mutation(monkeypatch,tmp_path):
    bad=fake_result();bad['composition_benefit']=0.5
    monkeypatch.setattr(m.consumer,'consume',lambda *args:bad)
    with pytest.raises(ValueError):m.write_receipt('/saved','r'*64,'p'*64,'f'*64,tmp_path/'r.json')

def test_writer_rejects_admission(monkeypatch,tmp_path):
    bad=fake_result();bad['P2_admission']=True
    monkeypatch.setattr(m.consumer,'consume',lambda *args:bad)
    with pytest.raises(ValueError):m.write_receipt('/saved','r'*64,'p'*64,'f'*64,tmp_path/'r.json')

def test_cli_contract_is_exclusive():
    assert m.CAP==2*1024**2 and m.CONSUMER_SHA=='9255f3a34149776305b8503b3088251c6461095337eda3957555688f0c469fa2'
