import hashlib
import pytest
from experiments.spectral_reachability import agent_e3_feedback_durable_ledger_20261029_v1 as m


def saved(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def test_real_disk_nested_failure_no_refund(tmp_path):
    p=tmp_path/'journal';j=m.Ledger(p)
    parent=j.intent('parent',{});child=j.intent('child',{})
    j.failed(child,'child',{'repr':'inner'});j.failed(parent,'parent',{'repr':'outer'});j.close()
    result=m.audit(p,saved(p));assert not result['complete'] and not result['full_native_strict']
    assert result['counts']=={'parent':dict(attempted=1,returned=0,failed=1),'child':dict(attempted=1,returned=0,failed=1)}
    assert p.read_bytes().count(b'\n')==4 and b'\n\n' not in p.read_bytes()


def test_real_disk_pending_and_tamper(tmp_path):
    p=tmp_path/'journal';j=m.Ledger(p);j.intent('public_step',{});j.close()
    h=saved(p);assert not m.audit(p,h)['complete']
    with pytest.raises(ValueError):m.audit(p,'0'*64)
    p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError):m.audit(p,saved(p))


def test_write_cap_fail_before_event(tmp_path):
    p=tmp_path/'journal';j=m.Ledger(p,cap=20)
    with pytest.raises(ValueError):j.intent('public_step',{})
    j.close();assert p.read_bytes()==b''
