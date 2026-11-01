"""RAM-only dispatch and saved authority seams; no torch or native environment."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_commitment_controls_20261029_v1 as m
import agent_p3_fullcommit_live_20261028_v1_tests as fixtures


def request(mode='fixedH1',boundary=5,commitment=None):
    base,old=fixtures.request()
    req=m.make_request(base,old['task']['root_context'],slot=0,mode=mode,
        boundary=boundary,raw10=old['raw10'],observation={**old['observation'],'boundary':boundary},
        commitment=commitment)
    return base,req


def planned(mode='fixedH1',boundary=5,commitment=None,backend=fixtures.RamBackend):
    base,req=request(mode,boundary,commitment);store=fixtures.Events()
    ledger=m.core.Ledger(m.caps(mode),sink=store.event)
    result=m.plan_node(req,base,lambda t,s:backend(t,s),store,ledger,permit_ram=True)
    return base,req,result,store


@pytest.fixture(scope='module')
def h3():return planned('fixedH3')


@pytest.mark.parametrize('mode,h,n', [('fixedH1',1,4),('fixedH3',3,8),('adaptive',3,4)])
def test_real_lower_dispatch(mode,h,n):
    _,req,r,_=planned(mode)
    assert r['status']=='sealed' and len(r['commands'])==h*5
    assert len(r['candidates'])==n and all(x['status']=='returned' for x in r['candidates'])
    assert r['resource_receipt']['attempted']==m.live.costs(h,n,schedule=True)
    assert r['resource_receipt']['completed']==r['resource_receipt']['attempted']
    assert r['request_sha256']==m.digest(req)
    m.core.audit_ledger(r['resource_receipt'])


def test_complete_denominator_and_notmatched():
    rows=m.roster();assert len(rows)==6 and sum(len(r['nodes']) for r in rows)==18
    assert [(r['source_slot'],r['mode']) for r in rows]==[(s,mo) for s in (0,1) for mo in m.MODES]
    assert all(r['actual_control_returned'] is None for r in rows)
    c=m.contract();assert c['budget_match']=='notmatched'
    assert c['independent_confirmation'] is c['freshness'] is c['P3_admission'] is False
    a=m.accounting();assert [r['full_completion_predict'] for r in a['per_episode']]==[51,75,None]
    assert a['physical']['actual_physics'] is None


def test_h3_tail_refused_not_truncated():
    _,_,r,_=planned('fixedH3',boundary=10)
    assert r['status']=='refused' and r['refusal']=='fullH_does_not_fit'
    assert r['commands'] is None and not r['candidates']
    assert r['resource_receipt']['attempted']==m.live.costs(schedule=True)


def test_pending_pays_full_signal_without_schedule():
    pending=dict(start=5,end=20,seal_sha256='e'*64)
    _,_,r,_=planned('fixedH3',boundary=10,commitment=pending)
    assert r['status']=='observed_commitment' and r['commands'] is None
    assert r['resource_receipt']['attempted']==m.live.costs()


def test_commitment_exact_full15_no_replacement(h3):
    _,req,result,_=h3;c=m.Commitment(req,result)
    for b in (5,10,15):
        p=c.packet(b)
        with pytest.raises(ValueError):c.packet(b)
        assert p.tobytes()==np.asarray(result['commands'],np.float32)[b-5:b].tobytes()
        c.acknowledge(p)
        active=c.active(b+5)
        assert (active is None)==(b==15)
        if active:assert active==dict(start=5,end=20,seal_sha256=c.seal)
    with pytest.raises(ValueError):c.packet(20)


def test_native_early_stop_retains_short_denominator(h3):
    _,req,result,_=h3;c=m.Commitment(req,result);p=c.packet(5)
    with pytest.raises(ValueError):c.acknowledge(p[:2])
    c.acknowledge(p[:2],terminated=True)
    assert c.returned==2 and c.active(7) is None
    with pytest.raises(ValueError):c.packet(7)


def test_wrong_actual_action_rejected(h3):
    _,req,result,_=h3;c=m.Commitment(req,result);p=c.packet(5);p[0,0]+=.1
    with pytest.raises(ValueError):c.acknowledge(p)


@pytest.mark.parametrize('field', ['base_task_sha256','raw10','observation','mode','source_slot','goal'])
def test_request_identity_mutations(field):
    base,req=request()
    if field=='goal':req['task']['goal'][0]+=.1
    elif field=='raw10':req['raw10'][0][0]+=.1
    elif field=='observation':req[field]['observed_latent_sha256']='0'*64
    elif field=='source_slot':req[field]=True
    elif field=='mode':req[field]='oracle'
    else:req[field]='0'*64
    with pytest.raises(ValueError):m.validate_request(req,base)


def test_ram_not_silent_cpu_fallback():
    base,req=request();store=fixtures.Events();ledger=m.core.Ledger(m.caps('fixedH1'),sink=store.event)
    with pytest.raises(ValueError):
        m.plan_node(req,base,lambda t,s:fixtures.RamBackend(t,s),store,ledger)
    assert not ledger.events


def test_failure_nonrefundable_and_preserved():
    class Broken(fixtures.RamBackend):
        def replay(self,*args):raise RuntimeError('explicit RAM failure')
    base,req=request();store=fixtures.Events();ledger=m.core.Ledger(m.caps('fixedH1'),sink=store.event)
    with pytest.raises(RuntimeError):
        m.plan_node(req,base,lambda t,s:Broken(t,s),store,ledger,permit_ram=True)
    out=store.events[-1]['result'];rr=out['resource_receipt']
    assert out['status']=='failed' and len(out['candidates'])==4
    assert rr['attempted']['replay_calls']==1 and rr['completed']['replay_calls']==0
    remain=m.remaining_caps('fixedH1',[out])
    assert remain['replay_calls']==m.caps('fixedH1')['replay_calls']-1


def test_build_uses_only_task_inputs_not_outcomes(monkeypatch,tmp_path):
    # Synthetic metadata authority seam, explicitly not a sealed scientific input.
    base,req=request();fake=dict(sources={},tasks=[base,deepcopy(base)],layers=[dict(commands=[[0.,0.]]*5)],
        policy={'frozen':True},modelbinding={'immutable':'fixture'},expected_model_hash='1'*64,source_manifest='fixture')
    monkeypatch.setattr(m,'upstream',lambda:deepcopy(fake))
    files=[]
    for i in range(6):
        p=tmp_path/str(i);p.write_bytes(str(i).encode());files.append(p)
    monkeypatch.setattr(m,'own_files',lambda:files[:3])
    monkeypatch.setattr(m,'REPORT',files[3]);monkeypatch.setattr(m,'STRICT',files[4]);monkeypatch.setattr(m,'PREPARED',files[5])
    first=m.build();assert first['tasks']==fake['tasks'] and first['roster']==m.roster()
    monkeypatch.setattr(m,'build',lambda:first)
    path=tmp_path/'prepared.json';path.write_bytes(m.core.canonical(first))
    assert m.verify(path,m.sha(path))==first
    path.write_bytes(b'{}')
    with pytest.raises(ValueError):m.verify(path,'1'*64)


def test_authority_wrong_sha_before_reconstruction(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda p:'0'*64)
    monkeypatch.setattr(m.live,'consume',lambda *a:pytest.fail('must reject before old consume'))
    with pytest.raises(ValueError):m.upstream()


def test_strict_receipt_not_status_only(monkeypatch):
    hashes={m.REPORT:m.REPORT_SHA,m.STRICT:m.STRICT_SHA,m.PREPARED:m.PREPARE_SHA}
    monkeypatch.setattr(m,'sha',lambda p:hashes[p])
    monkeypatch.setattr(m.live,'consume',lambda *a:dict(complete=True,counts={'actual':7}))
    monkeypatch.setattr(m.signal,'read',lambda p:dict(complete=True,counts={'actual':8}))
    with pytest.raises(ValueError):m.upstream()


def test_source_drift_rejected_by_prepare_builder(monkeypatch,tmp_path):
    source=tmp_path/'sealed.py';source.write_bytes(b'old')
    monkeypatch.setattr(m,'upstream',lambda:dict(sources={str(source):m.sha(source)}))
    # Freeze fixture source digest before mutation; own metadata files may be these RAM files.
    expected=m.sha(source)
    monkeypatch.setattr(m,'upstream',lambda:dict(sources={str(source):expected}))
    monkeypatch.setattr(m,'own_files',lambda:[])
    monkeypatch.setattr(m,'REPORT',source);monkeypatch.setattr(m,'STRICT',source)
    monkeypatch.setattr(m,'PREPARED',source);source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source closure'):m.build()


def test_no_live_launch_cli():
    with pytest.raises(SystemExit):m.main(['run'])


def test_accounting_cli(capsys):
    m.main(['accounting']);assert 'notmatched' in capsys.readouterr().out
