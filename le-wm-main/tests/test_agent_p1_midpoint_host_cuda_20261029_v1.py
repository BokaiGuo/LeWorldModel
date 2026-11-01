from copy import deepcopy
import time
import pytest
from experiments.spectral_reachability import agent_p1_midpoint_host_cuda_20261029_v1 as m
import test_agent_nonlinear_suffix_observed_recompile_20261028_v2 as old

inputs=old.inputs
complete=old.complete


@pytest.fixture(scope='module')
def saved(complete):
    spec=old.spec_for(complete)
    journal=old.n.parent.Events()
    ledger=m.core.Ledger(m.F.caps(26,8),sink=journal.event)
    result=m.method.compile_slot(spec,old.RAM(spec['request']['task'],journal,ledger),journal)
    return spec,dict(result=result,resources=ledger.receipt()),journal.events


def test_full_saved_26_forward_native_and_original_link(saved):
    spec,row,events=deepcopy(saved)
    out=m.audit_row(spec,row,events)
    assert out['midpoint_count']==8
    assert row['resources']['attempted']==row['resources']['completed']==m.F.caps(26,8)
    assert len([x for x in events if x['kind']=='forward_started'])==26


@pytest.mark.parametrize('change',['midpoint_Q','midpoint_raw','selection','AD','plan','drop','unknown'])
def test_saved_mutations_rejected(saved,change):
    spec,row,events=deepcopy(saved);r=row['result']
    if change=='midpoint_Q':r['midpoint_rows'][0]['midpoint_Q']+=1
    elif change=='midpoint_raw':r['midpoint_rows'][0]['midpoint_commands'][0][0]+=.1
    elif change=='selection':r['selected_iteration']=99
    elif change=='AD':next(e for e in events if e['kind']=='AD_API_attempt')['scalar_outputs']=True
    elif change=='plan':r['original']['plan']['parent_plan_sha256']='0'*64
    elif change=='drop':events.remove(next(e for e in events if e['kind']=='program_packet'))
    else:events.insert(1,dict(kind='unrecognized'))
    with pytest.raises((ValueError,StopIteration)):m.audit_row(spec,row,events)


def test_fixed_budget_and_pins():
    c=m.contract()
    assert c['per_slot']==m.F.caps(26,8) and c['total']==m.F.caps(104,32)
    assert c['fixed_denominator']==4 and c['stop_on_first_failure']
    for module,h in m.PINS:assert m.sha(module.__file__)==h
    assert all(r['status']=='not_started' for r in m.roster())


def test_authorization_precedes_io(tmp_path):
    with pytest.raises(ValueError):m.run('nonexistent','bad',str(tmp_path/'out'),'wrong')


@pytest.mark.parametrize('fail',[False,True])
def test_disk_four_slots_full_consume_and_partial(complete,tmp_path,monkeypatch,fail):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    specs=[old.spec_for(complete,i) for i in range(4)]
    parent=dict(rows=specs,expected_model_hash='d'*64,inherited_resources={})
    p=dict(original_prepare=parent,production_ready=True,bindings={})
    rows=m.roster();store=m.Store(tmp_path/'results'/(m.NAME+'_RAM'),time.monotonic());n=0
    def factory(task,journal,ledger):
        nonlocal n
        n+=1
        if fail and n==2:raise RuntimeError('injected constructor failure')
        return old.RAM(task,journal,ledger)
    if fail:
        with pytest.raises(RuntimeError):m.dispatch(p,store,rows,lambda:None,factory)
        assert [r['status'] for r in rows]==['sealed','failed_partial','not_started','not_started']
    else:
        m.dispatch(p,store,rows,lambda:None,factory)
        assert all(r['status']=='sealed' for r in rows)
    for spec,row in zip(specs,rows):
        if row['resources'] is None:continue
        events=m.F.load_journal(store.path,row);m.transport.audit_resources(row['resources'],events)
        if row['status']=='sealed':m.audit_row(spec,row,events)
    prepared=tmp_path/'RAM_prepare.json';prepared.write_bytes(m.raw(p));h=m.sha(prepared)
    store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=h))
    monkeypatch.setattr(m,'verify',lambda path,expected:p if str(path)==str(prepared) and expected==h else m.require(False,'fixture identity'))
    precision=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
    artifacts={str(f.relative_to(store.path)):m.sha(f) for f in store.path.rglob('*') if f.is_file()}
    report=dict(schema=m.NAME+'.report',status='partial_failclosed' if fail else 'complete_model_only',contract=m.contract(),
        prepared=str(prepared),prepare_sha256=h,rows=rows,errors=['injected'] if fail else [],fixed_denominator=4,
        source_verified=True,source_manifest_sha256=m.digest({}),model_before=dict(original_state_dict='d'*64),
        model_after=dict(original_state_dict='d'*64),host=None,precision_before=precision,precision_after=precision,
        CUDA_peak_bytes=[0,0],wall_seconds=1.,artifact_sha256=artifacts,P1_admission=False,freshness=False,physical_release=False)
    store.put('report.json',report,final=True)
    receipt=m.consume(store.path,m.sha(store.path/'report.json'))
    assert receipt['complete'] is (not fail) and len(receipt['rows'])==4
    assert receipt['actual_new_compute']['completed']['predict_API']==(26 if fail else 104)
    (store.path/'unexpected').write_bytes(b'x')
    with pytest.raises(ValueError):m.consume(store.path,m.sha(store.path/'report.json'))
