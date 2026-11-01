"""RAM interleaving and real saved fullarray regressions; no model calls."""
from copy import deepcopy
from pathlib import Path
import importlib
import sys
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import agent_p3_child_budget_host_cuda_20261028_v1 as m


@pytest.fixture(scope='module')
def saved():
    p=m.base.read(m.ROOT/'results/agent_p3_child_program_cuda_20261028_v1_prepare_MAIN01.json',64*1024**2)
    signals=m.base.read(m.compiler.SIGNAL_RECEIPT,256*1024**2)
    compiled=m.base.read(m.budget.STRICT,128*1024**2)
    wiring=dict(children=p['children'],tasks=p['tasks'],limits=m.budget.inputs_template())
    return dict(wiring=wiring,sources={},profile=m.profile(),expected_model_hash=p['expected_model_hash']),signals,compiled


def signal_groups():
    group=None;trace=[]
    for event,_ in m.paid.iter_journal(m.compiler.SIGNAL_OUTPUT/m.paid.JOURNAL):
        if event['kind']!='child_trace':continue
        if group is not None and group!=event['group']:
            yield group,trace;trace=[]
        group=event['group'];trace.append(event['payload'])
    if group is not None:yield group,trace


def program_groups():
    group=None;trace=[]
    for event,_ in m.paid.iter_journal(m.budget.REPORT.parent/m.paid.JOURNAL):
        if event['kind']!='compiler_trace':continue
        if group is not None and group!=event['group']:
            yield group,trace;trace=[]
        group=event['group'];trace.append(event['payload'])
    if group is not None:yield group,trace


def replay_operations(sink, receipt):
    """Only RAM native attempt/return records, with ORIGINAL saved nesting."""
    sequence=[]
    for op in receipt['events']:
        sequence.append((op['started_ns'],0,dict(kind='resource_attempt',entry=dict(op,status='attempted',ended_ns=None))))
        sequence.append((op['ended_ns'],1,dict(kind='resource_return',entry=op)))
    for _,_,e in sorted(sequence,key=lambda x:x[:2]):sink.event(e)


@pytest.fixture(scope='module')
def structural_panel(saved):
    p,signals,compiled=saved
    events=[];position=0
    def emit(e):events.append(dict(kind='work',slot=position,payload=deepcopy(e)))
    session=m.native.Session(p['wiring'],emit)
    before=dict(original_state_dict=p['expected_model_hash'],all_buffers='a'*64)
    events.append(dict(kind='denominator',rows=m.initial_rows(p),fixed_denominator=48,child_denominator=24,
                       sources_sha256=m.digest(p['sources'])))
    events.append(dict(kind='model_before',hashes=before))
    for position in range(48):
        child=p['wiring']['children'][position//2];mode=m.contract.MODES[position%2]
        row=session.rows[position]
        opened=session.gate.open(child['child_id'],mode,child['context_sha256'],child['ancestry']['task_binding_sha256'],
                                observed=child['status']=='observed_child')
        if opened['dispatch_allowed']:
            a=signals['rows'][position//2]['result']['acquisitions'][position%2]
            row.update(signal=deepcopy(a['signal']),signal_resources=deepcopy(a['resource_receipt']),
                decision=deepcopy(signals['rows'][position//2]['result']['rows'][position%2]['decision']))
            replay_operations(m.native.GateSink(session.gate,opened['ticket'],emit),a['resource_receipt'])
            sealed=session.gate.seal(opened['ticket'],row['decision'],row['signal']['scheduling_signal'])
            if sealed['dispatch_allowed']:
                branch=m.branch_for(child,mode,row)
                row['compiler']=deepcopy(compiled['rows'][position])
                row['compiler']['branch_sha256']=m.digest(branch)
                replay_operations(m.native.GateSink(session.gate,opened['ticket'],emit),row['compiler']['resource_receipt'])
                session.gate.finish(opened['ticket']);row['status']='budget_complete_native_unverified'
            else:row['status']=sealed['reason']
        events.append(dict(kind='slot_return',slot=position,row=deepcopy(row)))
    events.extend([dict(kind='model_after',hashes=before),dict(kind='sources_final',sha256=m.digest(p['sources']))])
    report=dict(model_before=before,model_after=before,session=session.receipt(),journal_tail='f'*64)
    return p,report,events


def iter_ram(events):
    for e in events:yield e,'f'*64


@pytest.fixture
def structural_validators(monkeypatch):
    calls=[]
    def signal(task,child,acquisition,events,schedules):
        calls.append(('signal',child['child_id']))
        return m.core.FrozenSignals(**acquisition['signal']['scheduling_signal'])
    def compiler(task,child,branch,row,events):calls.append(('program',child['child_id']))
    monkeypatch.setattr(m.paid,'validate_trace',signal)
    monkeypatch.setattr(m.compiler,'validate_branch',compiler)
    return calls


def test_full48_gate_interleaving_census_and_actual_not_caps(structural_panel,structural_validators):
    p,r,events=structural_panel
    census,actual=m.validate_stream(iter_ram(events),p,r)
    assert census==dict(compiled=41,refused=5,not_reached=2)
    assert actual['fixed']!=actual['adaptive']
    assert actual['fixed']['attempted']['predict_API']==391
    assert actual['adaptive']['attempted']['predict_API']==634
    assert len([c for c in structural_validators if c[0]=='signal'])==46
    assert len([c for c in structural_validators if c[0]=='program'])==41


@pytest.mark.parametrize('tamper',['ungated','late_gate','early_completion','duplicate_return','wrong_slot','dropped_slot','unknown','footer','gate_cap','actual'])
def test_negative_interleaving_failclosed(structural_panel,structural_validators,tamper):
    p,original,source=structural_panel
    # Shallow outer lists with isolated replaced records: avoid copying all RAM
    # events for every negative case; no frozen report or producer is changed.
    events=list(source);r=dict(original)
    i=next(i for i,e in enumerate(events) if e['kind']=='work' and e['payload']['kind']=='resource_attempt')
    if tamper=='ungated':del events[i-1]
    elif tamper=='late_gate':events[i-1],events[i]=events[i],events[i-1]
    elif tamper=='early_completion':
        j=next(j for j,e in enumerate(events) if e['kind']=='work' and e['payload']['kind']=='resource_return')
        events[j],events[j+1]=events[j+1],events[j]
    elif tamper=='duplicate_return':
        j=next(j for j,e in enumerate(events) if e['kind']=='work' and e['payload']['kind']=='resource_return')
        events.insert(j,events[j])
    elif tamper=='wrong_slot':events[i]=dict(events[i],slot=1)
    elif tamper=='dropped_slot':events=[e for e in events if not(e['kind']=='slot_return' and e['slot']==0)]
    elif tamper=='unknown':events.insert(i,dict(kind='fake'))
    elif tamper=='footer':events=events[:-1]
    elif tamper=='gate_cap':
        p=dict(p,wiring=deepcopy(p['wiring']));p['wiring']['limits']['common_node_cap']['predict_API']-=1
    elif tamper=='actual':
        r=dict(r,session=dict(r['session'],gate=deepcopy(r['session']['gate'])))
        r['session']['gate']['actual']['fixed']['attempted']['predict_API']-=1
    with pytest.raises(ValueError):m.validate_stream(iter_ram(events),p,r)


@pytest.fixture(scope='module')
def real_slot(saved):
    p,signals,compiled=saved
    group,st=next(signal_groups());pg,pt=next(program_groups())
    child=p['wiring']['children'][0];mode='fixed';assert group==pg==child['child_id']+':fixed'
    a=signals['rows'][0]['result']['acquisitions'][0]
    row=dict(m.initial_rows(p)[0],signal=deepcopy(a['signal']),signal_resources=deepcopy(a['resource_receipt']),
             decision=deepcopy(signals['rows'][0]['result']['rows'][0]['decision']),
             compiler=deepcopy(compiled['rows'][0]),status='budget_complete_native_unverified')
    branch=m.branch_for(child,mode,row);row['compiler']['branch_sha256']=m.digest(branch)
    pt=[dict(e,branch_sha256=m.digest(branch)) if e['kind']=='bank' else e for e in pt]
    gate=m.budget.WholeNodeGate(p['wiring']['limits'],p['wiring']['children'],p['wiring']['tasks'])
    opened=gate.open(child['child_id'],mode,child['context_sha256'],child['ancestry']['task_binding_sha256'])
    sink=m.native.GateSink(gate,opened['ticket'],lambda e:None)
    for e in st:sink.event(e)
    gate.seal(opened['ticket'],row['decision'],row['signal']['scheduling_signal'])
    seal=deepcopy(gate.events[-1])
    sink=m.native.GateSink(gate,opened['ticket'],lambda e:None)
    for e in pt:sink.event(e)
    gate.finish(opened['ticket'])
    return p,row,st,pt,gate,seal


def test_real_saved_slot_full_native_signal_ridge_replays_and_gate(real_slot):
    p,row,st,pt,gate,seal=real_slot
    assert m.validate_slot(p,0,row,st,pt,gate,seal)=='compiled'


@pytest.mark.parametrize('tamper',['J','raw','history','goal','decision','commands','selection','io','missing_array'])
def test_real_saved_fullarray_tamper_is_not_a_success(real_slot,tamper):
    p,row,st,pt,gate,seal=real_slot
    row=deepcopy(row);st=list(st);pt=list(pt)
    if tamper=='J':row['signal']['responses'][0]['jacobian'][0][0]+=1.
    elif tamper=='raw':row['compiler']['selected']['raw5'][0][0]+=.01
    elif tamper=='history':
        p=deepcopy(p);p['wiring']['children'][0]['context']['action_history'][0][0]+=1.
    elif tamper=='goal':
        p=deepcopy(p);p['wiring']['tasks'][0]['goal'][0]+=1.
    elif tamper=='decision':row['decision']['candidates']=8
    elif tamper=='commands':row['compiler']['candidates'][0]['actions'][0][0]+=.01
    elif tamper=='selection':row['compiler']['selected']['selection_digest']='0'*64
    elif tamper=='io':
        i=next(i for i,e in enumerate(pt) if e['kind']=='io_input')
        pt[i]=dict(pt[i],operation_index=-1)
    else:st.pop(next(i for i,e in enumerate(st) if e['kind']=='raw_backward'))
    with pytest.raises(ValueError):m.validate_slot(p,0,row,st,pt,gate,seal)


def test_authorization_before_any_verify_or_model_import(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(m,'verify',lambda *a:calls.append('BAD'))
    with pytest.raises(ValueError,match='authorization'):
        m.run('/ABS','0'*64,str(tmp_path/'output'),'NO')
    assert calls==[]


def test_pins_and_actual_upstream_prepare_are_not_future():
    m.pins()
    assert m.sha(m.UPSTREAM)==m.UPSTREAM_SHA


def test_engineering_caps_not_scientific_quota():
    p=m.profile()
    assert p['output_bytes']==512*1024**2 and p['final_reserve_bytes']==256*1024**2
    assert p['wall_seconds']==1800 and p['CPU_neural'] is False
    assert all(v is False for v in m.FLAGS.values())


@pytest.mark.parametrize('tamper',['truncated','concat','tail','hash'])
def test_actual_journal_encoding_negative(tmp_path,tamper):
    import zlib
    payload=dict(index=0,previous='0'*64,payload=dict(kind='RAM_only'))
    payload['sha256']=m.digest(payload)
    if tamper=='hash':payload['sha256']='0'*64
    raw=m.canonical(payload)+b'\n';encoded=zlib.compress(raw,wbits=31)
    if tamper=='truncated':encoded=encoded[:-3]
    elif tamper=='concat':encoded+=encoded
    elif tamper=='tail':encoded+=b'junk'
    path=tmp_path/'events.gz';path.write_bytes(encoded)
    with pytest.raises(ValueError):list(m.paid.iter_journal(path))


def test_store_cap_refuses_before_backend(tmp_path):
    profile=dict(m.profile(),output_bytes=256*1024**2+64,final_reserve_bytes=256*1024**2)
    store=m.paid.Store(tmp_path/'RAM_store',profile,free=lambda:2**40)
    try:
        with pytest.raises(ValueError,match='cap'):store.event(dict(kind='RAM_TOO_BIG',value='x'*4096))
        assert store.index==0 and store.tail=='0'*64
    finally:store.close()


@pytest.mark.parametrize('boundary',['disk','wall','memory'])
def test_preentry_engineering_guard_does_not_advance_actual_journal(tmp_path,boundary):
    now=[0.];free=[2**40]
    store=m.paid.Store(tmp_path/'RAM_guard',m.profile(),clock=lambda:now[0],free=lambda:free[0])
    try:
        if boundary=='disk':free[0]=0
        elif boundary=='wall':now[0]=1801.
        else:store.memory=lambda:(5*1024**3,5*1024**3)
        with pytest.raises(ValueError):store.event(dict(kind='RAM_REFUSED'))
        assert store.index==0 and store.tail=='0'*64
    finally:
        free[0]=2**40
        store.close()


def test_run_disk_reservation_refuses_before_import(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'exclusive',lambda p:Path(p))
    monkeypatch.setattr(m,'verify',lambda *a:dict(profile=m.profile()))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    def no_store(*a,**kw):raise AssertionError('store/model must not start')
    monkeypatch.setattr(m.paid,'Store',no_store)
    with pytest.raises(ValueError,match='reservation'):
        m.run('/RAM','a'*64,str(tmp_path/'RAM_no_output'),m.AUTH)
    assert not (tmp_path/'RAM_no_output').exists()


def test_wrong_native_prepare_identity_refused_before_native_verify(monkeypatch):
    calls=[]
    monkeypatch.setattr(m.native,'verify',lambda *a:calls.append('BAD'))
    with pytest.raises(ValueError,match='fixed actual native prepare'):
        m.material(str(m.UPSTREAM),'0'*64)
    assert calls==[]


def test_consume_requires_external_report_identity_before_producer(monkeypatch,tmp_path):
    calls=[]
    path=tmp_path/'report.json';path.write_bytes(b'{}')
    monkeypatch.setattr(m,'verify',lambda *a:calls.append('BAD'))
    with pytest.raises(ValueError,match='external actual terminal'):
        m.consume(str(tmp_path),'0'*64)
    assert calls==[]


@pytest.mark.parametrize('boundary',['engineering_failed','partial_session','native_flag','missing_artifacts'])
def test_terminal_report_outer_contract_is_failclosed(saved,tmp_path,boundary):
    p,_,_=saved
    r=dict(schema=m.NAME+'.report',status='complete_native_panel_not_admitted',prepared='/RAM',prepare_sha256='a'*64,
        session=dict(schema=m.native.NAME+'.ram_or_unverified_session',rows=m.initial_rows(p),gate={},
            fixed_denominator=48,child_denominator=24,stopped=False,same_cap=True,
            matched_actual_vector=False,native_verified=False,**m.FLAGS),errors=[],model_before=None,model_after=None,
        host=None,CUDA_peak_bytes=None,sources_verified=True,cleanup_ok=True,
        source_manifest_sha256=m.digest(p['sources']),wall_seconds=0.,journal_tail='0'*64,artifact_sha256={},**m.FLAGS)
    if boundary=='engineering_failed':r['status']='engineering_failed';r['errors']=[dict(message='cap')]
    elif boundary=='partial_session':r['session']['rows']=r['session']['rows'][:-1]
    elif boundary=='native_flag':r['session']['native_verified']=True
    with pytest.raises(ValueError):m.validate_report(tmp_path,'a'*64,p,r)
