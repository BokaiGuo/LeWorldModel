"""RAM-only arithmetic/source contracts; no torch model, files or prepare run."""
from copy import deepcopy
from types import SimpleNamespace
import gzip
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_kinodynamic_baseline_cuda_20261027_v1 as m


def task():
    t=dict(goal=[.25]*192,action_mean=m.blind.d2p.ACTION_MEAN.astype(float).tolist(),
        action_std=m.blind.d2p.ACTION_STD.astype(float).tolist())
    t['root_context']=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=[[x]*192 for x in (-.5,-.25,0.)],
        action_history=m.core.normalized_tokens(t,m.moving_plan.fixed_raw()).tolist())
    return t


class ArrayTensor:
    """Test-only array wrapper, NOT a CPU/GPU model or torch tensor."""
    def __init__(self,array):self.array=np.asarray(array,np.float32);self.shape=self.array.shape;self.device=SimpleNamespace(type='cuda')
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.array


class ArrayBackend:
    def __init__(self,t,meter,fail=None):self.t,self.meter,self.calls,self.fail=t,meter,0,fail
    def step(self,before,raw):
        self.calls+=1;want=m.moving_plan.incoming(self.t,before,raw)
        for resource,array in (('action_encoder',[want['action_history']]),('predict',[want['latent_history']])):
            self.meter.begin(resource,(ArrayTensor(array),))
            if self.calls==self.fail and resource=='predict':raise ValueError('mock failure after native predictor attempt')
            self.meter.end(resource,None)
        # Explicit fixture arithmetic, no neural operations. Depend on both
        # actual history tokens so a zero-history substitution is observable.
        z=np.asarray(before['latent_history'][-1],np.float32)
        h=np.asarray(before['action_history'],np.float32)
        last=(z+np.float32(np.asarray(raw,np.float32).sum()*.03+h.sum()*.001)).tolist()
        out=ArrayTensor(np.asarray([last,last,last],np.float32))
        self.meter.begin('pred_projector',(out,));self.meter.end('pred_projector',out)
        return m.roll(self.t,before,raw,last)
    def replay(self,before,raw):
        c=deepcopy(before)
        for p in np.asarray(raw).reshape(-1,5,2):c=self.step(c,p.tolist())
        return c


def fixture(slot=0,fail=None):
    t=task();events=[];meter=m.NativeMeter(t,events.append,lambda:None,lambda:True)
    backend=ArrayBackend(t,meter,fail);dispatch=m.Dispatch(t,backend,meter,events.append)
    return t,events,meter,backend,dispatch


@pytest.mark.parametrize('slot',[0,1])
def test_full_schedule_and_three_replays(slot):
    t,events,meter,backend,dispatch=fixture(slot)
    result=m.beam(t,m.bank(slot),dispatch)
    assert backend.calls==84 and dispatch.index==83 and m.counts_ok(meter.counts)
    attempts=[x['call'] for x in events if x['kind']=='dispatch_attempt']
    assert [x['phase'] for x in attempts].count('planning_H1')==16
    assert [x['phase'] for x in attempts].count('planning_H2')==64
    assert [x['path'] for x in attempts[:16]]==[[j] for j in range(16)]
    assert [x['path'] for x in attempts[16:80]]==[p+[j] for p in result['retained_paths'] for j in range(16)]
    assert result['search_terminal']==result['fresh_segmented_terminal']==result['fresh_whole_terminal']
    saved=m.SavedDispatch(t,events);assert m.beam(t,m.bank(slot),saved)==result;saved.done()
    assert result['H1_denominator']==16 and result['H2_denominator']==64
    assert len(result['commands'])==10 and len(result['path'])==2
    # A real complete two-root journal is far below bounded decode/output caps.
    raw=b''.join(m.canonical(dict(source_slot=slot,task_id='a'*64,event=e)) for e in events)
    assert len(raw)*2<32*1024**2 and len(gzip.compress(raw))*2<7*1024**2


def test_stable_terminal_only_no_H1_winner_padding():
    t=task();calls=[]
    def dispatch(before,raw,phase,path):
        calls.append((phase,path));out=deepcopy(before)
        # H1 exact goal is lower than ANY H2. A cross-horizon selector would
        # return a length1 path; this contract MUST still choose length2.
        out['latent_history'][-1]=[.25 if phase=='planning_H1' else 4.]*192
        if phase=='fresh_segmented' and len(path)==1:out['latent_history'][-1]=[.25]*192
        return out
    r=m.beam(t,m.bank(0),dispatch)
    assert r['path']==[0,0] and r['first_costs']==[0.]*16 and r['terminal_cost']>0
    assert r['retained_paths']==[[0],[1],[2],[3]] and len(calls)==83


def test_nonzero_original_and_rolled_histories():
    t,events,meter,backend,dispatch=fixture();m.beam(t,m.bank(0),dispatch)
    calls=[e['call'] for e in events if e['kind']=='dispatch_attempt']
    root=t['root_context'];assert root['action_history'][0]!=root['action_history'][1]
    assert all(x['before']==root for x in calls[:16])
    mid=calls[16]['before'];parent=m.bank(0)[calls[16]['path'][0]]
    assert mid['latent_history'][:2]==root['latent_history'][1:]
    assert mid['action_history'][0]==root['action_history'][1]
    assert mid['action_history'][1]==m.core.normalized_tokens(t,parent).reshape(10).tolist()
    assert calls[-1]['before']==root and np.asarray(calls[-1]['raw']).shape==(10,2)


@pytest.mark.parametrize('slot',[0,1])
def test_old_bank_preserved_exactly(slot):
    # Read only the saved declaration, never old physical outcomes. Runtime
    # uses literals and SHA only; no old context/identity input is passed.
    assert m.sha(m.BANK_SOURCE)==m.BANK_SOURCE_SHA
    saved=m.read(m.BANK_SOURCE)['prepared'][slot]['bank']
    assert saved==m.bank(slot)
    b=np.asarray(saved);assert np.array_equal(b,b.astype(np.float32))
    assert np.array_equal(b[:,0],b[:,4]) and not b[0].any()
    assert np.any(np.linalg.norm(np.arctanh(b).reshape(16,10),axis=1)>.75*np.sqrt(5))


def test_contract_budget_and_claims():
    b=m.budget();assert b['predict']==b['action_encoder']==b['pred_projector']==84
    assert b['planning_predict']==80 and b['AD']==b['encoder']==b['physics']==0
    c=m.contract();assert c['total_budget']['predict']==168
    assert c['P1_admission'] is c['P2_admission'] is c['independent_confirmation'] is False
    assert c['official_reproduction'] is c['equal_search_domain'] is c['equal_total_compute'] is False
    assert c['output_cap_bytes']==8*1024**2 and c['reserve_bytes']==200*1024**2
    assert m.bank(0)!=m.bank(1)


@pytest.mark.parametrize('bad',[-1,2,True,0.,'0'])
def test_bank_slot_strict(bad):
    with pytest.raises(ValueError):m.bank(bad)


@pytest.mark.parametrize('edit',['drop','duplicate','wrong_context','wrong_actions','wrong_successor','extra','partial','wrong_phase'])
def test_saved_receipts_tamper_rejected(edit):
    t,events,meter,backend,dispatch=fixture();result=m.beam(t,m.bank(0),dispatch);e=deepcopy(events)
    if edit=='drop':e.pop(2)
    elif edit=='duplicate':e.insert(2,deepcopy(e[1]))
    elif edit=='wrong_context':e[3]['entry']['value'][0][0]+=1
    elif edit=='wrong_actions':e[1]['entry']['value'][0][0]+=1
    elif edit=='wrong_successor':e[6]['entry']['value'][0]+=1
    elif edit=='extra':e.append(deepcopy(e[-1]))
    elif edit=='partial':e=e[:-2]
    else:e[0]['call']['phase']='fresh_whole'
    s=m.SavedDispatch(t,e)
    with pytest.raises((ValueError,KeyError)):
        m.beam(t,m.bank(0),s);s.done()
    assert result['H2_denominator']==64


def test_failure_attempt_counted_no_retry():
    t,events,meter,backend,dispatch=fixture(fail=3)
    with pytest.raises(ValueError,match='mock failure'):m.beam(t,m.bank(0),dispatch)
    assert meter.counts['predict']==dict(attempted=3,completed=2)
    assert meter.counts['action_encoder']==dict(attempted=3,completed=3)
    assert meter.counts['pred_projector']==dict(attempted=2,completed=2)
    assert events[-1]['kind']=='dispatch_error'
    with pytest.raises(ValueError,match='no retry'):dispatch(t['root_context'],m.bank(0)[0],'planning_H1',[0])
    assert backend.calls==3


def test_native_predispatch_cap_and_no_grad_guard():
    t,events,meter,backend,dispatch=fixture();m.beam(t,m.bank(0),dispatch)
    meter.start(t['root_context'],m.bank(0)[0])
    with pytest.raises(ValueError,match='84'):meter.begin('action_encoder',(ArrayTensor([np.zeros((3,10))]),))
    t,events,meter,backend,dispatch=fixture();meter.no_grad=lambda:False;meter.start(t['root_context'],m.bank(0)[0])
    with pytest.raises(ValueError,match='no-grad'):meter.begin('action_encoder',(None,))


@pytest.mark.parametrize('edit',['context_extra','nan_bank','short_bank','long_packet','out_of_bounds'])
def test_input_scope_and_bank_shape_fail_before_dispatch(edit):
    t=task();b=m.bank(0);called=[]
    if edit=='context_extra':t['root_context']['witness']='not permitted'
    elif edit=='nan_bank':b[0][0][0]=float('nan')
    elif edit=='short_bank':b.pop()
    elif edit=='long_packet':b[0]*=2
    else:b[0][0][0]=1.001
    with pytest.raises(ValueError):m.beam(t,b,lambda *x:called.append(x))
    assert called==[]


def test_moving_task_loader_saved_only_actual_exports():
    # Legitimate nonzero loader, never old blind zero-history schema.
    for slot,(p,h) in enumerate(zip(m.moving_plan.EXPORTS,m.moving_plan.EXPORT_SHAS,strict=True)):
        t=m.moving_plan.load_export(p,h)['gn_task']
        assert t['root_raw_action_history']==m.moving_plan.fixed_raw().astype(float).tolist()
        assert t['root_context']['action_history']==m.core.normalized_tokens(t,m.moving_plan.fixed_raw()).tolist()
        assert len(t['root_context']['latent_history'])==3


def test_cuda_only_wrapper_no_old_material_or_global_replacement():
    import ast
    source=m.Path(m.__file__).read_text();tree=ast.parse(source)
    attrs=[x.attr for x in ast.walk(tree) if isinstance(x,ast.Attribute)]
    assert 'material' not in attrs and 'compile_arm' not in attrs and 'search_programs' not in attrs
    assert "if __name__=='__main__':main()" in source
    assert "require_cuda_allocation('cuda:0')" in source and 'SDPBackend.MATH' in source
    assert not any(isinstance(x,(ast.Assign,ast.AugAssign)) and any(isinstance(y,ast.Attribute) for y in (x.targets if isinstance(x,ast.Assign) else [x.target])) and
        any(isinstance(y,ast.Attribute) and isinstance(y.value,ast.Name) and y.value.id in ('blind','moving','matched','moving_plan') for y in (x.targets if isinstance(x,ast.Assign) else [x.target])) for x in ast.walk(tree))


def test_counts_reject_missing_excess_or_failed():
    good={r:dict(attempted=84,completed=84) for r in ('predict','action_encoder','pred_projector')}
    assert m.counts_ok(good)
    for r in good:
        bad=deepcopy(good);bad[r]['completed']-=1;assert not m.counts_ok(bad)


def test_whole_path_divergence_is_fatal_not_tolerance_or_retry():
    t,events,meter,backend,dispatch=fixture()
    def changed(before,raw,phase,path):
        out=dispatch(before,raw,phase,path)
        if phase=='fresh_whole':out['latent_history'][-1][0]+=1e-8
        return out
    with pytest.raises(ValueError,match='THREE'):m.beam(t,m.bank(0),changed)
    assert backend.calls==84


def consumer_fixture(monkeypatch,edit=None):
    """Complete own-consume filesystem simulated in RAM, never opens output."""
    output=m.Path('/__ram_only_kinodynamic_fixture__');events=[];slots=[];rows=[];results={}
    for s in (0,1):
        t,e,meter,backend,dispatch=fixture(s);result=m.beam(t,m.bank(s),dispatch)
        slot=dict(source_slot=s,task_export=f'/approved/task_{s}.json.gz',task_sha256=str(s+1)*64,
            task_id=str(s+3)*64,blind_sha256=str(s+5)*64);slots.append(slot)
        name=f'root_{s}.result.json.gz';results[str(output/name)]=result
        rows.append(dict(**slot,arm=m.ARM,status='model_plan_sealed',resources=meter.counts,
            result_file=name,commands=result['commands'],path=result['path'],terminal_cost=result['terminal_cost']))
        events.extend(dict(source_slot=s,task_id=slot['task_id'],event=x) for x in e)
    p=dict(slots=slots,banks=[m.bank(0),m.bank(1)],contract=m.contract(),expected_model_hash='c'*64)
    r=dict(schema=m.NAME+'.report',status='fixed_H2_beam_plans_sealed_not_admitted',errors=[],first_error=None,
        prepared='/approved/prepare.json',prepare_sha256='b'*64,commands_released=True,source_verified=True,
        model_unchanged=True,contract=p['contract'],model_hash_before='c'*64,model_hash_after='c'*64,rows=rows,
        resources=m.total_counts(rows),P1_admission=False,P2_admission=False,training=False,AD=0,new_encoder=0,
        new_projector=0,new_physics=0,new_images=0)
    if edit=='row_command':r['rows'][0]['commands'][0][0]+=.01
    elif edit=='resources':r['rows'][0]['resources']['predict']['completed']=83;r['resources']=m.total_counts(rows)
    elif edit=='missing_root':r['rows'].pop()
    elif edit=='partial':r['status']='partial_or_failed'
    elif edit=='model':r['model_hash_after']='d'*64
    elif edit=='admission':r['P1_admission']=True
    elif edit=='native_input':events[1]['event']['entry']['value'][0][0]+=1
    elif edit=='tail':events.pop()
    plain=b''.join(m.canonical(e) for e in events);raw=gzip.compress(plain,mtime=0)
    if edit=='gzip_partial':raw=raw[:-8]
    elif edit=='gzip_trailing':raw+=gzip.compress(b'\n')
    binding=dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256'])
    values={str(output/'binding.json'):binding,**results}
    hashes={m.Path(k).name:m.core.digest(v) for k,v in values.items()};hashes['events.jsonl.gz']=m.blind.encode.digest(raw)
    r['artifact_sha256']=deepcopy(hashes);values[str(output/'report.json')]=r
    report_sha=m.core.digest(r)
    hashes['report.json']=report_sha
    monkeypatch.setattr(m,'read',lambda path,*a:deepcopy(values[str(path)]))
    monkeypatch.setattr(m,'sha',lambda path:hashes[m.Path(path).name])
    monkeypatch.setattr(m,'verify',lambda path,h:deepcopy(p))
    monkeypatch.setattr(m.moving_plan,'load_export',lambda path,h:dict(gn_task=task()))
    monkeypatch.setattr(m.Path,'iterdir',lambda path:iter(output/name for name in hashes))
    monkeypatch.setattr(m.Path,'read_bytes',lambda path:raw)
    return output,report_sha


def test_own_consume_complete_RAM(monkeypatch):
    output,h=consumer_fixture(monkeypatch);c=m.consume(output,h)
    assert c['status']=='fixed_H2_commands_verified_not_admitted' and c['errors']==[]
    assert [x['source_slot'] for x in c['plans']]==[0,1]
    assert all(np.asarray(x['commands']).shape==(10,2) and len(x['path'])==2 for x in c['plans'])
    assert c['P1_admission'] is c['P2_admission'] is c['equal_total_compute'] is False


@pytest.mark.parametrize('edit',['row_command','resources','missing_root','partial','model','admission','native_input','tail','gzip_partial','gzip_trailing'])
def test_own_consume_failure_no_release_RAM(monkeypatch,edit):
    output,h=consumer_fixture(monkeypatch,edit)
    with pytest.raises((ValueError,KeyError)):m.consume(output,h)


@pytest.mark.parametrize('case',['cap','floor'])
def test_prepare_rejects_cap_floor_without_writing(monkeypatch,case):
    monkeypatch.setattr(m,'build',lambda:dict(data='x'*(256*1024 if case=='cap' else 1)))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='reserve'):m.prepare('/__ram_only_no_file__.json')


def test_verify_reconstruction_RAM(monkeypatch):
    p=dict(schema=m.NAME+'.prepare',value=1);monkeypatch.setattr(m,'read',lambda *a:deepcopy(p))
    monkeypatch.setattr(m,'sha',lambda *a:'a'*64);monkeypatch.setattr(m,'build',lambda:deepcopy(p))
    assert m.verify('/__ram__','a'*64)==p
    with pytest.raises(ValueError,match='external'):m.verify('/__ram__','b'*64)
    monkeypatch.setattr(m,'build',lambda:dict(p,value=2))
    with pytest.raises(ValueError,match='reconstruction'):m.verify('/__ram__','a'*64)


def test_partial_counts_do_not_pad_unstarted_root():
    rows=[dict(resources={r:dict(attempted=3,completed=2) for r in ('action_encoder','predict','pred_projector')}),dict(resources=None)]
    assert m.total_counts(rows)=={r:dict(attempted=3,completed=2) for r in ('action_encoder','predict','pred_projector')}
