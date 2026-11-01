"""Own physical authority + real-sized array/hook RAM tests; no neural/physics."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as h
from experiments.spectral_reachability import agent_terminal_witness_terminal_rgb_20261028_v1 as m


def fixture(missing=()):
    plans=[];actual=[];slots=[];panels=[];modelrows=[]
    q=np.eye(192).tolist()
    for i,(root,condition) in enumerate(m.producer.ORDER):
        goal=np.zeros(192,np.float32).tolist();pred=np.full(192,.5,np.float32).tolist()
        ctx=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=[[0.]*192 for _ in range(3)],action_history=[[0.]*10 for _ in range(2)])
        commands=np.full((10,2),i/16,np.float32).tolist();rawsha=m.hash_raw(commands)
        slot=dict(index=i,root_slot=root,condition=condition,task_id=f'{i+1:064x}',
            task_sha256=f'{i+5:064x}',blind_sha256=f'{i+9:064x}',packet=dict(root_context=ctx,goal=goal),q=q)
        plan=dict(schema=m.producer.NAME+'.execution_plan',index=i,root_slot=root,source_slot=root,condition=condition,
            arm='terminal_witness_mixed222',task_id=slot['task_id'],task_sha256=slot['task_sha256'],blind_sha256=slot['blind_sha256'],
            selected_index=0,commands=commands,raw_float32_sha256=rawsha,raw_steps=10,
            selector_analysis_sha256=m.producer.ANALYSIS_SHA,source_report_sha256=m.producer.mixed.SOURCE_SHA,
            nonlinear_terminal_Q=.25,model_prefix_uphill_from_anchor=False,context_sha256=m.selector.sha_object(ctx),
            goal_sha256=m.selector.sha_object(goal),q_sha256=m.selector.base.digest(q),
            original_stage_plus_ridge_index=1,selection_uses_physics=False,P1_admission=False)
        plan['selection_digest']=m.producer.mixed.core.digest(plan)
        state=[0.]*7;cp=[];trajectory=[]
        for step in ([0,5] if i in missing else [0,5,10]):
            c=dict(step=step,png=f'row{i}_plan_{step}.png',file_sha256='a'*64,pixel_sha256='b'*64,
                state=deepcopy(state),snapshot=dict(RAM_ONLY=True))
            cp.append(c);trajectory.append(dict(deepcopy(c),phase='plan',condition=condition,
                overlay_index=1 if condition=='on' else 0))
        a=dict(index=i,root_slot=root,condition=condition,arm=plan['arm'],task_id=slot['task_id'],
            selection_digest=plan['selection_digest'],plan=deepcopy(plan),status='native_stopped' if i in missing else 'horizon_completed',
            own_returned=9 if i in missing else 10,checkpoints=cp,trajectory=trajectory,
            missing_checkpoints=[10] if i in missing else [],final_state=deepcopy(state),final_metrics=dict(native7_distance=1.),
            native_success=False,terminated=False,truncated=False,initial_exact=True)
        panel=dict(family='mixedmode',index=i,root_slot=root,condition=condition,task_sha256=slot['task_sha256'],
            candidate_denominator=222,terminal_available=222,literal_forward_requests=[],status='complete_model_only',
            context_sha256=plan['context_sha256'],goal_sha256=plan['goal_sha256'],q_sha256=plan['q_sha256'],P1_admission=False,
            nonlinear_terminal_Q=dict(index=0,score=.25,commands=deepcopy(commands),raw_float32_sha256=rawsha,
                model_prefix_uphill_from_anchor=False),
            candidates=[dict(index=j,predicted_terminal_sha256=m.selector.sha_object(pred)) for j in range(222)])
        candidate=dict(index=0,predicted_terminal=deepcopy(pred),nonlinear_terminal_cost=.25,
            commands=deepcopy(commands),context=dict(latent_history=[[],[],deepcopy(pred)]))
        modelrows.append(dict(index=i,result=dict(candidates=[deepcopy(candidate) for _ in range(222)])))
        slots.append(deepcopy(slot));plans.append(deepcopy(plan));actual.append(deepcopy(a));panels.append(deepcopy(panel))
    authority=dict(analysis_sha256=m.producer.ANALYSIS_SHA,plans_sha256=m.producer.mixed.core.digest(plans))
    prep=dict(schema=m.producer.NAME+'.prepare',plans=plans,selector_authority=deepcopy(authority))
    strict=dict(schema=m.producer.NAME+'.consume',status=m.producer.RECORDED,rows=actual,errors=[],
        fixed_denominator=4,known_roots=2,fresh_roots=0,inherited=0,P1_admission=False,scientific_admission=False,
        freshness=False,selector_authority=deepcopy(authority))
    analysis=dict(schema=m.selector.NAME+'.analysis',panels=panels)
    compiled=dict(slots=slots);model=dict(rows=modelrows)
    return prep,strict,analysis,compiled,model


def prepared(missing=()):
    rows=m.fixed_rows(*fixture(missing),'/RAM_ONLY/physical')
    return dict(schema=m.NAME+'.prepare',rows=rows,contract=m.contract(),
        expected_calls={k:4-len(missing) if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})


def test_own_physical_schema_four_exact_terminal_joins():
    p=prepared();assert len(p['rows'])==4
    assert [r['condition'] for r in p['rows']]==['off','on','off','on']
    assert all(r['predicted_Q']==.25 and r['source']['step']==10 for r in p['rows'])
    assert all(Path(r['source']['path']).name==r['source']['png'] for r in p['rows'])


@pytest.mark.parametrize('fault',['old_schema','task','raw','goal','context','selection','checkpoint_early',
    'checkpoint_missing','frame_condition','frame_state','drop_row','prediction','analysis_sha','plan_digest'])
def test_physical_model_joins_reject_tampering(fault):
    p,s,a,c,r=fixture()
    if fault=='old_schema':s['schema']=m.producer.mixed.NAME+'.consume'
    elif fault=='task':p['plans'][0]['task_sha256']='0'*64
    elif fault=='raw':s['rows'][0]['plan']['commands'][0][0]=.9
    elif fault=='goal':c['slots'][0]['packet']['goal'][0]=1.
    elif fault=='context':c['slots'][0]['packet']['root_context']['latent_history'][0][0]=1.
    elif fault=='selection':a['panels'][0]['nonlinear_terminal_Q']['index']=1
    elif fault=='checkpoint_early':s['rows'][0]['own_returned']=9
    elif fault=='checkpoint_missing':s['rows'][0]['missing_checkpoints']=[10]
    elif fault=='frame_condition':s['rows'][0]['trajectory'][-1]['condition']='on'
    elif fault=='frame_state':s['rows'][0]['checkpoints'][-1]['state'][0]=1.
    elif fault=='drop_row':s['rows'].pop()
    elif fault=='prediction':r['rows'][0]['result']['candidates'][0]['predicted_terminal'][0]=2.
    elif fault=='analysis_sha':s['selector_authority']['analysis_sha256']='0'*64
    else:p['plans'][0]['selection_digest']='0'*64
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,s,a,c,r,'/RAM_ONLY/physical')


def execute(monkeypatch,missing=(),forbidden=None,tokens=257):
    p=prepared(missing);rows=deepcopy(p['rows']);store=h.old.Memory()
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    model=h.Model(tokens);torch=h.fake_torch();errors=[]
    trace=m.bridge.Trace(model,torch,store,lambda:None,p['expected_calls'])
    guard=m.bridge.native.Ledger(2,lambda e:store.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    def encode(pix,row,scope,ordinal):
        guard.image=ordinal
        with m.bridge.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(h.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(h.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.encode_guard(model,torch,guard,errors):m.process(p,rows,store,encode)
    return p,dict(schema=m.NAME+'.report',rows=rows,contract=m.contract(),errors=errors,blocked=guard.blocked,
        modules=trace.entries,guards=guard.entries,counts=m.bridge.counts(trace.entries)),store


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)


def test_complete_encoder_projector_native_arrays_four(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    out=m.validate(p,r,s.events,s.load)
    assert len(out)==4 and len(r['modules'])==len(r['guards'])==8
    assert r['counts']['encoder']==dict(attempted=4,completed=4)
    assert all(x['predicted_Q']==.25 and x['actual_Q'] is not None and not x['P1_admission'] for x in out)


@pytest.mark.parametrize('missing',[(0,),(1,3),(0,1,2,3)])
def test_missing_cp10_keeps_four_null_no_padding(monkeypatch,missing):
    p,r,s=execute(monkeypatch,missing);out=m.validate(p,r,s.events,s.load)
    assert len(out)==4 and sum(x['latent_missing'] for x in out)==len(missing)
    assert all(out[i]['actual_Q'] is None for i in missing)


@pytest.mark.parametrize('fault',['event','event_tail','preprocess','pixels','latent','hidden197','plan','guard','count','blocked'])
def test_strict_saved_validation_no_unknown_filter_or_preprocess_shortcut(replay,monkeypatch,fault):
    p,r0,s0=replay;r=deepcopy(r0);events=deepcopy(s0.events);s=h.old.Memory();s.files=dict(s0.files)
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    if fault=='event':events.insert(0,dict(kind='UNKNOWN'))
    elif fault=='event_tail':events.append(dict(kind='UNKNOWN'))
    elif fault=='preprocess':r['rows'][0]['preprocessed']=s.array('badpre',np.zeros((1,1,3,224,224),np.float32))
    elif fault=='pixels':r['rows'][0]['pixels']=s.array('badpixels',np.zeros((224,224,3),np.uint8))
    elif fault=='latent':r['rows'][0]['latent']=s.array('badlatent',np.zeros(192,np.float32))
    elif fault=='hidden197':r['rows'][0]['modules'][0]['output']=s.array('bad197',np.zeros((1,197,192),np.float32))
    elif fault=='plan':r['rows'][0]['goal192'][0]=1.
    elif fault=='guard':r['guards'].pop()
    elif fault=='count':r['counts']['encoder']['completed']-=1
    else:r['blocked']=['predict']
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,events,s.load)


@pytest.mark.parametrize('forbidden',['predict','AD','action_encoder'])
def test_frozen_guard_forbids_non_encoder_computation(monkeypatch,forbidden):
    with pytest.raises(ValueError):execute(monkeypatch,forbidden=forbidden)


def test_cpu_authority_refused_before_io():
    with pytest.raises(ValueError):m.run('/not/read','a'*64,'/not/created',authorization='CPU')


def test_encode_guard_never_replaces_torch_globals_and_restores_profile():
    import sys
    model=h.Model();torch=h.fake_torch();ledger=m.bridge.native.Ledger(1,lambda e:None);errors=[]
    original=(torch.autograd.grad,torch.autograd.backward,torch.autograd.functional.jacobian,
        torch.Tensor.backward,torch.func.jacrev,sys.getprofile(),model.predict,model.encoder.forward)
    with m.encode_guard(model,torch,ledger,errors):
        assert (torch.autograd.grad,torch.autograd.backward,torch.autograd.functional.jacobian,
            torch.Tensor.backward,torch.func.jacrev)==original[:5]
        with pytest.raises(ValueError,match='forbidden'):torch.autograd.functional.jacobian(None)
    assert (torch.autograd.grad,torch.autograd.backward,torch.autograd.functional.jacobian,
        torch.Tensor.backward,torch.func.jacrev,sys.getprofile(),model.predict,model.encoder.forward)==original
    assert errors==[] and ledger.blocked


def test_thread_probe_unwraps_decorators_not_shared_wrapper_code():
    from functools import wraps
    def wrapper(fn):
        @wraps(fn)
        def shared(*a,**kw):return fn(*a,**kw)
        return shared
    def ad(*a,**kw):pytest.fail('AD must be stopped before body')
    torch=h.fake_torch();torch.autograd.grad=wrapper(ad)
    safe=wrapper(lambda:123);ledger=m.bridge.native.Ledger(1,lambda e:None)
    with m.encode_guard(h.Model(),torch,ledger,[]):
        assert safe()==123
        with pytest.raises(ValueError,match='forbidden'):torch.autograd.grad(None)
