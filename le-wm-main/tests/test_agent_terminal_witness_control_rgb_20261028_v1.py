"""RAM original mixed/allmode authority and whole-row routing; no neural/physics."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as h
from experiments.spectral_reachability import agent_terminal_witness_control_rgb_20261028_v1 as m


def fixture(missing=()):
    slots=[];plans=[];modelrows=[];actual=[];oldrows=[];roots=[]
    for root in range(2):
        roots.append(dict(root_slot=root,root_state=[float(root)]*7,goal=[5.]*7,root_sha256=f'{root+20:064x}',
            goal_pose=[1.,2.,3.],goal_pixels={'off':'c'*64,'on':'d'*64},renderer={'RAM_ONLY':True}))
    for i,(root,condition) in enumerate(m.producer.ORDER):
        context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=[[0.]*192 for _ in range(3)],action_history=[[0.]*10 for _ in range(2)])
        slot=dict(index=i,root_slot=root,condition=condition,task_id=f'{i+1:064x}',task_sha256=f'{i+5:064x}',
            blind_sha256=f'{i+9:064x}',q=np.eye(192).tolist(),packet=dict(root_context=context,goal=[0.]*192))
        commands=np.full((10,2),i/16,np.float32).tolist();cs=[]
        for j in range(222):
            cs.append(dict(index=j,objective=.5+j,stage_cost=.25,predicted_terminal=[.5]*192,
                nonlinear_terminal_cost=.25,commands=deepcopy(commands),prefix_commands=deepcopy(commands[:5]),
                suffix_commands=deepcopy(commands[5:]),context=dict(latent_history=[[],[],[.5]*192])))
        result=dict(candidates=cs,winner=deepcopy(cs[0]))
        # Real model_plan schema and digest are constructed by the frozen producer.
        body=m.model.model_plan(slot,result);plan=dict(deepcopy(body),whole_run_report_sha256=m.producer.REPORT_SHA)
        prior=deepcopy(plan);prior['schema']=m.producer.allmode.PRODUCER+'.model_plan';prior['arm']='allmode_full10_conditional_D2P'
        prior.pop('selection_digest');prior.pop('whole_run_report_sha256')
        prior['selection_digest']=m.producer.core.digest(prior);prior['whole_run_report_sha256']='a'*64
        cp=[];trajectory=[]
        for step in ([0,5] if i in missing else [0,5,10]):
            f=dict(step=step,png=f'b{i}_plan_{step}.png',file_sha256='a'*64,pixel_sha256='b'*64,
                state=[float(root)]*7,snapshot=dict(RAM_ONLY=True))
            cp.append(f);trajectory.append(dict(deepcopy(f),phase='plan',condition=condition,overlay_index=1 if condition=='on' else 0))
        evidence=dict(index=i,root_slot=root,condition=condition,arm=prior['arm'],task_id=slot['task_id'],
            plan=deepcopy(prior),selection_digest=prior['selection_digest'],status='native_stopped' if i in missing else 'horizon_completed',
            root_state=deepcopy(roots[root]['root_state']),goal=deepcopy(roots[root]['goal']),root_sha256=roots[root]['root_sha256'],
            initial_exact=True,own_returned=9 if i in missing else 10,checkpoints=deepcopy(cp),trajectory=deepcopy(trajectory),
            missing_checkpoints=[10] if i in missing else [],final_state=[float(root)]*7,
            final_metrics=dict(native7_distance=1.),native_success=False,terminated=False,truncated=False)
        oldrows.append(deepcopy(evidence))
        if i<2:
            a=deepcopy(evidence);a.update(arm=m.producer.ARM,plan=deepcopy(plan),selection_digest=plan['selection_digest'],
                route='execute',inherited_evidence=None,evidence_origin='new_native_execution_known_root')
        else:
            a=dict(index=i,root_slot=root,condition=condition,arm=m.producer.ARM,task_id=slot['task_id'],
                plan=deepcopy(plan),selection_digest=plan['selection_digest'],route='inherit',status='inherited_saved_physical',
                inherited_evidence=deepcopy(evidence),evidence_origin='allmode_MAIN01_historical_not_fresh',
                own_returned=0,checkpoints=[],trajectory=[],missing_checkpoints=[0,5,10],
                **{k:deepcopy(evidence[k]) for k in ('final_state','final_metrics','native_success','terminated','truncated')})
        slots.append(deepcopy(slot));plans.append(deepcopy(plan));actual.append(a)
        modelrows.append(dict(index=i,plan=deepcopy(body),result=result))
    routes=[dict(index=i,old_index=i,mode='execute' if i<2 else 'inherit') for i in range(4)]
    old=dict(schema=m.producer.OLD+'.consume',rows=oldrows,physical_report_sha256=m.producer.OLD_REPORT_SHA)
    prep=dict(schema=m.producer.NAME+'.prepare',plans=plans,routes=routes,source_roots=deepcopy(roots),old_physical_consume=old,
        arguments=dict(old_output='/RAM_ONLY/allmode',old_report_sha256=m.producer.OLD_REPORT_SHA))
    strict=dict(schema=m.producer.NAME+'.consume',status=m.producer.RECORDED,rows=actual,errors=[],fixed_denominator=4,
        known_roots=2,fresh_roots=0,inherited=2,physical_report_sha256=m.PHYSICAL_REPORT_SHA,
        producer_report_sha256=m.producer.REPORT_SHA,old_report_sha256=m.producer.OLD_REPORT_SHA,
        P1_admission=False,scientific_admission=False,freshness=False)
    compiled=dict(slots=slots);oldprep=dict(source_roots=deepcopy(roots))
    oldcompiled=deepcopy(compiled);report=dict(rows=modelrows)
    return prep,strict,compiled,report,oldprep,oldcompiled


def prepared(missing=()):
    rows=m.fixed_rows(*fixture(missing),'/RAM_ONLY/mixed')
    return dict(schema=m.NAME+'.prepare',contract=m.contract(),rows=rows,
        expected_calls={k:4-len(missing) if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})


def test_true_whole_row_inheritance_not_missing_or_spliced():
    p=prepared();rows=p['rows']
    assert len(rows)==4 and all(r['source']['step']==10 for r in rows)
    assert [r['source_route'] for r in rows]==['execute','execute','inherit','inherit']
    assert all(Path(rows[i]['source']['path']).parent==Path('/RAM_ONLY/mixed') for i in (0,1))
    assert all(Path(rows[i]['source']['path']).parent==Path('/RAM_ONLY/allmode') for i in (2,3))
    assert all(r['predicted_Q']==.25 and r['original_objective']==.5 for r in rows)


@pytest.mark.parametrize('fault',['schema','route','inherited_row','inherited_raw','inherited_context','inherited_goal',
    'inherit_wrapper_outcome','new_row_plan','root','physical_goal','condition','cp_state','cp_early','cp_missing',
    'old_report','winner','model_raw','model_goal','count'])
def test_real_routes_and_identity_mutations_rejected(fault):
    p,s,c,r,op,oc=fixture()
    if fault=='schema':s['schema']=m.frozen.producer.NAME+'.consume'
    elif fault=='route':s['rows'][2]['route']='execute'
    elif fault=='inherited_row':s['rows'][2]['inherited_evidence']['trajectory'].pop()
    elif fault=='inherited_raw':p['old_physical_consume']['rows'][2]['plan']['commands'][0][0]=.9
    elif fault=='inherited_context':oc['slots'][2]['packet']['root_context']['latent_history'][0][0]=1.
    elif fault=='inherited_goal':op['source_roots'][1]['goal'][0]=1.
    elif fault=='inherit_wrapper_outcome':s['rows'][2]['final_state'][0]=9.
    elif fault=='new_row_plan':s['rows'][0]['plan']['commands'][0][0]=.9
    elif fault=='root':s['rows'][0]['root_state'][0]=1.
    elif fault=='physical_goal':s['rows'][0]['goal'][0]=1.
    elif fault=='condition':s['rows'][0]['trajectory'][-1]['condition']='on'
    elif fault=='cp_state':s['rows'][0]['checkpoints'][-1]['state'][0]=9.
    elif fault=='cp_early':s['rows'][0]['own_returned']=9
    elif fault=='cp_missing':s['rows'][0]['missing_checkpoints']=[10]
    elif fault=='old_report':p['arguments']['old_report_sha256']='0'*64
    elif fault=='winner':r['rows'][0]['result']['winner']=deepcopy(r['rows'][0]['result']['candidates'][1])
    elif fault=='model_raw':r['rows'][0]['result']['winner']['commands'][0][0]=.9
    elif fault=='model_goal':c['slots'][0]['packet']['goal'][0]=1.
    else:s['rows'].pop()
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,s,c,r,op,oc,'/RAM_ONLY/mixed')


def execute(monkeypatch,missing=()):
    p=prepared(missing);rows=deepcopy(p['rows']);store=h.old.Memory();torch=h.fake_torch();model=h.Model();errors=[]
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    trace=m.bridge.Trace(model,torch,store,lambda:None,p['expected_calls'])
    ledger=m.bridge.native.Ledger(1,lambda e:store.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    def encode(pix,row,scope,ordinal):
        ledger.image=ordinal
        with m.bridge.physical.saved.overlay.traced_scope(trace,scope,row):
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(h.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(h.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.frozen.encode_guard(model,torch,ledger,errors):m.process(p,rows,store,encode)
    return p,dict(schema=m.NAME+'.report',contract=m.contract(),rows=rows,errors=errors,blocked=ledger.blocked,
        modules=trace.entries,guards=ledger.entries,counts=m.bridge.counts(trace.entries)),store


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)


def test_complete_arrays_events_guard_no_schema_conversion(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    result=m.validate(p,r,s.events,s.load)
    assert len(result)==4 and r['counts']['encoder']==dict(attempted=4,completed=4)
    assert [x['source_route'] for x in result]==['execute','execute','inherit','inherit']
    assert all(x['predicted_Q']==.25 and not x['P1_admission'] for x in result)


@pytest.mark.parametrize('missing',[(0,),(2,),(1,3),(0,1,2,3)])
def test_real_source_missing_only_no_inheritance_false_missing(monkeypatch,missing):
    p,r,s=execute(monkeypatch,missing);out=m.validate(p,r,s.events,s.load)
    assert len(out)==4 and sum(x['latent_missing'] for x in out)==len(missing)


@pytest.mark.parametrize('fault',['event','pre','pixels','latent','hidden197','route','goal','guard','count','blocked'])
def test_saved_native_pixel_preprocess_and_event_mutations(replay,monkeypatch,fault):
    p,r0,s0=replay;r=deepcopy(r0);events=deepcopy(s0.events);s=h.old.Memory();s.files=dict(s0.files)
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    if fault=='event':events.append(dict(kind='UNKNOWN'))
    elif fault=='pre':r['rows'][0]['preprocessed']=s.array('badpre',np.zeros((1,1,3,224,224),np.float32))
    elif fault=='pixels':r['rows'][0]['pixels']=s.array('badpixel',np.zeros((224,224,3),np.uint8))
    elif fault=='latent':r['rows'][0]['latent']=s.array('badlatent',np.zeros(192,np.float32))
    elif fault=='hidden197':r['rows'][0]['modules'][0]['output']=s.array('bad197',np.zeros((1,197,192),np.float32))
    elif fault=='route':r['rows'][2]['source_route']='execute'
    elif fault=='goal':r['rows'][0]['goal192'][0]=1.
    elif fault=='guard':r['guards'].pop()
    elif fault=='count':r['counts']['encoder']['completed']-=1
    else:r['blocked']=['predict']
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,events,s.load)


def test_main_wrapper_must_contain_exact_real_receipt():
    x=fixture()[1];summary=dict(schema='MAIN.mixedmode_physical_receipt.v1',receipt=deepcopy(x))
    m.validate_summary(summary,x)
    summary['receipt']['inherited']=0
    with pytest.raises(ValueError):m.validate_summary(summary,x)


def test_host_authority_before_any_io():
    with pytest.raises(ValueError):m.run('/not/read','a'*64,'/not/create','CPU')
