"""RAM-only metadata/PNG/callback mocks. Never import or run neural/physics producers."""
from copy import deepcopy
from io import BytesIO
import math

import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import moving_prediction_matched_sampling_comparison_20261027_v1 as c


def seal(events):
    out=[];previous=None
    for i,e in enumerate(events):
        e={k:v for k,v in e.items() if k not in ('index','previous_sha256','event_sha256')}
        e=dict(e,index=i,previous_sha256=previous);e['event_sha256']=c.ident(e)
        previous=e['event_sha256'];out.append(e)
    return out,b''.join(c.canonical(e)+b'\n' for e in out)


@pytest.fixture(scope='module')
def saved():
    mem={};root='/RAM';goals={s:[100.+s,0.,100.,0.,.5,0.,0.] for s in (0,1)};initial=[0.]*7
    def put(name,value):
        value=value if isinstance(value,bytes) else c.canonical(value)
        path=root+'/'+name;mem[path]=value;return dict(file=path,sha256=c.sha(value))
    source_prepare={'RAM_only':True};source_events=[dict(kind='run_started',prepare=source_prepare)];source_rows=[]
    for s in (0,1):
        rootsha=c.sha(np.asarray(initial+goals[s],np.float64).tobytes());blind=c.ident(['blind',s])
        source_rows.append(dict(slot=dict(index=s),root_sha256=rootsha,blind_payload_sha256=blind))
        source_events.extend([dict(kind='native_return',slot=s,phase='construction',step=1,state=goals[s]),
            dict(kind='fresh_observation',slot=s,phase='replay0',step=10,state=initial,snapshot=dict(state=initial))])
    source_events.append(dict(kind='run_finished',status='fixed_slots_processed',counts={'RAM':2}))
    source_events,sb=seal(source_events)
    pack=dict(schema=c.math_saved.SOURCEPACK_SCHEMA,sourcepack_complete=True,status='fixed_slots_processed',errors=[],
        rows=source_rows,counts={'RAM':2},prepare_id=c.ident(source_prepare),journal_chain_tail=source_events[-1]['event_sha256'])
    sourcepack=put('sourcepack.json',pack);source_journal=put('source.jsonl',sb);source_sha=sourcepack['sha256']
    targets=c.math_saved.source_targets(pack,source_events)
    buf=BytesIO();Image.fromarray(np.zeros((224,224,3),np.uint8)).save(buf,format='PNG');png=buf.getvalue()
    fsha=c.sha(png);psha=c.sha(np.zeros((224,224,3),np.uint8).tobytes())
    refs={'sourcepack':sourcepack,'source_journal':source_journal}
    for name in ('physical_source','planner_source','planner_tests','planner_doc'):refs[name]=put(name+'.txt',b'RAM source seal, never imported')
    inherited={r:dict(attempted=8,completed=8) for r in ('encoder','projector')}
    contract=dict(roots=2,groups=4,arms=list(c.ARMS),iterations=3,samples=24,candidates_per_group=72,horizon=2,raw_steps=10,batch_size=1,
        extra_replay=0,padding=0,nominal_evaluations=0,AD=0,J=0,VJP=0,SVD=0,new_encoder=0,new_image_projector=0,new_physics=0,
        Q='original identity192 terminal mean192 squared error',with_target=False,source_authority_sha256=source_sha,inherited_encoding_once=inherited,
        training=False,Qfit=False,MATH=True,TF32=False,autocast=False,CPU_neural_fallback=False)
    slots=[dict(source_slot=s,task_id=c.ident(['task',s]),blind_sha256=source_rows[s]['blind_payload_sha256'],
        task_export='/RAM/task'+str(s),task_sha256=c.ident(['task bytes',s]),task_contract_sha256=c.ident(['tc',s])) for s in (0,1)]
    pp=dict(schema=c.PLANNER+'.prepare',contract=contract,expected_model_hash='a'*64,slots=slots)
    refs['planner_prepare']=put('planner_prepare.json',pp)
    plans=[];pr=[]
    for s,a in c.keys(c.ARMS):
        p=dict(schema=c.PLANNER+'.model_plan',**slots[s],arm=a,commands=np.zeros((10,2),np.float32).tolist(),
            logits=np.zeros((2,10)).tolist(),prediction={'RAM_only':True},model_terminal_cost=1.25+s,selected_index=0,
            with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,official_MPPI=False)
        p['selection_sha256']=c.ident(p);plans.append(p)
        pr.append(dict(source_slot=s,arm=a,status='model_plan_sealed',resources=c.budget(144),
            predictor_samples=dict(attempted=144,completed=144),selection_sha256=p['selection_sha256'],commands=p['commands']))
    planner=dict(schema=c.PLANNER+'.report',status='matched_sampling_plans_sealed_not_admitted',errors=[],first_error=None,
        commands_released=True,source_verified=True,model_unchanged=True,model_hash_before='a'*64,model_hash_after='a'*64,
        prepared=refs['planner_prepare']['file'],prepare_sha256=refs['planner_prepare']['sha256'],contract=contract,
        resources=c.budget(576),predictor_samples=dict(attempted=576,completed=576),inherited_encoding_once=inherited,rows=pr)
    refs['planner_report']=put('planner_report.json',planner)
    plans=[dict(p,whole_run_report_sha256=refs['planner_report']['sha256']) for p in plans]
    receipt=dict(schema=c.PLANNER+'.consume',rows=plans,resources=c.budget(576),predictor_samples=dict(attempted=576,completed=576),
        P1_admission=False,official_MPPI=False,equal_total_compute=False,prediction_calls_and_samples_matched_between_sampling_arms=True,
        inherited_encoding_once=inherited)
    entry=dict(authority=c.PLANNER,output='/RAM',report_sha256=refs['planner_report']['sha256'],
        source_sha256=refs['planner_source']['sha256'],tests_sha256=refs['planner_tests']['sha256'],innerdoc_sha256=refs['planner_doc']['sha256'])
    manifest=dict(schema=c.PHYSICAL+'.selection',sourcepack=refs['sourcepack']['file'],sourcepack_sha256=source_sha,
        selection_fixed_before_physics=True,physical_outcome_selection=False,arms=list(c.ARMS),authorities=[entry])
    refs['physical_manifest']=put('manifest.json',manifest)
    physical_plans=[]
    for p in plans:
        s=p['source_slot'];ph={f:p[f] for f in ('commands','task_id','blind_sha256','task_export','task_sha256','task_contract_sha256')}
        ph.update(root_slot=s,root_sha256=source_rows[s]['root_sha256'],arm=p['arm'],compiler_authority=c.PLANNER,
            compiler_report=refs['planner_report']['file'],compiler_report_sha256=refs['planner_report']['sha256'],
            compiler_selection_sha256=p['selection_sha256'],own_plan_sha256=c.ident(p))
        ph['selection_digest']=c.ident(ph);physical_plans.append(ph)
    physical_prepare=dict(schema=c.PHYSICAL+'.prepare',plans=physical_plans,sourcepack_sha256=source_sha,
        manifest=refs['physical_manifest']['file'],manifest_sha256=refs['physical_manifest']['sha256'],
        bindings={r['file']:r['sha256'] for r in refs.values()},contract=dict(caps=c.CAPS,with_target=False,own_horizon=10,warmup=10))
    refs['physical_prepare']=put('physical_prepare.json',physical_prepare)
    binding=dict(prepared=refs['physical_prepare']['file'],prepare_sha256=refs['physical_prepare']['sha256'])
    refs['physical_binding']=put('execution_binding.json',binding)
    events=[dict(kind='run_started',prepare=physical_prepare,execution_binding=binding)];rows=[];actual=[]
    def operation(resource,where,**fields):
        r=dict(resource=resource,**where,**fields,status='pending');events.append(dict(kind='operation_dispatch_intent',receipt=r))
        r=dict(r,status='returned');actual.append(r);events.append(dict(kind='operation_returned',receipt=r))
    for i,p in enumerate(physical_plans):
        s,a=p['root_slot'],p['arm'];where=dict(root_slot=s,arm=a,selection_digest=p['selection_digest'],branch=i)
        row=dict(branch_index=i,root_slot=s,arm=a,selection_digest=p['selection_digest'],selected_plan=p,status='horizon_completed',
            initial_exact=True,root_state=initial,goal=goals[s],PNG=22,checkpoints=[],warmup_returned=10,
            own_attempted=10,own_returned=10,full_horizon=True,native_success=False,first_error=None,secondary_errors=[])
        operation('environment_construct',where)
        events.append(dict(kind='branch_started',**where,plan=p,raw_warmup=c.fixed_raw().astype(float).tolist(),raw_float32_bits=c.fixed_raw().tobytes().hex()))
        events.append(dict(kind='goal_attached',**where,goal=goals[s]))
        for phase in ('warmup','plan'):
            for t in range(11):
                state=initial.copy();state[0]=(t if phase=='plan' else 0)*(2 if a==c.ARMS[1] else 1)
                if t:
                    cmd=c.fixed_raw()[t-1] if phase=='warmup' else c.commands(p['commands'])[t-1]
                    operation('public_step',where,phase=phase,step=t,raw_action=cmd.astype(float).tolist())
                    events.append(dict(kind='returned_transition',**where,phase=phase,step=t,state=state,
                        command=cmd.astype(float).tolist(),returned_goal_state=goals[s]))
                    if phase=='warmup':events.append(dict(kind='warmup_return',**where,step=t,raw_float32_bits=cmd.tobytes().hex()))
                operation('explicit_render',where)
                name=f'b{i}_{phase}_{t}.png';mem['/RAM/'+name]=png
                cp=dict(phase=phase,step=t,png=name,file_sha256=fsha,pixel_sha256=psha,state=state,
                        observation_state=state,snapshot=dict(state=state))
                events.extend([dict(kind='fresh_observation',**where,receipt=cp),
                    dict(kind='guard_receipt',**where,phase=phase,step=t,accepted=True,render_preserved_dynamics=True)])
                if phase=='plan' and t in (0,5,10):row['checkpoints'].append(cp)
            if phase=='plan':row['final_state']=state
        for resource,cap in c.CAPS.items():
            if resource in ('environment_construct','environment_close','explicit_render','public_step'):continue
            for _ in range(cap//4):operation(resource,where)
        events.extend([dict(kind='cleanup_intent',**where),dict(kind='cleanup_returned',**where)])
        events.append(dict(kind='branch_closed',receipt=row));rows.append(row)
    counts={k:dict(attempted=v,completed=v) for k,v in c.CAPS.items()}
    status='matched_sampling_physical_recorded_not_admitted'
    events.append(dict(kind='run_finished',status=status,counts=counts,errors=[]));events,journal=seal(events)
    refs['physical_journal']=put('journal.jsonl',journal)
    report=dict(schema=c.PHYSICAL+'.report',status=status,rows=rows,errors=[],counts=counts,contract=physical_prepare['contract'],
        sourcepack_sha256=source_sha,prepared=binding['prepared'],prepare_sha256=binding['prepare_sha256'],
        actual_operation_receipts=actual,journal_sha256=refs['physical_journal']['sha256'],journal_chain_tail=events[-1]['event_sha256'],
        total_internal_physics_scope='hook_observed_native_work')
    refs['physical_report']=put('physical_report.json',report)
    bref=put('comparison_binding.json',dict(schema=c.NAME+'.binding',files=refs,reference=None))
    return dict(mem=mem,refs=refs,binding=bref,report=report,physical_prepare=physical_prepare,events=events,targets=targets,
        planner=planner,planner_prepare=pp,plans=plans,receipt=receipt,source_sha=source_sha)


@pytest.fixture(autouse=True)
def synthetic_authority(monkeypatch,saved):
    # Test-only synthetic source authority; no public CLI override of the real pin.
    monkeypatch.setattr(c.math_saved,'SOURCEPACK_SHA',saved['source_sha'])


def consume(s):
    def call(refs,name,read):
        assert name==c.PLANNER
        return deepcopy(s['receipt'])
    return call


def physical(s,**changes):
    args=dict(report=s['report'],prepared=s['physical_prepare'],events=s['events'],targets=s['targets'],arms=c.ARMS,
              frame=lambda name:s['mem']['/RAM/'+name]);args.update(changes)
    return c.physical_material(**args)


def test_RAM_full_binding_to_comparison_end_to_end(saved):
    b=saved['binding'];out=c.material(b['file'],b['sha256'],read=saved['mem'].__getitem__,consume=consume(saved))
    assert len(out['rows'])==4 and len(out['frame_sha256'])==88
    assert out['root_denominator']==2 and out['branch_denominator']==4
    assert out['planner_resources']==c.budget(576)
    assert out['physical_budget']['actual']['physics_step']['completed']==816
    assert out['physical_budget']['actual']['public_step']['completed']==80
    assert out['inherited_encoding_once']['encoder']['completed']==8
    assert out['relative_GN_D2P']['status']=='missing_reference_not_compared'
    assert all(r['actual_latent_Q'] is None for r in out['rows'])
    assert all(r['planner_budget']['resources']==c.budget(144) for r in out['rows'])
    assert out['new_neural_calls']==out['new_physics_calls']==0
    assert out['P1_admission'] is out['equal_total_FLOPs'] is False
    assert len(c.canonical(out))<2*1024**2


def test_join_uses_identity_not_report_order(saved):
    report=deepcopy(saved['report']);report['rows'].reverse()
    out=physical(saved,report=report)
    assert [(r['root_slot'],r['arm']) for r in out['rows']]==list(c.keys(c.ARMS))
    assert all(p['second_minus_first_terminal']['native_distance']<0 for p in c.paired(out['rows'],c.ARMS))


@pytest.mark.parametrize('axis',['predict_API','predictor','action_encoder','pred_proj'])
def test_group_budget_cannot_balance_143_145(saved,axis):
    r=deepcopy(saved['planner']);r['rows'][0]['resources'][axis]={'attempted':143,'completed':143}
    r['rows'][1]['resources'][axis]={'attempted':145,'completed':145}
    blobs={'planner_report':c.canonical(r),'planner_prepare':c.canonical(saved['planner_prepare'])}
    with pytest.raises(ValueError,match='each group'):c.planner_material(saved['refs'],blobs,consume(saved),saved['mem'].__getitem__)


def test_predictor_samples_not_API_or_context_rows(saved):
    r=deepcopy(saved['planner']);r['rows'][0]['predictor_samples']={'attempted':432,'completed':432}
    with pytest.raises(ValueError,match='each group'):
        c.planner_material(saved['refs'],{'planner_report':c.canonical(r),'planner_prepare':c.canonical(saved['planner_prepare'])},consume(saved),saved['mem'].__getitem__)


@pytest.mark.parametrize('field',['extra_replay','nominal_evaluations','padding','AD'])
def test_no_extra_evaluation_or_hidden_neural_budget(saved,field):
    p=deepcopy(saved['planner_prepare']);p['contract'][field]=1;r=deepcopy(saved['planner']);r['contract']=p['contract']
    # Keep external SHA consistent to isolate protocol gate, without writing files.
    refs=deepcopy(saved['refs']);refs['planner_prepare']['sha256']=c.sha(c.canonical(p));r['prepare_sha256']=refs['planner_prepare']['sha256']
    with pytest.raises(ValueError,match='no extra'):c.planner_material(refs,{'planner_report':c.canonical(r),'planner_prepare':c.canonical(p)},consume(saved),saved['mem'].__getitem__)


def test_all_four_branches_required(saved):
    with pytest.raises(ValueError):c.index(saved['report']['rows'][:-1],c.ARMS)
    with pytest.raises(ValueError):c.index([saved['report']['rows'][0]]*4,c.ARMS)


@pytest.mark.parametrize('field,value',[('goal',[1.]*7),('root_state',[1.]*7),('selection_digest','0'*64)])
def test_original_goal_root_selection_not_labels(saved,field,value):
    r=deepcopy(saved['report']);r['rows'][0][field]=value
    with pytest.raises(ValueError):physical(saved,report=r)


def test_guard_and_actual7state_required(saved):
    e=deepcopy(saved['events']);next(x for x in e if x['kind']=='guard_receipt' and x['phase']=='plan' and x['step']==5)['accepted']=False
    with pytest.raises(ValueError,match='guard'):physical(saved,events=e)


def test_failed_capture_keeps_missing10_null_full_denominator(saved):
    r=deepcopy(saved['report']);e=deepcopy(saved['events']);r['status']='partial_or_failed';r['errors']=[{'RAM_error':True}]
    row=r['rows'][0];row['status']='failed';row['first_error']={'RAM_capture':True};row['checkpoints']=row['checkpoints'][:-1]
    next(x for x in e if x['kind']=='branch_closed')['receipt']=row
    e[-1].update(status=r['status'],errors=r['errors'])
    out=physical(saved,report=r,events=e)
    assert len(out['rows'])==4 and out['rows'][0]['terminal10'] is None
    assert out['rows'][0]['own_returned']==10 and out['rows'][0]['missing_checkpoints']==[10]
    assert c.paired(out['rows'],c.ARMS)[0]['second_minus_first_terminal']['native_distance'] is None


def test_intent_is_not_actual_callback_entry(saved):
    report=deepcopy(saved['report']);events=deepcopy(saved['events'])
    # A failed post-intent dispatch leaves intent but no actual receipt/return.
    where=dict(root_slot=0,arm=c.ARMS[0],selection_digest=report['rows'][0]['selection_digest'],branch=0)
    events.insert(-1,dict(kind='operation_dispatch_intent',receipt=dict(resource='environment_construct',**where,status='pending')))
    result=c.operation_counts(report,saved['physical_prepare'],events,c.ARMS,False)
    assert result['actual']['environment_construct']['attempted']==4


def test_lost_return_journal_preserves_actual_receipt_lower_bound(saved):
    report=deepcopy(saved['report']);report['status']='partial_or_failed'
    events=deepcopy(saved['events']);events.pop(next(i for i,e in enumerate(events) if e['kind']=='operation_returned' and e['receipt']['resource']=='physics_step'))
    result=c.operation_counts(report,saved['physical_prepare'],events,c.ARMS,False)
    assert result['actual']['physics_step']['completed']==816
    assert result['journal_entry_return_lower_bounds']['physics_step']['completed']==815
    assert not result['journal_complete_returns']


def test_actual_return_without_final_transition_label_retained(saved):
    report=deepcopy(saved['report']);events=deepcopy(saved['events']);report['status']='partial_or_failed'
    report['errors']=[{'stage':'durable_return_log','RAM_only':True}]
    row=report['rows'][0];row['status']='failed';row['checkpoints']=row['checkpoints'][:-1];row['first_error']=report['errors'][0]
    events=[e for e in events if not(e['kind']=='returned_transition' and e['root_slot']==0 and e['arm']==c.ARMS[0] and e['phase']=='plan' and e['step']==10)]
    next(e for e in events if e['kind']=='branch_closed')['receipt']=row
    events[-1].update(status=report['status'],errors=report['errors'])
    out=physical(saved,report=report,events=events)
    assert out['rows'][0]['own_returned']==10 and out['rows'][0]['terminal10'] is None
    assert out['rows'][0]['actual_return_without_transition_label']['plan']==1


def test_nonzero_raw_history_dispatch_cannot_be_rescaled(saved):
    e=deepcopy(saved['events']);x=next(x for x in e if x['kind']=='operation_dispatch_intent' and x['receipt']['resource']=='public_step')
    x['receipt']['raw_action']=[.001,0]
    with pytest.raises(ValueError,match='raw command'):physical(saved,events=e)


def test_native_wrapped_and_velocity_semantics():
    state=np.zeros(7);state[4]=2*math.pi+.8;m=c.metrics(np.zeros(7),state)
    assert m['native_success'] and not m['wrapped_allxy_success'] and m['negative_native_angle']
    state=np.zeros(7);state[5:]=100;m=c.metrics(np.zeros(7),state)
    assert m['native_success'] and m['native_distance']>100 and not m['success_has_velocity_condition']


def test_native_flag_preserved_with_independent_disagreement(saved):
    report=deepcopy(saved['report']);events=deepcopy(saved['events']);report['rows'][0]['native_success']=True
    next(e for e in events if e['kind']=='branch_closed')['receipt']=report['rows'][0]
    out=physical(saved,report=report,events=events)
    assert out['rows'][0]['native_success_reported'] is True
    assert out['rows'][0]['native_flag_recomputation_mismatch'] is True
    assert out['rows'][0]['terminal10']['native_success'] is False


@pytest.mark.parametrize('turn',[-2,-1,0,1,2])
def test_wrapped_2pi_invariance(turn):
    state=np.zeros(7);state[4]=.8+turn*2*math.pi
    assert c.metrics(np.zeros(7),state)['wrapped_angle_error']==pytest.approx(.8)


def test_GN_D2P_pair_only_on_exact_identity(saved):
    rows=physical(saved)['rows'];ref=[]
    for s,a in c.keys(c.GN_ARMS):
        r=deepcopy(rows[2*s]);r['arm']=a
        for key in ('task_sha256','task_id','task_contract_sha256'):r[key]='a'*64
        ref.append(r)
    for r in rows:
        for key in ('task_sha256','task_id','task_contract_sha256'):r[key]='a'*64
    result=c.relative(rows,ref)
    assert len(result['pairs'])==8 and all(p['status']=='descriptive_only_compute_unmatched' for p in result['pairs'])
    assert all(p['budget_match'] is False for p in result['pairs'])
    ref[0]['horizon']=3
    result=c.relative(rows,ref)
    assert result['pairs'][0]['native_distance_minus_reference'] is None
    assert result['pairs'][0]['mismatched_fields']==['horizon']


def test_pixels_and_journal_are_externally_bound(saved):
    mem=dict(saved['mem']);mem['/RAM/b0_plan_5.png']+=b'corrupt'
    with pytest.raises(ValueError,match='PNG'):c.material(saved['binding']['file'],saved['binding']['sha256'],read=mem.__getitem__,consume=consume(saved))
    mem=dict(saved['mem']);mem['/RAM/journal.jsonl']+=b'corrupt'
    with pytest.raises(ValueError,match='physical_journal'):c.material(saved['binding']['file'],saved['binding']['sha256'],read=mem.__getitem__,consume=consume(saved))


def test_external_binding_rejected_before_consumer(saved):
    with pytest.raises(ValueError,match='binding SHA'):
        c.material(saved['binding']['file'],'0'*64,read=saved['mem'].__getitem__,consume=lambda *a:pytest.fail('must not consume'))


def test_GN_reference_saved_adapter_RAM_contract(saved):
    """Legacy GN handoff/report are distinct, bound to exact commands and native labels."""
    mem=dict(saved['mem']);refs={}
    def put(key,value):
        data=value if isinstance(value,bytes) else c.canonical(value);path='/RAM/reference_'+key
        mem[path]=data;refs[key]=dict(file=path,sha256=c.sha(data));return path
    for name in ('planner_source','planner_tests','planner_doc'):put(name,b'RAM reference source seal')
    pp=dict(schema=c.GN_PLANNER+'.prepare',contract={'mathematical_vjp_per_root':dict(zip(c.GN_ARMS,(768,2112)))})
    put('planner_prepare',pp)
    p_rows=[]
    for s,a in c.keys(c.GN_ARMS):p_rows.append(dict(source_slot=s,arm=a,resources={'predict':dict(attempted=78,completed=78)},AD_API_resources={'RAM':True}))
    pr=dict(schema=c.GN_PLANNER+'.report',status='moving_history_plans_sealed_not_admitted',errors=[],commands_released=True,
        model_unchanged=True,prepared=refs['planner_prepare']['file'],prepare_sha256=refs['planner_prepare']['sha256'],contract=pp['contract'],rows=p_rows)
    put('planner_report',pr);plans=[]
    for p in saved['plans']:
        p=deepcopy(p);p['arm']=c.GN_ARMS[c.ARMS.index(p['arm'])];p['schema']=c.GN_PLANNER+'.model_plan'
        p.pop('whole_run_report_sha256');p.pop('selection_sha256');p['selection_sha256']=c.ident(p)
        p['whole_run_report_sha256']=refs['planner_report']['sha256'];plans.append(p)
    handoff=dict(schema='moving_history_planner_handoff.v1',status='commands_verified_before_physics',errors=[],
        projection=dict(source_report_sha256=refs['planner_report']['sha256']),data=dict(plans=[{k:v for k,v in p.items() if k!='prediction'} for p in plans]))
    put('planner_handoff',handoff)
    p=deepcopy(saved['physical_prepare']);p['schema']=c.GN_PHYSICAL+'.prepare';p['bindings']={refs['planner_handoff']['file']:refs['planner_handoff']['sha256']}
    p['contract']['caps']={r:v for r,v in c.CAPS.items() if not r.startswith('environment_')}
    newplans=[]
    for old in p['plans']:
        q=dict(root_slot=old['root_slot'],arm=c.GN_ARMS[c.ARMS.index(old['arm'])],commands=old['commands'],
            compiler_report=refs['planner_handoff']['file'],compiler_report_SHA=refs['planner_handoff']['sha256'])
        q['selection_digest']=c.ident(q);newplans.append(q)
    p['plans']=newplans;put('physical_prepare',p)
    bind=dict(prepared=refs['physical_prepare']['file'],prepare_sha256=refs['physical_prepare']['sha256']);put('physical_binding',bind)
    report=deepcopy(saved['report']);report['schema']=c.GN_PHYSICAL+'.report';report['status']='moving_history_physical_recorded'
    report['contract']=p['contract'];report['counts']={r:v for r,v in report['counts'].items() if not r.startswith('environment_')}
    for i,row in enumerate(report['rows']):
        row['arm']=newplans[i]['arm'];row['selection_digest']=newplans[i]['selection_digest'];row.pop('selected_plan')
    events=[]
    for old in saved['events']:
        e=deepcopy(old);kind=e['kind']
        if kind in ('cleanup_intent','cleanup_returned'):continue
        if kind.startswith('operation_'):
            op=e['receipt']
            if op['resource'].startswith('environment_'):continue
            e={k:v for k,v in op.items() if k!='status'};e['kind']='operation_attempted' if kind=='operation_dispatch_intent' else 'operation_completed'
        if 'arm' in e:
            i=e['branch'];e['arm']=newplans[i]['arm'];e['selection_digest']=newplans[i]['selection_digest']
        if kind=='run_started':e=dict(kind='run_started',prepare=p)
        if kind=='branch_started':e['plan']=newplans[e['branch']]
        if kind=='branch_closed':e=dict(kind='branch_closed',receipt=report['rows'][e['receipt']['branch_index']])
        if kind=='run_finished':e=dict(kind='run_finished',status=report['status'],counts=report['counts'],errors=[])
        events.append(e)
    events,data=seal(events);put('physical_journal',data);report['journal_sha256']=refs['physical_journal']['sha256'];report['journal_chain_tail']=events[-1]['event_sha256']
    put('physical_report',report)
    def saved_consume(refs,name,read):
        assert name==c.GN_PLANNER
        return dict(schema=name+'.consume',status='moving_commands_verified_not_admitted',plans=plans)
    result=c.reference_material(refs,saved['targets'],saved_consume,mem.__getitem__)
    assert len(result['rows'])==4 and result['physical_budget']['actual']['public_step']['completed']==80
    for row in result['rows']:assert row['task_sha256']==saved['plans'][row['root_slot']*2]['task_sha256']
    assert result['rows'][0]['planner_budget']['mathematical_VJP']==768


def test_help_and_mock_CLI_no_file_output(monkeypatch,capsys):
    with pytest.raises(SystemExit) as e:c.main(['--help'])
    assert e.value.code==0
    monkeypatch.setattr(c,'material',lambda *a:dict(status='RAM_only'))
    monkeypatch.setattr(c.math_saved,'write_new',lambda *a:'a'*64)
    assert c.main(['--binding','/RAM/b.json','--binding-sha','a'*64,'--output','/RAM/no_write.json'])==0
    assert 'RAM_only' in capsys.readouterr().out
