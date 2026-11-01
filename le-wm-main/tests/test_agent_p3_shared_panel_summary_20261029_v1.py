"""Synthetic RAM/saved bytes only; no imported producer, strict replay or neural work."""
import ast
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_shared_panel_summary_20261029_v1 as m


FLAGS=dict(P1_admission=False,P3_admission=False,freshness=False,scientific_admission=False)


def z(value):return np.full(192,value,np.float32).tolist()


def context(value):return dict(latent_history=[z(value)]*3,action_history=[[0.]*10]*2,
    layout='latent3_action2',context_length=3,latent_dim=192,action_dim=10)


def ledger(prior=0):
    """Literal actual-shaped operation list, constructed without producer math."""
    ops=[]
    def add(resource,units,shape=None,directions=None,phase='program'):
        delta=dict.fromkeys(m.AXES,0);delta.update(units)
        i=len(ops);ops.append(dict(index=i,resource=resource,shape=shape,directions=directions,delta=delta,
            status='returned',started_ns=10*i,ended_ns=10*i+3,scope=dict(phase=phase),error=None))
    for i in range(44):
        phase='diagnostic' if i<8 else 'program'
        add('predict',dict(predict_API=1,predict_samples=1,predict_context_tokens=3),[1,3,192],phase=phase)
        add('action_encoder',dict(action_encoder_calls=1,action_encoder_samples=1,action_encoder_context_tokens=3),[1,3,10],phase=phase)
        add('predictor',dict(predictor_calls=1,predictor_samples=1,predictor_context_tokens=3),[1,3,192],phase=phase)
        add('pred_proj',dict(pred_proj_calls=1,pred_proj_rows=3),[3,192],phase=phase)
    for resource,key,count in [('jacobian','jacobian_API',2),('grad','grad_API',384),('raw_backward','raw_backward_callbacks',384),
        ('array_svd','array_SVD_calls',4),('solve','solve_calls',2),('signal_compile','signal_compile_calls',2),
        ('program_compile','program_compile_calls',2),('replay','replay_calls',18),('metric','metric_calls',8),('schedule','schedule_decisions',1)]:
        for i in range(count):add(resource,{key:1},phase='program' if resource in ('solve','program_compile','replay','metric') else 'diagnostic')
    for _ in range(2):add('vjp',dict(VJP_directions=192),directions=192,phase='diagnostic')
    totals={k:sum(o['delta'][k] for o in ops) for k in m.AXES};wall={}
    for op in ops:wall[op['resource']]=wall.get(op['resource'],0)+3
    return dict(caps={k:(3-prior)*v for k,v in totals.items()},attempted=deepcopy(totals),completed=deepcopy(totals),
        events=ops,inclusive_operation_wall_ns=wall,FLOPs=None,cuda_ns=None,backward_engine_calls=None,wall_ns=100000)


def fixture(stops=(20,20,20,20,20,20,20,14),refused=()):
    """Eight actual-shaped rows; synthetic values have no predictive meaning."""
    contract=dict(schema=m.PRODUCER+'.contract',arms=list(m.ARMS),raw_target=20,**FLAGS)
    tasks=[dict(goal=z(0.),projection=np.eye(192).tolist(),root_context=context(root/8),
        action_mean=[0.,0.],action_std=[1.,1.]) for root in (0,1)]
    p=dict(schema=m.PRODUCER+'.prepare',contract=contract,tasks=tasks)
    r=dict(schema=m.PRODUCER+'.report',contract=deepcopy(contract),fixed_denominator=8,node_denominator=24,
        panel_denominator=48,candidate_denominator=192,known_tasks=2,errors=[],pending={},rows=[],nodes=[],
        counts=m.counters(m.NATIVE_AXES),per_episode_neural=[],status='shared_panel_recorded',**FLAGS)
    s=dict(schema=m.PRODUCER+'.consume',complete=True,status='saved_shared_panel_verified_NOT_admitted',
        fixed_denominator=8,node_denominator=24,panel_denominator=48,candidate_denominator=192,**FLAGS)
    events=[];arrays={};opid=0
    native_goal=[0.]*7
    def native(eid,kind,region):
        nonlocal opid
        op=dict(episode_id=eid,resource=kind,operation_id=opid,units=1,region=region);opid+=1
        events.extend([dict(kind='operation_attempted',**op),dict(kind='operation_completed',**op)])
        for axis in ('attempted','completed'):r['counts'][kind][axis]+=1
    for i,(root,arm) in enumerate((root,arm) for root in (0,1) for arm in m.ARMS):
        stop=stops[i];eid=f's{root}_{arm}';refuse=i in refused;bank=arm.split('_',1)[1]
        rawval=0. if bank=='endpoints' else .015625
        actions=[[0.,0.]]*min(stop,5)+[[rawval,rawval]]*max(0,stop-5)
        row=dict(episode_id=eid,source_slot=root,mode=arm,status='completed20' if stop==20 else 'contract_refused' if refuse else 'native_stopped_control',
            initial_exact=True,control_returned=stop,warmup_returned=10,executed_raw=deepcopy(actions),encodings=[],
            terminated=stop<20 and not refuse,truncated=False)
        r['rows'].append(row)
        for _ in range(10):native(eid,'public_step','warmup')
        for step in range(stop+1):
            state=[100.+root,100.,100.,100.,.5,2.,3.]
            if step==stop and stop<20 and not refuse:state=[0.]*7
            if step:
                native(eid,'public_step','prefix' if step<=5 else 'suffix')
                command=deepcopy(actions[step-1]);events.append(dict(kind='authorized_dispatch',episode_id=eid,step=step,command=command))
                reward=-float(np.linalg.norm(np.asarray(state)-native_goal))
                events.append(dict(kind='control_return',episode_id=eid,step=step,command=deepcopy(command),reward=reward,
                    terminated=row['terminated'] and step==stop,truncated=False))
            events.append(dict(kind='fresh_observation',episode_id=eid,receipt=dict(phase='plan',step=step,state=state,
                goal=deepcopy(native_goal),pixel_sha256=m.digest((root,arm,step,'pixels')),file_sha256=m.digest((root,arm,step,'PNG')))))
        for step in (5,10,15,20):
            if step>stop or (step==stop and step<20 and not refuse):continue
            value=np.full(192,root/8+step/32,np.float32);key=f'{eid}_{step}'
            arrays[key]=value;row['encodings'].append(dict(step=step,latent=key));native(eid,'encoder','encoding');native(eid,'projector','encoding')
        row['actual_Q20']=None if stop<20 else float(np.mean(np.asarray(arrays[f'{eid}_20'],np.float64)**2))
        ns=[];cached=deepcopy(tasks[root]['root_context'])
        for b in (5,10,15):
            node=dict(episode_id=eid,source_slot=root,mode=arm,boundary=b,status='not_reached',request=None,receipt=None,resource_receipt=None)
            r['nodes'].append(node);ns.append(node)
            if b>=stop and not(refuse and b==stop):continue
            rr=ledger(len([n for n in ns[:-1] if n['receipt'] is not None]));policy=arm.split('_',1)[0]
            h=1 if policy=='fixedH1' else 3
            current=deepcopy(cached);current['latent_history']=deepcopy(cached['latent_history'][1:])+[z(root/8+b/32)]
            history=np.asarray([[0.,0.]]*10+actions[:b],np.float32)[-10:]
            current['action_history']=history.reshape(2,10).tolist();cached=deepcopy(current)
            request=dict(task=dict(tasks[root],root_context=current),raw10=history.tolist())
            panels=[];fractions=[0.,1/3,2/3,1.] if bank=='endpoints' else [1/8,3/8,5/8,7/8]
            for horizon in (1,3):
                candidates=[];raws=[]
                for k in range(4):
                    commands=np.full((5*horizon,2),rawval+k/16,np.float32).tolist();raws.append(commands)
                    path=[context(root/8+b/32+.25+k/8+j/32) for j in range(horizon)]
                    cost=float(np.mean(np.asarray(path[-1]['latent_history'][-1],np.float64)**2))
                    candidates.append(dict(index=k,status='returned',actions=deepcopy(commands),whole=path,segmented=deepcopy(path),cost=cost))
                panels.append(dict(horizon=horizon,status='returned',bank=dict(actions=raws,fractions=fractions),
                    candidates=candidates,winner=deepcopy(candidates[0]),selected_path=deepcopy(candidates[0]['whole'])))
            isrefuse=refuse and b==stop
            receipt=dict(status='refused' if isrefuse else 'sealed',panels=panels,policy=dict(decision=dict(horizon=h),supported=not isrefuse),
                selected_horizon=None if isrefuse else h,commands=None if isrefuse else deepcopy(panels[0 if h==1 else 1]['winner']['actions'][:5]))
            node.update(status=receipt['status'],request=request,receipt=receipt,resource_receipt=rr)
        accounts=[n['resource_receipt'] for n in ns if n['resource_receipt'] is not None]
        totals={a:{k:sum(rr[a][k] for rr in accounts) for k in m.AXES} for a in ('attempted','completed')};wall={}
        phases={phase:dict.fromkeys(m.AXES,0) for phase in ('signal_and_policy','program')}
        for rr in accounts:
            for key,v in rr['inclusive_operation_wall_ns'].items():wall[key]=wall.get(key,0)+v
            for op in rr['events']:
                phase='program' if op['scope']['phase'] in ('program','selected') else 'signal_and_policy'
                for key,v in op['delta'].items():phases[phase][key]+=v
        expected=ledger()['completed']
        r['per_episode_neural'].append(dict(episode_id=eid,caps={k:3*v for k,v in expected.items()},
            **totals,inclusive_operation_wall_ns=wall,phases=phases))
    return p,r,s,events,arrays


@pytest.fixture(scope='module')
def full():return fixture()


def test_all8_24nodes_48panels_8pairs_independent_math(full):
    p,r,s,events,arrays=full;out=m.summarize(p,r,s,events,arrays.__getitem__)
    assert len(out['rows'])==8 and len(out['pairs'])==8 and out['known_roots']==2 and out['fresh_roots']==0
    assert out['panel_denominator']==48 and out['candidate_denominator']==192
    assert out['rows'][-1]['control_returned']==14 and out['rows'][-1]['steps'][15] is out['rows'][-1]['steps'][20] is None
    assert out['rows'][-1]['eventtime_terminal_native_success'] is True and out['rows'][-1]['native_success_at20'] is None
    assert out['pairs'][7]['at20']['actual_Q_delta'] is None
    row=out['rows'][0];assert row['counts']['neural']['predict_API']['completed']==132
    assert row['counts']['neural']['VJP_directions']['completed']==1152 and row['counts']['neural']['jacobian_API']['completed']==6
    assert row['counts']['neural']['grad_API']['completed']==1152 and row['counts']['native']['encoder']['completed']==4
    assert [x['panels'][1]['predicted_endpoint_step'] for x in row['nodes']]==[20,25,30]
    assert row['nodes'][1]['panels'][1]['error_decomposition'] is None
    assert row['nodes'][0]['panels'][1]['causal_endpoint_pair'] is False
    assert all(x is False for x in (out['P1_admission'],out['P3_admission'],out['equal_FLOPs'],out['full_strict_rebuilt_here']))


def test_clean_refusal_keeps_both_paid_panels_and_null_future():
    p,r,s,events,arrays=fixture(stops=(5,)*8,refused=tuple(range(8)))
    out=m.summarize(p,r,s,events,arrays.__getitem__)
    for row in out['rows']:
        assert row['status']=='contract_refused' and row['steps'][10] is row['steps'][20] is None
        assert row['nodes'][0]['executed_horizon'] is None and len(row['nodes'][0]['panels'])==2
        assert row['counts']['neural']['predict_API']['completed']==44
        assert all(not p['selected_for_execution'] for p in row['nodes'][0]['panels'])


def test_all_unreached_preserved_as_null():
    p,r,s,e,a=fixture(stops=(4,)*8);out=m.summarize(p,r,s,e,a.__getitem__)
    assert all(row['steps'][5] is row['steps'][20] is None and len(row['nodes'])==3 for row in out['rows'])
    assert all(pair['at20']['right_minus_left'] is None for pair in out['pairs'])


def test_independent_native_axes_and_angle_quirk():
    goal=[0.,0.,0.,0.,0.,3.,4.];state=[3.,4.,0.,12.,4*np.pi+2.,0.,0.]
    value=m.physical_metric(goal,state)
    assert value['joint_xy']==13 and value['velocity_error']==5 and value['agent_xy']==5 and value['object_xy']==12
    assert value['native_angle_term']<0 and value['native_success'] and value['angle_wrapped']>np.pi/9
    assert value['native7_reward']==-value['native7_distance']


def test_hand_calculable_error_identity_and_mean192():
    d=m.error_decomposition(z(1.),z(3.),z(2.))
    assert (d['model_Q'],d['real_Q'],d['cross'],d['quadratic'])==(4.,1.,2.,1.)
    assert d['identity_rounding_residual']==0 and not d['scientific_tolerance_changed']
    a=np.zeros(192);a[0]=12.;assert m.latent_q(z(0.),a)==144/192
    assert m.sign(1e-300)=='positive' and m.sign(-0.)=='zero' and m.sign(None)=='missing'


@pytest.mark.parametrize('fault',['schema','missing_row','row_order','node_order','goal','scalarQ','winner','paneldrop','panelreplay',
    'raw','actualreward','terminated','missingframe','context','nativegoal','ADunits','accountVJP','phase','wall','counts','strictpartial'])
def test_exact_fail_closed_mutations(full,fault):
    p,r,s,e,a=deepcopy(full)
    if fault=='schema':r['schema']='agent_p3_effective_schedule_20261029_v1.report'
    elif fault=='missing_row':r['rows'].pop()
    elif fault=='row_order':r['rows'].reverse()
    elif fault=='node_order':r['nodes'].reverse()
    elif fault=='goal':r['nodes'][0]['request']['task']['goal'][0]+=.1
    elif fault=='scalarQ':r['nodes'][0]['receipt']['panels'][0]['candidates'][0]['cost']+=1
    elif fault=='winner':r['nodes'][0]['receipt']['panels'][0]['winner']['index']=3
    elif fault=='paneldrop':r['nodes'][0]['receipt']['panels'].pop()
    elif fault=='panelreplay':r['nodes'][0]['receipt']['panels'][0]['selected_path'][0]['latent_history'][-1][0]+=.1
    elif fault=='raw':r['rows'][0]['executed_raw'][5][0]=.5
    elif fault=='actualreward':next(x for x in e if x['kind']=='control_return')['reward']+=1
    elif fault=='terminated':next(x for x in e if x['kind']=='control_return')['terminated']=True
    elif fault=='missingframe':e.pop(next(i for i,x in enumerate(e) if x['kind']=='fresh_observation'))
    elif fault=='context':r['nodes'][0]['request']['task']['root_context']['latent_history'][-1][0]+=.1
    elif fault=='nativegoal':next(x for x in e if x['kind']=='fresh_observation' and x['receipt']['step']==1)['receipt']['goal'][6]=1.
    elif fault=='ADunits':
        op=next(x for x in r['nodes'][0]['resource_receipt']['events'] if x['resource']=='vjp');op['delta']['VJP_directions']-=1
    elif fault=='accountVJP':r['per_episode_neural'][0]['completed']['VJP_directions']-=1
    elif fault=='phase':r['per_episode_neural'][0]['phases']['program']['replay_calls']-=1
    elif fault=='wall':r['nodes'][0]['resource_receipt']['inclusive_operation_wall_ns']['grad']-=1
    elif fault=='counts':r['counts']['public_step']['completed']-=1
    else:s['complete']=False
    with pytest.raises((ValueError,KeyError,IndexError)):m.summarize(p,r,s,e,a.__getitem__)


def test_no_producer_math_import_or_dynamic_execution():
    tree=ast.parse(Path(m.__file__).read_text());modules=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):modules.extend(x.name for x in n.names)
        elif isinstance(n,ast.ImportFrom):modules.append(n.module)
    assert all(x=='numpy' or x.split('.')[0] in {'__future__','argparse','copy','hashlib','io','json','os','pathlib'} for x in modules)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'eval','exec','__import__'} for n in ast.walk(tree))


def test_failed_attempt_remains_paid_not_refunded():
    p,r,s,e,a=fixture(stops=(5,)*8,refused=tuple(range(8)))
    n=r['nodes'][0];rr=n['resource_receipt'];op=rr['events'][-1];assert op['resource']=='vjp'
    op['status']='failed';n['status']='failed';rr['completed']['VJP_directions']-=192
    account=r['per_episode_neural'][0];account['completed']['VJP_directions']-=192
    account['phases']['signal_and_policy']['VJP_directions']-=192
    paid=m.paid_counts(r,e)[r['rows'][0]['episode_id']]
    assert paid['neural']['grad_API']==dict(attempted=384,completed=384)
    assert paid['neural']['VJP_directions']==dict(attempted=384,completed=192)
    assert all(x['status']=='unknown' and x['attempted'] is x['completed'] is None for x in paid['unknown_work'].values())


def test_partial_native_attempt_is_not_successfully_closed_or_refunded():
    p,r,s,e,a=fixture(stops=(4,)*8)
    i=next(i for i,x in enumerate(e) if x['kind']=='operation_completed')
    done=e.pop(i);ident=done['operation_id'];pending={k:v for k,v in done.items() if k!='kind'}
    r['pending']={str(ident):pending};r['counts'][done['resource']]['completed']-=1
    paid=m.paid_counts(r,e)[done['episode_id']]
    assert paid['native'][done['resource']]['attempted']==paid['native'][done['resource']]['completed']+1
    assert paid['pending_native'][0]['operation_id']==ident
    with pytest.raises(ValueError):m.summarize(p,r,s,e,a.__getitem__)


@pytest.mark.parametrize('fault',['shape','directions','unknown'])
def test_resource_units_are_rebuilt_from_native_shape_not_just_delta(fault):
    op=deepcopy(ledger()['events'][0])
    if fault=='shape':op['shape']=[1,2,192]
    elif fault=='directions':op['directions']=1
    else:op['resource']='hidden_flops_guess'
    with pytest.raises(ValueError):m.operation_delta(op)


def save_npy(path,a):
    stream=io.BytesIO();np.save(stream,a,allow_pickle=False);path.write_bytes(stream.getvalue())
    return dict(path=str(path),file_sha256=m.sha(path),array_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        shape=list(a.shape),dtype=str(a.dtype))


@pytest.mark.parametrize('fault',[None,'sha','array','float64','shape','tail','outside'])
def test_actual_saved_NPY_layout_and_byte_mutations(tmp_path,fault):
    output=tmp_path/'output';output.mkdir();a=np.zeros(192,np.float32);a[0]=-0.
    path=output/'real.npy'
    if fault=='outside':path=tmp_path/'outside.npy'
    if fault=='float64':a=a.astype(np.float64)
    rec=save_npy(path,a)
    if fault=='sha':rec['file_sha256']='e'*64
    elif fault=='array':rec['array_sha256']='e'*64
    elif fault=='shape':rec['shape']=[1,192]
    elif fault=='tail':path.write_bytes(path.read_bytes()+b'tail');rec['file_sha256']=m.sha(path)
    if fault is None:
        bindings={};out=m.load_npy(output,rec,bindings)
        assert out.tobytes()==a.tobytes() and np.signbit(out[0]) and str(path) in bindings
    else:
        with pytest.raises(ValueError):m.load_npy(output,rec,{})


def saved_corpus(tmp_path,monkeypatch):
    """Temporary actual-shaped files; synthetic strict is not real certification."""
    p,r,s,events,arrays=fixture(stops=(10,4,4,4,4,4,4,4))
    root=tmp_path/'root';out=root/'results'/'RAM';out.mkdir(parents=True)
    (root/'experiments/spectral_reachability').mkdir(parents=True)
    (root/'tests').mkdir();(root/'innerdocs').mkdir()
    source=root/'experiments/spectral_reachability'/(m.PRODUCER+'.py');source.write_text('# synthetic ABI fixture; never executed\n')
    (root/'tests'/('test_'+m.NAME+'.py')).write_text('# RAM fixture binding only\n')
    (root/'innerdocs'/(m.NAME+'.md')).write_text('RAM fixture documentation only\n')
    monkeypatch.setattr(m,'ROOT',root)
    p.update(sources={str(source):m.sha(source)},expected_model_hash='f'*64)
    def dump(path,obj):path.write_bytes(m.raw(obj));return m.sha(path)
    prep=root/'prepare.json';prepare_sha=dump(prep,p)
    for row in r['rows']:
        for enc in row['encodings']:
            key=enc['latent'];enc['latent']=save_npy(out/(key+'.npy'),arrays[key])
    execution=out/'execution_binding.json';dump(execution,dict(prepared=str(prep),prepare_sha256=prepare_sha))
    bookends=[dict(kind='run_started',contract=r['contract'])]+events+[
        dict(kind='run_finished',status=r['status'],counts=r['counts'],nodes_sha256=m.digest(r['nodes']))]
    lines=[];previous=None
    for i,event in enumerate(bookends):
        e=dict(event,index=i,previous_sha256=previous);e['event_sha256']=m.digest(e);previous=e['event_sha256'];lines.append(m.raw(e)+b'\n')
    journal=out/'journal.jsonl';journal.write_bytes(b''.join(lines))
    r.update(prepared=str(prep),prepare_sha256=prepare_sha,journal_sha256=m.sha(journal),journal_chain_tail=previous,
        model_before=dict(original_state_dict='f'*64),model_after=dict(original_state_dict='f'*64),
        source_manifest_sha256=m.digest(p['sources']),final_check=dict(model_unchanged=True),wall_seconds=1.25,CUDA_peak_bytes=[0,0])
    report=out/'report.json';report_sha=dump(report,r)
    s.update(report_sha256=report_sha,prepare_sha256=prepare_sha,per_episode_neural=deepcopy(r['per_episode_neural']))
    strict=root/'strict.json';strict_sha=dump(strict,s)
    return dict(prepared=str(prep),prepare_sha=prepare_sha,report=str(report),report_sha=report_sha,strict=str(strict),strict_sha=strict_sha)


def test_saved_roundtrip_report_strict_NPY_and_eight_pairs(tmp_path,monkeypatch):
    args=saved_corpus(tmp_path,monkeypatch);out=m.analyze(**args)
    assert out['fixed_denominator']==out['pair_denominator']==8 and not out['producer_arithmetic_imported']
    assert out['rows'][0]['steps'][5]['actual_Q']==25/1024 and out['rows'][0]['steps'][10]['actual_Q'] is None
    path=tmp_path/(m.NAME+'_RAM.json');m.write(path,out)
    m.same(m.read(path),m.analyze(**args),'real saved writer/reader deterministic summary')
    m.main(['verify','--summary',str(path),'--summary-sha',m.sha(path)])
    with pytest.raises(ValueError):m.write(path,out)


@pytest.mark.parametrize('fault',['prepare_sha','report_sha','strict_sha','source','journal','array','schema','strictPrepare'])
def test_saved_authority_byte_mutations_rejected(tmp_path,monkeypatch,fault):
    args=saved_corpus(tmp_path,monkeypatch)
    if fault.endswith('_sha'):args[fault]='e'*64
    elif fault=='source':(m.ROOT/'experiments/spectral_reachability'/(m.PRODUCER+'.py')).write_bytes(b'changed')
    elif fault=='journal':(Path(args['report']).parent/'journal.jsonl').write_bytes(b'{}\n')
    elif fault=='array':next(Path(args['report']).parent.glob('*.npy')).write_bytes(b'bad')
    elif fault=='schema':
        obj=m.read(args['report']);obj['schema']='agent_p3_effective_schedule_20261029_v1.report'
        Path(args['report']).write_bytes(m.raw(obj));args['report_sha']=m.sha(args['report'])
        strict=m.read(args['strict']);strict['report_sha256']=args['report_sha'];Path(args['strict']).write_bytes(m.raw(strict));args['strict_sha']=m.sha(args['strict'])
    else:
        strict=m.read(args['strict']);strict['prepare_sha256']='d'*64;Path(args['strict']).write_bytes(m.raw(strict));args['strict_sha']=m.sha(args['strict'])
    with pytest.raises(ValueError):m.analyze(**args)


@pytest.mark.parametrize('new_after_failure',[False,True])
def test_entry_order_parent_unwind_and_successful_child_not_refunded(new_after_failure):
    p,r,s,e,a=fixture(stops=(4,)*8);node=r['nodes'][0];ops=[]
    for i,(resource,start,end,status) in enumerate([
        ('signal_compile',0,100,'failed'),('grad',10,20,'returned'),('raw_backward',30,40,'failed')]):
        op=dict(index=i,resource=resource,shape=None,directions=None,status=status,started_ns=start,ended_ns=end,
            scope=dict(phase='diagnostic'));op['delta']=m.operation_delta(op);ops.append(op)
    if new_after_failure:
        op=dict(index=3,resource='solve',shape=None,directions=None,status='returned',started_ns=50,ended_ns=60,
            scope=dict(phase='diagnostic'));op['delta']=m.operation_delta(op);ops.append(op)
    totals={axis:{k:sum(op['delta'][k] for op in ops if axis=='attempted' or op['status']=='returned') for k in m.AXES}
        for axis in ('attempted','completed')}
    wall={}
    for op in ops:wall[op['resource']]=wall.get(op['resource'],0)+op['ended_ns']-op['started_ns']
    account=r['per_episode_neural'][0]
    rr=dict(caps=deepcopy(account['caps']),events=ops,**totals,inclusive_operation_wall_ns=wall,wall_ns=101)
    node.update(status='failed',resource_receipt=rr)
    account.update(**deepcopy(totals),inclusive_operation_wall_ns=deepcopy(wall))
    account['phases']['signal_and_policy']=deepcopy(totals['completed'])
    if new_after_failure:
        with pytest.raises(ValueError,match='earliest failure'):m.paid_counts(r,e)
    else:
        paid=m.paid_counts(r,e)[node['episode_id']]
        assert paid['neural']['signal_compile_calls']==dict(attempted=1,completed=0)
        assert paid['neural']['grad_API']==dict(attempted=1,completed=1)
        assert paid['neural']['raw_backward_callbacks']==dict(attempted=1,completed=0)
        assert paid['wall']['status']=='unknown' and paid['wall']['seconds'] is None


def test_exact_eight_contrast_directions_and_ties(full):
    p,r,s,e,a=full;out=m.summarize(p,r,s,e,a.__getitem__)
    assert [(x['source_slot'],x['contrast']) for x in out['pairs']]==[
        (root,kind) for root in (0,1) for kind in ('same_bank_policy','same_bank_policy','same_policy_bank','same_policy_bank')]
    for pair in out['pairs']:
        assert pair['direction']=='right_minus_left'
        cp0=pair['steps'][0]
        assert cp0['signs']['joint_xy']=='zero' and cp0['native_success']==dict(left=False,right=False)
        assert set(pair['resource_deltas']['neural'])==set(m.AXES)
