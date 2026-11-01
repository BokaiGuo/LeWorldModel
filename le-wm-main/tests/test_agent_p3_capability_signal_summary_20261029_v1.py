"""Synthetic saved schemas/analytic diagonal matrices only, NOT HOST evidence."""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_capability_signal_summary_20261029_v1 as m


def physical_row(i,root,arm,missing=False):
    state=[float({'H1':3,'H3':3,'Z':2,'P':4}[arm]),0.,0.,0.,0.,0.,0.]
    s2=state[0]**2
    metrics=dict(state7=state,goal7=[0.]*7,native7_distance=state[0],native7_squared=s2,
        joint_xy_error=state[0],joint_xy_squared=s2,agent_velocity_error=0.,agent_velocity_squared=0.,
        raw_angle_error=0.,raw_angle_squared=0.,wrapped_angle_error=0.,native_angle_term=0.,native_success=True)
    cp=dict(step=10,state=state,snapshot=dict(state=state,full_fixture_body={'not_state_only':True}),
            pixel_sha256=m.digest(['RAM pixel',root,arm]),file_sha256=m.digest(['RAM file',root,arm]))
    return dict(index=i,root_slot=root,arm=arm,control_step=10,absolute_step=20,
        status='no_witness' if missing else 'horizon_completed',available=not missing,
        missing_reason='no_witness' if missing else None,task_id=m.digest(['RAM base',root]),
        endpoint=None if missing else cp,metrics=None if missing else metrics)


def ctx(latent,history):
    return dict(context_length=3,latent_dim=192,action_dim=10,layout='RAM_only_full3',
                latent_history=latent,action_history=history)


def child(t,z,raw):
    before=t['root_context']
    a=((np.asarray(raw,np.float32)-np.asarray(t['action_mean'],np.float32))/np.asarray(t['action_std'],np.float32)).reshape(10)
    return ctx(before['latent_history'][1:]+[z], [before['action_history'][-1],a.tolist()])


def fixture():
    """Final producer ABI, but no producer import/math or real results."""
    flags=dict.fromkeys(m.FALSE_FLAGS,False)
    e=dict(schema=m.ENDPOINT+'.summary',denominator=dict(C_logical_slots=4,C_no_witness=3,ZP_references=4,comparisons=8),
        actual_time=dict(control=10,absolute=20),known_roots=2,fresh=False,scientific_admission=False,
        P1_admission=False,P3_admission=False,full_C3_demonstrated=False,
        C_slots=[physical_row(i,r,a,i!=0) for i,(r,a) in enumerate(m.LOGICAL)],
        ZP_root_references=[physical_row(i,r,a) for i,(r,a) in enumerate(((0,'Z'),(0,'P'),(1,'Z'),(1,'P')))],
        comparisons=[{'fixture':i} for i in range(8)],
        authorities={k:dict(path='/RAM_ONLY/'+k,sha256=m.digest(k)) for k in ('C','ZP')},
        inherited_accounting={k:dict(physical_report_sha256=m.digest(['RAM report',k])) for k in ('C','ZP')})
    ref=dict(prepared='/RAM_ONLY/prepared.json',prepare_sha256=m.digest('RAM prepare'),
             report='/RAM_ONLY/report.json',report_sha256=m.digest('RAM report'),
             completion='/RAM_ONLY/completion.json',completion_sha256=m.digest('RAM completion'))
    r=dict(schema=m.STAGE+'.consume',status='saved_complete_NOT_admitted',fixed_signals=7,fixed_probes=5,
        signals=[],probes=[],plans=[],metrics=[],verification=dict(full_native_closure=True,native_calls=17,
        saved_SVD_calls=7,saved_new_neural_calls=0,saved_new_physics=0),
        resource_receipt=dict(caps=m.expected_resources(),attempted=m.expected_resources(),completed=m.expected_resources(),
            FLOPs=None,cuda_ns=None,backward_engine_calls=None,events=[{'explicit_RAM_metadata':True}]),
        unknowns=dict(FLOPs=None,backward_engine_calls=None,cuda_ns=None),
        upstream_fullstrict_rebuilt_here=False,bridge_authority={'RAM_ONLY':True},inherited_counts={'RAM_encoder':5},
        inherited_physical_counts={'RAM_fixture_only':True},prepared=ref['prepared'],prepare_sha256=ref['prepare_sha256'],
        report_sha256=ref['report_sha256'],completion_sha256=ref['completion_sha256'],**flags)
    tasks=[]
    for i in range(7):
        root=i if i<2 else m.ORDER[i-2][0]
        goal=[.125]*10+[0.]*181+[.25]
        diag=np.arange(1,11,dtype=np.float64)
        j=np.zeros((192,10),np.float32);j[:10]=np.diag(diag)
        nominal=np.zeros(192,np.float32)
        # Independent diagonal closed form, inactive ball; no producer bounded_ridge.
        params=diag*np.asarray(goal[:10])/(diag**2+.002)
        remaining=np.asarray(goal)-j.astype(np.float64)@params
        linear=float(remaining@remaining/192);penalty=float(.002*(params@params)/192)
        solve=dict(params=params.tolist(),singular_values=diag[::-1].tolist(),linear_terminal_cost=linear,
                   ridge_penalty=penalty,own_reach_value=float((remaining@remaining+.002*(params@params))/192),
                   radius=m.RADIUS,norm=float(np.linalg.norm(params)),ball_multiplier=0.)
        incoming=np.zeros((10,2),np.float32);incoming[:5]=.125;incoming[5:]=-.25
        mean=[.125,-.125];std=[.5,.25]
        history=((incoming-np.asarray(mean,np.float32))/np.asarray(std,np.float32)).reshape(2,10).tolist()
        c=ctx([[0.]*192,[.25]*192,[.5]*192],history)
        bind={k:m.digest(['RAM binding',root,i,k]) for k in ('task_id','root_sha256','model_sha256')}
        bind['model_sha256']=m.digest('RAM shared model')
        t=dict(binding=bind,root_context=c,goal=goal,projection=np.eye(192).tolist(),action_mean=mean,action_std=std)
        for key,value in (('goal_sha256',goal),('projection_sha256',t['projection']),
                         ('normalizer_sha256',[mean,std]),('context_sha256',c)):
            bind[key]=m.digest(value)
        response=dict(horizon=1,nominal=nominal.tolist(),jacobian=j.tolist(),
            binding_sha256=m.digest(bind),parent_context_sha256=m.digest(c),
            producer_sha256='6ce7722eca40cdd5f8958ba9590d8dc3d76d38bd10895d5ca494c1db3b6d100b')
        s=dict(index=i,status='complete',response=response,solve=solve,nominal_context=child(t,nominal.tolist(),[[0.,0.]]*5))
        r['signals'].append(s);tasks.append((t,incoming))
    for i,(root,role) in enumerate(m.ORDER):
        s=r['signals'][i+2];t,incoming=tasks[i+2]
        raw=np.tanh(np.asarray(s['solve']['params'],np.float64)).astype('<f4').reshape(5,2)
        end=(np.asarray(s['response']['nominal'])+np.asarray(s['response']['jacobian'])@raw.reshape(10)).astype(np.float32)
        whole=[child(t,end.tolist(),raw.tolist())]
        probe=dict(index=i,status='complete',raw=raw.tolist(),whole=whole,segmented=deepcopy(whole))
        r['probes'].append(probe)
        physical=e['C_slots'][0] if role=='C' else e['ZP_root_references'][2*root+(role=='P')]
        source=deepcopy(physical['endpoint']);source['path']=f'/RAM_ONLY/png{i}.png'
        pk='C' if role=='C' else 'ZP'
        identity=dict(namespace=m.BRIDGE,index=i,root_slot=root,role=role,absolute_steps=[10,15,20],
            history_sources=[dict(absolute_step=x,snapshot=source['snapshot'],pixel_sha256=source['pixel_sha256']) for x in (10,15,20)],
            source_receipt=e['authorities'][pk],source_report=dict(path='/RAM_ONLY/report',sha256=e['inherited_accounting'][pk]['physical_report_sha256']),
            raw10_float32_sha256=hashlib.sha256(incoming.tobytes()).hexdigest())
        def nativeq(v):return float(np.mean((np.asarray(v,np.float64)-np.asarray(t['goal'],np.float64))**2))
        Q=dict(actual_cp10=nativeq(t['root_context']['latent_history'][-1]),nominal=nativeq(s['response']['nominal']),
               whole=nativeq(whole[0]['latent_history'][-1]),segmented=nativeq(whole[0]['latent_history'][-1]))
        plan=dict(schema=m.STAGE+'.probe_plan',index=i,root_slot=root,role=role,checkpoint=10,probe_checkpoint=15,
            task=t,task_sha256=m.digest(t),signal=deepcopy(s['response']),solve=deepcopy(s['solve']),nominal_context=deepcopy(s['nominal_context']),
            raw5=raw.tolist(),raw_float32_bits=raw.tobytes().hex(),raw_float32_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            params_float64=deepcopy(s['solve']['params']),whole=whole,segmented=deepcopy(whole),Q=Q,
            authority=ref,bridge_authority=r['bridge_authority'],native_call_ids=dict(signal=i+2,whole=7+2*i,segmented=8+2*i),
            causal_raw10=incoming.tolist(),causal_raw10_bits=incoming.tobytes().hex(),endpoint_identity=identity,
            source=source,base_task_id=physical['task_id'],native_seal_sha256=m.digest('RAM seal'),**flags)
        plan['witness_sha256']=m.digest(plan);r['plans'].append(plan)
    for kind,indices in (('actual',range(5)),('nominal',range(7))):
        for i in indices:
            if kind=='actual':v=r['plans'][i]['Q']['actual_cp10']
            else:
                goal=tasks[i][0]['goal'];v=float(np.mean((np.asarray(r['signals'][i]['response']['nominal'])-np.asarray(goal))**2))
            r['metrics'].append(dict(kind=kind,index=i,Q=v))
    for i,p in enumerate(r['plans']):
        for field in ('whole','segmented'):r['metrics'].append(dict(kind=field,index=i,Q=p['Q'][field]))
    # Two independently serialized authorities have no shared Python references.
    return json.loads(m.encode(r)),json.loads(m.encode(e))


@pytest.fixture(scope='module')
def panel():return fixture()


def test_all7_full192x10_and_fixed5_model_probes(panel):
    r,e=deepcopy(panel);x=m.summarize(r,e)
    assert len(x['signals'])==7 and len(x['probes'])==5 and len(x['comparisons'])==8
    assert [s['control_step'] for s in x['signals']]==[5,5,10,10,10,10,10]
    assert sum(len(row['math']['singular_values']) for row in x['signals'])==70
    assert sum(np.asarray(row['math']['jacobian']).size for row in x['signals'])==13440
    assert all(len(s['math']['rank_prefixes'])==11 for s in x['signals'])
    assert x['C_dual_control_conditions'][0]['C_immediate_cost_both_controls'] is False
    assert [x['available'] for x in x['comparisons']]==[True,True]+[False]*6
    assert all(p['actual_probe_gain'] is None and not p['physical_gain_guaranteed'] for p in x['probes'])
    assert all(x[k] is False for k in m.FALSE_FLAGS)
    assert [(p['root_slot'],p['A_role'],p['B_role']) for p in x['same_time_role_pairs']]==[(0,'C','Z'),(0,'C','P'),(0,'Z','P'),(1,'Z','P')]
    assert x['comparisons'][0]['R10_C_vs_control']['relation']=='tie'
    assert x['comparisons'][2]['R10_C_vs_control']['relation']=='null'
    assert x['local_array_SVD_calls']==7 and x['winner'] is None


def test_diagonal_svd_target_ridge_and_all_rank_contributions(panel):
    r,_=panel;s=r['signals'][0];goal=r['plans'][0]['task']['goal']
    x=m.signal_math(s['response'],s['solve'],goal)
    assert x['singular_values']==list(map(float,range(10,0,-1)))
    assert x['orthogonal_target_energy']==.25**2/192
    for mode,diag in zip(x['modes'],range(10,0,-1),strict=True):
        assert mode['target_coordinate']==.125
        assert mode['bounded_coordinate']==diag*.125/(diag*diag+.002)
        assert mode['cap_objective_change']==pytest.approx(0.,abs=1e-20)
    assert x['rank_prefixes'][0]['linear_terminal']==pytest.approx(math.fsum([.125**2]*10+[.25**2])/192,rel=1e-15)
    assert x['rank_prefixes'][-1]['omitted_target_energy']==0.
    assert x['rank_prefixes'][-1]['ridge_objective']==pytest.approx(s['solve']['own_reach_value'])
    assert abs(x['target_energy_roundoff'])<1e-18


def test_active_ball_and_nullspace_no_new_threshold():
    j=np.zeros((192,10),np.float32);j[0,0]=1.
    goal=[10.]+[0.]*191
    mu=10/m.RADIUS-1.-.002
    params=[m.RADIUS]+[0.]*9
    remaining=(10-m.RADIUS)**2/192;penalty=.002*m.RADIUS**2/192
    s=dict(params=params,singular_values=[1.]+[0.]*9,ball_multiplier=mu,
           linear_terminal_cost=remaining,ridge_penalty=penalty,own_reach_value=remaining+penalty)
    x=m.signal_math(dict(jacobian=j.tolist(),nominal=[0.]*192),s,goal)
    assert x['modes'][0]['bounded_coordinate']==pytest.approx(m.RADIUS)
    assert all(v['singular_value']==0 for v in x['modes'][1:])
    assert all(v['bounded_coordinate']==0 for v in x['modes'][1:])
    assert x['modes'][0]['cap_objective_change']>0
    assert len(x['rank_prefixes'])==11 and not x['physical_gain_guaranteed']


@pytest.mark.parametrize('a,b,relation',[(0.,-0.,'tie'),(5e-324,-5e-324,'greater'),(-5e-324,5e-324,'less'),(None,1.,'null')])
def test_no_multiplication_strict_sign_ties_and_null(a,b,relation):
    assert m.cmp(a,b)['relation']==relation


@pytest.mark.parametrize('kind',['RAM','partial','denominator','VJP','unknown','flag','metricorder','signalorder',
    'Jshape','Jbool','badproducer','planorder','witness','goal','params','incoming','history','whole',
    'source','snapshot','physicalauthority','pixel','cp5_endpoint','borrow_C'])
def test_fail_closed_schema_and_binding_mutations(panel,kind):
    r,e=deepcopy(panel)
    if kind=='RAM':r['status']='RAM_fixture_only_NOT_evidence'
    elif kind=='partial':r['verification']['full_native_closure']=False
    elif kind=='denominator':r['signals'].pop()
    elif kind=='VJP':r['resource_receipt']['completed']['VJP_directions']-=1
    elif kind=='unknown':r['unknowns']['FLOPs']=0
    elif kind=='flag':r['P3_admission']=True
    elif kind=='metricorder':r['metrics'].reverse()
    elif kind=='signalorder':r['signals'].reverse()
    elif kind=='Jshape':r['signals'][0]['response']['jacobian'].pop()
    elif kind=='Jbool':r['signals'][0]['response']['jacobian'][0][0]=True
    elif kind=='badproducer':r['signals'][0]['response']['producer_sha256']='0'*64
    elif kind=='planorder':r['plans'].reverse()
    elif kind=='witness':r['plans'][0]['witness_sha256']='0'*64
    elif kind=='goal':r['plans'][0]['task']['goal'][0]+=.5
    elif kind=='params':r['plans'][0]['params_float64'][0]+=.1
    elif kind=='incoming':r['plans'][0]['causal_raw10'][0][0]+=.25
    elif kind=='history':r['plans'][0]['whole'][0]['action_history'][0][0]+=.25
    elif kind=='whole':r['plans'][0]['segmented'][0]['latent_history'][-1][0]+=.25
    elif kind=='source':r['plans'][0]['source']['state'][0]+=1.
    elif kind=='snapshot':r['plans'][0]['source']['snapshot']['full_fixture_body']['not_state_only']=False
    elif kind=='physicalauthority':r['plans'][0]['endpoint_identity']['source_receipt']['sha256']='0'*64
    elif kind=='pixel':r['plans'][0]['source']['pixel_sha256']='0'*64
    elif kind=='cp5_endpoint':e['ZP_root_references'][0]['control_step']=5
    else:e['C_slots'][1]['endpoint']=deepcopy(e['C_slots'][0]['endpoint'])
    if kind in ('incoming','history','whole','params','goal','snapshot','physicalauthority','pixel','source'):
        # Even an internally rehashed damaged plan must fail semantic joining.
        p=r['plans'][0];p.pop('witness_sha256');p['witness_sha256']=m.digest(p)
    with pytest.raises((ValueError,KeyError)):
        m.summarize(r,e)


def test_no_local_spectrum_physical_guarantee(panel):
    r,e=deepcopy(panel);x=m.summarize(r,e)
    assert all(p['predicted_improvement_vs_same_context_zero']>0 for p in x['probes'])
    assert all(p['full_protocol_condition'] is None for p in x['comparisons'])
    assert all(p['full_delayed_benefit_chain'] is None for p in x['C_dual_control_conditions'])


@pytest.mark.parametrize('slot',[0,5,11,12,21])
def test_original_exact_Q_value_mismatch_even_with_matching_rehashed_plan(panel,slot):
    r,e=deepcopy(panel);metric=r['metrics'][slot];metric['Q']+=1e-5
    kind,index=metric['kind'],metric['index']
    if kind=='actual':r['plans'][index]['Q']['actual_cp10']=metric['Q']
    elif kind=='nominal' and index>=2:r['plans'][index-2]['Q']['nominal']=metric['Q']
    elif kind in ('whole','segmented'):r['plans'][index]['Q'][kind]=metric['Q']
    for p in r['plans']:
        p.pop('witness_sha256');p['witness_sha256']=m.digest(p)
    with pytest.raises(ValueError,match='original.*Q'):
        m.summarize(r,e)


def test_causal_token_mismatch_after_all_raw_hashes_repaired(panel):
    r,e=deepcopy(panel);p=r['plans'][0]
    incoming=np.asarray(p['causal_raw10'],'<f4');incoming[0,0]=.5
    p['causal_raw10']=incoming.tolist();p['causal_raw10_bits']=incoming.tobytes().hex()
    p['endpoint_identity']['raw10_float32_sha256']=hashlib.sha256(incoming.tobytes()).hexdigest()
    p.pop('witness_sha256');p['witness_sha256']=m.digest(p)
    with pytest.raises(ValueError,match='incoming tokens'):
        m.summarize(r,e)


def test_native_original_angle_bug_is_not_silently_fixed():
    out=m.native_metrics([0.,0.,0.,0.,7.,0.,0.],[0.]*7)
    assert out['native_angle_term']<0 and out['native_success']
    assert out['wrapped_angle_error']>math.pi/9


def disk_fixture(tmp_path,panel,monkeypatch):
    r,e=deepcopy(panel)
    rp=tmp_path/'RAM_receipt.json';ep=tmp_path/'RAM_endpoints.json'
    rp.write_bytes(m.encode(r));ep.write_bytes(m.encode(e))
    refs=dict(stageB_receipt=dict(path=str(rp),sha256=m.sha(rp)),endpoint_summary=dict(path=str(ep),sha256=m.sha(ep)))
    # Only the fixed actual endpoint byte pin changes for an explicit synthetic input.
    # Real sealed source pins, build, write, verify and every math path remain intact.
    monkeypatch.setattr(m,'ENDPOINT_SHA',refs['endpoint_summary']['sha256'])
    return refs


def test_complete_temporary_disk_build_write_verify_and_tamper(tmp_path,panel,monkeypatch):
    refs=disk_fixture(tmp_path,panel,monkeypatch)
    result=m.build(refs)
    out=tmp_path/(m.NAME+'_RAM_fixture.json');pub=m.write(out,result)
    assert m.verify(out,pub['sha256'])['status']=='saved_arithmetic_verified_NOT_admitted'
    assert result['new_neural_calls']==result['new_physics_calls']==result['new_AD_calls']==0
    assert set(result['sources'])==set(m.source_closure())|{x['path'] for x in refs.values()}
    assert out.stat().st_size<m.OUTPUT
    with pytest.raises(FileExistsError):m.write(out,result)
    wrong=deepcopy(result);wrong['signals'][0]['math']['singular_values'][9]=123.
    tamper=tmp_path/(m.NAME+'_RAM_tamper.json');h=m.write(tamper,wrong)['sha256']
    with pytest.raises(ValueError):m.verify(tamper,h)
    Path(refs['stageB_receipt']['path']).write_bytes(b'{}')
    with pytest.raises(ValueError):m.build(refs)


def test_source_closure_real_sealed_bytes_and_deadline(tmp_path,panel,monkeypatch):
    refs=disk_fixture(tmp_path,panel,monkeypatch)
    sources=m.source_closure()
    assert sources[str(Path(m.__file__).resolve())]==m.sha(Path(m.__file__).resolve())
    assert all('/results/' not in p for p in sources)
    ticks=iter((0.,181.))
    with pytest.raises(ValueError,match='wall'):m.build(refs,clock=lambda:next(ticks))


@pytest.mark.parametrize('body',[b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":Infinity}'])
def test_bad_json_fail_closed(tmp_path,body):
    p=tmp_path/'RAM_bad.json';p.write_bytes(body)
    with pytest.raises(ValueError):m.read(p,m.sha(p))


def test_relative_paths_and_wrong_output_name_rejected(tmp_path):
    with pytest.raises(ValueError):m.absolute('relative.json')
    with pytest.raises(ValueError):m.write(tmp_path/'other.json',{})


def test_wrong_external_byte_SHA_rejected(tmp_path):
    p=tmp_path/'RAM_authority.json';p.write_bytes(b'{}')
    with pytest.raises(ValueError,match='external byte SHA'):m.read(p,'0'*64)


def test_cold_cli_no_producer_math_torch_or_physics_imports():
    tree=ast.parse(Path(m.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.ImportFrom):assert not n.module.startswith(('experiments','torch','pymunk'))
        if isinstance(n,ast.Import):assert not any(v.name.startswith(('experiments','torch','pymunk')) for v in n.names)
    script=f'import sys; from experiments.spectral_reachability import {m.NAME}; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules'
    subprocess.run([sys.executable,'-c',script],cwd=m.ROOT,check=True)
    subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'--help'],cwd=m.ROOT,check=True,capture_output=True)
