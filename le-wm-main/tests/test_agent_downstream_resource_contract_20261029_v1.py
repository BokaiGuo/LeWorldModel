"""Resource arithmetic and saved metadata only; no full strict or neural calls."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from experiments.spectral_reachability import agent_downstream_resource_contract_20261029_v1 as m


def p2_resource(n=6):
    values=dict(predict_API=n,predict_samples=n,predictor_context_tokens=3*n,
        action_encoder=n,predictor=n,pred_proj=n,pred_proj_rows=3*n,Jacobian=1,
        Jacobian_elements_requested=5760,mathematical_VJP_requested=192,AD_primal_packet_steps=3,
        scalar_backward=0,Adam_step=0,virtual_noise=0,root_SVD=2,root_ridge=1,
        encoder=0,projector=0,physics=0,training=0)
    return dict(attempted=values,completed=deepcopy(values),engine_internal_backward_calls=None,
        independent_JVP_calls=None,total_FLOPs=None)


def ledger(failed=False,blocked=False):
    specs=[('predict',[1,3,192],None),('action_encoder',[1,3,10],None),('predictor',[1,3,192],None),
        ('pred_proj',[3,192],None),('jacobian',None,None),('vjp',None,2),('grad',None,None),
        ('raw_backward',None,None),('solve',None,None),('array_svd',None,None)]
    a=dict.fromkeys(m.P3_AXES,0);c=dict(a);caps=dict.fromkeys(m.P3_AXES,100);ops=[];walls={}
    for i,(resource,shape,directions) in enumerate(specs):
        op=dict(index=i,resource=resource,shape=shape,directions=directions,started_ns=i*10,
            ended_ns=i*10+7,status='returned')
        d=dict.fromkeys(m.P3_AXES,0)
        if resource in m.SCALARS:d[m.SCALARS[resource]]=1
        elif resource=='vjp':d['VJP_directions']=2
        else:
            mapping=dict(predict=('predict_API','predict_samples','predict_context_tokens'),
                action_encoder=('action_encoder_calls','action_encoder_samples','action_encoder_context_tokens'),
                predictor=('predictor_calls','predictor_samples','predictor_context_tokens'))
            if resource=='pred_proj':d.update(pred_proj_calls=1,pred_proj_rows=3)
            else:d.update(zip(mapping[resource],(1,1,3)))
        op['delta']=d
        if i==len(specs)-1 and blocked:
            caps['array_SVD_calls']=0;op.update(status='blocked_before_dispatch',ended_ns=None)
        else:
            for k,v in d.items():a[k]+=v
            if i==len(specs)-1 and failed:op['status']='failed'
            else:
                for k,v in d.items():c[k]+=v
            walls[resource]=walls.get(resource,0)+7
        ops.append(op)
    return dict(caps=caps,events=ops,attempted=a,completed=c,inclusive_operation_wall_ns=walls)


def test_independent_ledger_vector_no_nested_double_count():
    out=m.p3_ledger(ledger());v=out['vector']
    assert v['predict_calls']['attempted']==v['predictor_calls']['attempted']==1
    assert v['predict_frames']['completed']==v['predictor_frames']['completed']==3
    assert v['VJP_directions']['completed']==2 and v['grad_API']['completed']==1
    assert v['raw_backward_callbacks']['completed']==1
    assert v['engine_backward']['status']==v['jacobian_elements']['status']=='unknown'
    assert v['scalar_backward']['status']==v['AD_primal_packets']['status']=='unknown'
    assert out['wall']['additive'] is False and out['wall']['total_seconds'] is None


@pytest.mark.parametrize('failed,blocked',[(True,False),(False,True)])
def test_nonrefund_and_predispatch(failed,blocked):
    result=m.p3_ledger(ledger(failed,blocked));v=result['vector']
    assert result['complete'] is False and result['disposition']=='failed_or_blocked_nonrefunded'
    assert v['array_SVD']['attempted']==int(failed) and v['array_SVD']['completed']==0


@pytest.mark.parametrize('successful_child_first',(False,True))
def test_original_Ledger_nested_failure_entry_order_nonrefund(successful_child_first):
    from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
    import itertools
    clock=itertools.count();caps=dict.fromkeys(core.AXES,10)
    native=core.Ledger(caps,clock=lambda:next(clock));native.scope=dict(fixture=True)
    def throw():raise RuntimeError('actual fixture child failure')
    def body():
        if successful_child_first:native.call('metric',lambda:1)
        native.call('solve',throw)
    with pytest.raises(RuntimeError):native.call('signal_compile',body)
    rr=native.receipt();assert rr['events'][0]['status']=='failed'
    # Actual event array is ENTRY order, even though parent failed last.
    assert rr['events'][0]['ended_ns']>rr['events'][-1]['ended_ns']
    result=m.p3_ledger(rr)
    assert result['complete'] is False and result['failed_operation_indices']==[0,len(rr['events'])-1]
    assert result['vector']['solve']['attempted']==1 and result['vector']['solve']['completed']==0
    if successful_child_first:assert rr['completed']['metric_calls']==1
    # A forged later operation with adjusted totals still must fail chronology.
    extra=deepcopy(rr['events'][-1]);extra.update(index=len(rr['events']),started_ns=100,ended_ns=101,status='returned')
    rr['events'].append(extra)
    with pytest.raises(ValueError,match='after earliest failure'):m.p3_ledger(rr)


@pytest.mark.parametrize('bad',('delta','shape','resource','count','wall','completed','index','unknownstatus','refund'))
def test_ledger_mutation_rejection(bad):
    rr=ledger();op=rr['events'][0]
    if bad=='delta':op['delta']['predict_samples']=2
    elif bad=='shape':op['shape'][1]=2
    elif bad=='resource':op['resource']='mystery'
    elif bad=='count':rr['attempted']['predict_API']=0
    elif bad=='wall':op['ended_ns']=-1
    elif bad=='completed':rr['completed']['predict_API']=2
    elif bad=='index':op['index']=True
    elif bad=='unknownstatus':op['status']='unknown'
    else:rr['caps']['predict_API']=0
    with pytest.raises(ValueError):m.p3_ledger(rr)


def test_p2_counter_normalization_unknowns_not_zero():
    v=m.p2_vector(p2_resource())
    assert v['predict_samples']['completed']==6 and v['predict_frames']['completed']==18
    assert v['encode_calls']['completed']==0 and v['encode_samples']['status']=='unknown'
    assert v['scalar_backward']['completed']==0 and v['VJP_directions']['completed']==192
    assert v['engine_backward']['status']==v['JVP']['status']=='unknown'
    assert 'not_rebuilt' in v['jacobian_requests']['evidence']


@pytest.mark.parametrize('bad',('bool','partialmissing','ratio','flops','refund'))
def test_p2_invalid_abi(bad):
    r=p2_resource()
    if bad=='bool':r['attempted']['predict_API']=True
    elif bad=='partialmissing':r['completed'].pop('root_ridge')
    elif bad=='ratio':r['attempted']['pred_proj_rows']=0
    elif bad=='flops':r['total_FLOPs']=1
    else:r['completed']['predict_API']=9
    with pytest.raises(ValueError):m.p2_vector(r)


def test_sum_unknown_propagates_and_rejects_wrong_units():
    a=m.p2_vector(p2_resource());b=deepcopy(a);v=m.sum_vectors([a,b])
    assert v['predict_calls']['completed']==12 and v['engine_backward']['status']=='unknown'
    b['predict_calls']['unit']='FLOPs'
    with pytest.raises(ValueError):m.sum_vectors([a,b])


def shared_rows():
    rows=[]
    identity={k:'a'*64 for k in m.IDENTITY}
    for strategy in ('baseline','schedule'):
        rows.append(dict(slot_id='0:'+strategy,root_key='0',strategy=strategy,identity=deepcopy(identity),
            resources=m.p2_vector(p2_resource()),wall=m.wall('search',2.,['search_IO'],['load'])))
    return rows


def test_sharedpanel_actual_consumer_even_equal_predict_not_equalcompute():
    rows=shared_rows();rows[1]['resources']['solve']['attempted']=3;rows[1]['resources']['solve']['completed']=3
    v=m.validate_shared_panel(rows,[r['slot_id'] for r in rows],['predict_calls','predict_samples'],
        dict(predict_calls=6),wall_scope='search',wall_cap_seconds=3.)
    assert v['status']=='selected_resource_axes_equal_only'
    assert v['total_FLOPs']=='unknown' and not v['equal_FLOPs'] and not v['equal_total_compute'] and not v['equal_wall']
    assert set(v['operator_FLOPs'].values())=={'unknown'}
    x=m.validate_shared_panel(rows,[r['slot_id'] for r in rows],['solve'],{})
    assert x['status']=='blocked' and '0:unequal_actual_axis:solve' in x['blockers']


@pytest.mark.parametrize('bad',('identity','missingidentity','unknown','cap','wallunknown','wallscope','wallcap'))
def test_sharedpanel_failclosed_blockers(bad):
    rows=shared_rows()
    if bad=='identity':rows[1]['identity']['goal']='b'*64
    elif bad=='missingidentity':rows[1]['identity']=None
    elif bad=='unknown':rows[1]['resources']['predict_calls']=m.empty_vector()['predict_calls']
    elif bad=='cap':rows[1]['resources']['predict_calls'].update(attempted=7)
    elif bad=='wallunknown':rows[1]['wall'].update(status='unknown',seconds=None)
    elif bad=='wallscope':rows[1]['wall']['scope']='whole_job'
    else:rows[1]['wall']['seconds']=4.
    v=m.validate_shared_panel(rows,[r['slot_id'] for r in rows],['predict_calls'],dict(predict_calls=6),
        wall_scope='search',wall_cap_seconds=3.)
    assert v['status']=='blocked' and v['scientific_admission'] is False


def test_fixed_denominator_no_missing_strategies():
    rows=shared_rows()
    with pytest.raises(ValueError):m.validate_shared_panel(rows[:1],[r['slot_id'] for r in rows],['solve'],{})
    with pytest.raises(ValueError):m.validate_shared_panel(rows,[r['slot_id'] for r in rows],['imagined_FLOPs'],{})
    with pytest.raises(ValueError):m.validate_shared_panel(rows,[r['slot_id'] for r in rows],['solve'],{},wall_scope='search')


def p2_fixture():
    order=[(0,'joint'),(0,'composition'),(0,'grasp'),(1,'grasp'),(1,'composition'),(1,'joint')]
    report=dict(schema=m.P2+'.report',fixed_denominator=6,
        rows=[dict(index=i,root_slot=s,arm=a) for i,(s,a) in enumerate(order)],timings=dict(total_seconds=20.))
    receipt=dict(schema=m.P2+'.consume',status='all6_common_cap_records_verified_not_admitted',fixed_denominator=6,
        resources=[p2_resource() for _ in order],accounts=[dict(elapsed_seconds=2.) for _ in order],
        inherited_encoding_cohort=dict(encoder=16,projector=16,off_allocation=None))
    return report,receipt


def test_existing_P2_saved_consumer_and_inherited_once():
    r,c=p2_fixture();v=m.p2_saved(r,c)
    assert len(v['rows'])==6 and v['full_strict_rebuilt_here'] is False
    assert v['inherited_once']['encoder']==16 and v['inherited_once']['off_allocation'] is None
    assert all(x['identity'] is None for x in v['rows'])


def p3_fixture():
    arms=('commit15_endpoints','commit15_midpoints','feedback5_endpoints','feedback5_midpoints')
    rows=[dict(episode_id=f's{s}_{a}',source_slot=s,mode=a,control_returned=20) for s in (0,1) for a in arms]
    nodes=[dict(episode_id=r['episode_id'],boundary=b,resource_receipt=ledger()) for r in rows for b in (5,10,15)]
    counts={k:dict(attempted=v,completed=v) for k,v in dict(encoder=4,projector=4,public_step=30,
        physics_step=304,explicit_render=32,reset=2,goal_bind=1).items()}
    r=dict(schema=m.P3+'.report',fixed_denominator=8,node_denominator=24,rows=rows,nodes=nodes,
        counts={},per_episode_neural=[],wall_seconds=100.)
    c=dict(schema=m.P3+'.consume',fixed_denominator=8,node_denominator=24,complete=True,counts={},per_episode_neural=[],
        physical_census=dict(per_episode={r['episode_id']:deepcopy(counts) for r in rows}))
    return r,c


def test_existing_P3_saved_consumer_all8_and_earlystop_unknown_wall():
    r,c=p3_fixture();r['rows'][0]['control_returned']=14;r['nodes'][2]['resource_receipt']=None
    v=m.p3_saved(r,c);assert len(v['rows'])==8
    assert v['rows'][0]['resources']['predict_calls']['completed']==2
    assert v['rows'][0]['control_returned']==14 and v['rows'][0]['wall']['status']=='unknown'
    assert v['rows'][0]['resources']['encode_calls']['completed']==4
    assert v['rows'][0]['resources']['encode_samples']['status']=='unknown'  # no fabricated visual batch
    assert v['rows'][0]['resources']['RGB_feedback']['completed']==4


def ref(path,value):path.write_bytes(m.raw(value));return dict(path=str(path),sha256=m.sha(path))


def test_saved_manifest_real_hashes_no_producer_import_or_fullstrict(tmp_path):
    r,c=p2_fixture();rr=ref(tmp_path/'report.json',r);c['report_sha256']=rr['sha256']
    cr=ref(tmp_path/'receipt.json',c)
    source=m.ROOT/'experiments/spectral_reachability'/(m.P2+'.py')
    manifest=dict(schema=m.NAME+'.inputs',kind='P2_common',report=rr,receipt=cr,
        sources=[dict(path=str(source),sha256=m.SOURCE_PINS[m.P2])])
    v=m.build_manifest(manifest)
    assert v['full_strict_rebuilt_here'] is False and v['total_FLOPs']=='unknown'
    cr['sha256']='0'*64
    with pytest.raises(ValueError):m.build_manifest(manifest)


def test_hash_duplicate_json_and_negative_zero_wall(tmp_path):
    path=tmp_path/'duplicate.json';path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError):m.read_bound(dict(path=str(path),sha256=m.sha(path)))
    assert m.wall('scope',0.,[],[])['status']=='known'
    with pytest.raises(ValueError):m.wall('scope',True,[],[])
    with pytest.raises(ValueError):m.wall('scope',float('nan'),[],[])


def test_no_old_execution_modules_imported():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules; assert not any("agent_p1_" in n or "agent_p2_" in n or "agent_p3_" in n for n in sys.modules)'
    subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,check=True)
