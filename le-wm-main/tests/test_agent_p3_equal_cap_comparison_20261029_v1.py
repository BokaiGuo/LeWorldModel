"""Model-free saved ABI fixtures, never experimental evidence."""
from copy import deepcopy
import json
import subprocess
import sys
import pytest
from experiments.spectral_reachability import agent_p3_equal_cap_comparison_20261029_v1 as m


def node(value=0.):
    raw=[[value,0.]]*5
    return dict(depth=1,indices=[0],cost=1.,edges=[dict(kind='progress',actions=raw)],
        actions=raw,context={'fixture':True},path=[])


def bundle(new):
    name=m.NEW if new else m.OLD
    zero=dict.fromkeys(m.AXES,0);caps={k:1000 for k in m.AXES}
    profile=dict(resource_caps={k:deepcopy(caps) for k in ('fixed','adaptive')},timer_scope='RAM fixture')
    p=dict(schema=name+'.prepare',tasks=[{'root':0},{'root':1}],profile=profile,expected_model_hash='a'*64)
    rows=[];arms=[]
    for i in range(4):
        rr=dict(caps=deepcopy(caps),attempted=deepcopy(zero),completed=deepcopy(zero),events=[],wall_ns=10,
            inclusive_operation_wall_ns={},wall_semantics='nested inclusive; never sum into total/FLOPs',
            FLOPs=None,cuda_ns=None,backward_engine_calls=None,unknowns_are_not_zero=True,physics_steps=0)
        rr['attempted']['predict_API']=rr['completed']['predict_API']=2+(1 if new else 0)
        q=dict(horizon=1,candidates=4,candidate_index=0,program_type='progress',extension_type='progress',
            parent_context_sha256='b'*64,binding_sha256='c'*64,prefix_actions=[])
        jobs=[dict(slot=j,status='not_requested',request=None,payload=None,node=None,error=None) for j in range(144)]
        jobs[0].update(status='returned',request=q,node=node(),payload={'actions':[[0.,0.]]*5})
        core=dict(mode=('fixed','adaptive')[i%2],status='callback_progress_complete_NOT_certified',jobs=jobs,
            model_plan=node(),diagnostic_best=node(),realized_progress_depth=1,realized_widths=[4],queues=[],resource_receipt=rr,
            **{k:None for k in m.UNKNOWN})
        row=dict(slot=i,root=i//2,mode=core['mode'],status=core['status'],result=core)
        if new:row.update(plan_available=True,released_plan=node(),timing=None)
        rows.append(row)
        arms.append(dict(slot=i,root=i//2,mode=core['mode'],status=core['status'],resources=deepcopy(rr),
            realized_progress_depth=1,realized_widths=[4],model_plan_available=True,**{k:None for k in m.UNKNOWN}))
    flags={k:False for k in ('P1_admission','P3_admission','freshness','equal_FLOPs')}
    r=dict(schema=name+'.report',rows=rows,errors=[],sources_verified=True,model_before={'fixture':'a'},wall_seconds=1.,**flags)
    s=dict(schema=name+'.consume',status='saved_equal_cap_verified_not_admitted' if new else 'saved_four_slot_engineering_verified',
        arms=arms,slot_statuses=[x['status'] for x in rows],**flags)
    if new:s['fixed_denominator']=dict(arms=4,requests_per_arm=144,total_requests=576)
    else:s['denominator']=4
    return p,r,s


def save_bundle(tmp,label,b):
    p,r,s=deepcopy(b);refs={}
    for kind,obj in (('prepare',p),('report',r),('receipt',s)):
        path=tmp/(label+'_'+kind+'.json')
        if kind in ('report','receipt'):obj.update(prepared=refs['prepare']['path'],prepare_sha256=refs['prepare']['sha256'])
        if kind=='receipt':obj['report_sha256']=refs['report']['sha256']
        path.write_bytes(m.canonical(obj));refs[kind]=dict(path=str(path),sha256=m.sha(path))
    return refs


def inputs(tmp,old=None,new=None):
    return {k:save_bundle(tmp,k,b) for k,b in (('old',old or bundle(False)),('new',new or bundle(True)))}


def test_full576_saved_read_and_independent27_delta(tmp_path):
    result=m.analyze(inputs(tmp_path))
    assert len(result['arms'])==4 and sum(len(a['jobs']) for a in result['arms'])==576
    for a in result['arms']:
        assert set(a['delta_new_minus_old']['attempted'])==set(m.AXES)
        assert a['delta_new_minus_old']['attempted']['predict_API']==1
        assert a['same_released_action_bits'] is True
        assert a['request_status_census']['new']=={'returned':1,'not_requested':143}
        assert all(a['new']['unknown'][k] is None for k in m.UNKNOWN)
    assert not result['budget_effectiveness_identified'] and not result['equal_actual_resources']


def test_missing_whole_arm_keeps144_null_and_deltas_unknown(tmp_path):
    b=bundle(True);b[1]['rows'][3].update(result=None,status='blocked_after_runtime_failure',released_plan=None,plan_available=False)
    b[2]['arms'].pop();b[2]['slot_statuses'][3]='blocked_after_runtime_failure'
    b[2]['status']='saved_equal_cap_partial_verified_not_admitted'
    a=m.analyze(inputs(tmp_path,new=b))['arms'][3]
    assert len(a['jobs'])==144 and all(x['new'] is None for x in a['jobs'])
    assert a['delta_new_minus_old'] is None and a['new']['resources'] is None
    assert a['request_status_census']['new']=={'unobserved':144}


def test_core_plan_is_not_driver_released_plan(tmp_path):
    b=bundle(True);b[1]['rows'][0].update(plan_available=False,released_plan=None)
    a=m.analyze(inputs(tmp_path,new=b))['arms'][0]
    assert a['new']['core_model_plan'] is not None and a['new']['released_plan'] is None
    assert a['same_released_action_bits'] is None


def test_float32_bits_signedzero_and_rounding_distinct():
    assert m.actions([[0.,1.]])!=m.actions([[-0.,1.]])
    a=m.actions([[1.,0.]]);b=m.actions([[1.+1e-10,0.]])
    assert a['float32_sha256']==b['float32_sha256'] and a['float64_sha256']!=b['float64_sha256']


@pytest.mark.parametrize('damage',('task','model','axes','counter','jobs','order','receipt_resources','receipt_geometry',
    'receipt_plan','receipt_flags','receipt_status','receipt_unknown','commoncap','release'))
def test_fail_closed_abi(tmp_path,damage):
    b=bundle(True);p,r,s=b
    if damage=='task':p['tasks'][0]['root']=99
    elif damage=='model':p['expected_model_hash']='f'*64
    elif damage=='axes':r['rows'][0]['result']['resource_receipt']['caps'].pop('grad_API')
    elif damage=='counter':r['rows'][0]['result']['resource_receipt']['attempted']['predict_API']=-1
    elif damage=='jobs':r['rows'][0]['result']['jobs'].pop()
    elif damage=='order':r['rows'].reverse()
    elif damage=='receipt_resources':s['arms'][0]['resources']['attempted']['predict_API']+=1
    elif damage=='receipt_geometry':s['arms'][0]['realized_progress_depth']=3
    elif damage=='receipt_plan':s['arms'][0]['model_plan_available']=False
    elif damage=='receipt_flags':s['P3_admission']=True
    elif damage=='receipt_status':s['status']='unverified'
    elif damage=='receipt_unknown':s['arms'][0]['scalar_backward_calls']=0
    elif damage=='commoncap':p['profile']['resource_caps']['fixed']['predict_API']+=1
    else:r['rows'][0]['released_plan']=node(.1)
    with pytest.raises(ValueError):m.analyze(inputs(tmp_path,new=b))


def test_external_hash_tamper_and_duplicate_json_rejected(tmp_path):
    refs=inputs(tmp_path);path=tmp_path/'new_receipt.json';path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='actual SHA'):m.analyze(refs)
    path.write_bytes(b'{"x":1,"x":2}')
    with pytest.raises(ValueError,match='duplicate'):m.load(dict(path=str(path),sha256=m.sha(path)))


def test_all_job_statuses_and_refused_H3_not_hidden(tmp_path):
    b=bundle(True);j=b[1]['rows'][0]['result']['jobs'][0]
    j.update(status='refused_unsupported_original_type',payload=None,node=None)
    j['request'].update(horizon=3,candidates=8,program_type='reconfiguration',extension_type='R')
    a=m.analyze(inputs(tmp_path,new=b))['arms'][0]
    assert a['jobs'][0]['new']['request']['horizon']==3
    assert a['jobs'][0]['same_extension_action_bits'] is None
    assert a['request_status_census']['new']['refused_unsupported_original_type']==1
    assert a['jobs'][0]['same_request'] is False


def test_cli_no_neural_or_producer_import():
    code=f'from experiments.spectral_reachability import {m.NAME} as m; import sys; assert "torch" not in sys.modules; assert not any("parent_native" in s for s in sys.modules); m.main(["--help"])'
    p=subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,text=True)
    assert p.returncode==0 and 'analyze' in p.stdout and 'verify' in p.stdout


def test_actual_writer_cli_verify_whole_summary(tmp_path,monkeypatch,capsys):
    refs=inputs(tmp_path);(tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    out=tmp_path/'results'/(m.NAME+'_fixture.json')
    args=['analyze','--output',str(out)]
    for side in ('old','new'):
        for kind in ('prepare','report','receipt'):
            args+=['--'+side+'-'+kind,refs[side][kind]['path'],'--'+side+'-'+kind+'-sha',refs[side][kind]['sha256']]
    m.main(args);receipt=json.loads(capsys.readouterr().out)
    assert receipt['sha256']==m.sha(out)
    m.main(['verify','--summary',str(out),'--summary-sha',m.sha(out)])
    assert json.loads(capsys.readouterr().out)['verified'] is True
    saved=json.loads(out.read_bytes());saved['arms'][0]['delta_new_minus_old']['completed']['predict_API']=999
    out.write_bytes(m.canonical(saved))
    with pytest.raises(ValueError,match='recomputation'):m.main(['verify','--summary',str(out),'--summary-sha',m.sha(out)])
    with pytest.raises(ValueError,match='new own output'):m.write(out,saved)


def test_timing_scope_overrun_not_wall_matching(tmp_path):
    b=bundle(True)
    b[1]['rows'][0]['timing']=dict(scope='synchronized fixture',cap_seconds=3600,failed=True,elapsed_ns=3601000000000,
        synchronization_attempts=2,synchronization_returned=2,probes=[dict(status='failed',overrun_ns=1000000000)])
    b[1]['rows'][0].update(released_plan=None,plan_available=False)
    t=m.analyze(inputs(tmp_path,new=b))['arms'][0]['new']['timing']
    assert t['max_overrun_ns']==1000000000 and t['failed_probe_count']==1 and 'probes' not in t
