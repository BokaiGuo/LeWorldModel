"""RAM/schema fixtures only: never real authority or execution evidence."""
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_physical_commit_authority_bridge_20261029_v1 as m

fixture=runpy.run_path(str(m.ROOT/'tests/test_agent_p3_physical_commit_preconditions_20261029_v1.py'))['data']


def save(p,obj):
    p.write_bytes(m.canonical(obj));return dict(path=str(p),sha256=m.sha(p))


def physical(tmp):
    d=fixture(tmp);rows=[];ops=[]
    for i in range(4):
        a=deepcopy(d['anchor']);trace=a['prefix51']
        for x in trace:
            s=x['receipt']['step'];phase=x['receipt']['phase']
            x['receipt']['png']=Path(d['actual_history']['frames'][0 if phase=='warmup' and s==5 else 1 if phase=='warmup' and s==10 else 2]['png']).name
        cp={k:deepcopy(trace[-3]['receipt'][k]) for k in ('png','file_sha256','pixel_sha256','state','snapshot')};cp['step']=5
        rows.append(dict(index=i,root_slot=i//2,arm=('endpoints','midpoints')[i%2],root_sha256=a['root_sha256'],
            task_id=a['task_id'],condition='off',goal=a['goal'],goal_pose=a['goal_pose'],initial_exact=True,warmup_returned=10,
            prefix51=dict(complete=True,frames=trace),checkpoints=[cp],
            plan=dict(prefix_commands=a['prefix_raw5'],source_witness=dict(task=deepcopy(d['task'])))))
        for phase,n,raw in [('warmup',10,a['warm_raw10']),('plan',20,a['prefix_raw5']*4)]:
            for step in range(1,n+1):ops.append(dict(index=len(ops),branch=i,resource='public_step',status='returned',phase=phase,step=step,
                raw_bits=np.asarray(raw[step-1],np.float32).tobytes().hex()))
    ops.append(dict(index=len(ops),branch=0,resource='explicit_render',status='returned'))
    counts=dict(public_step=dict(attempted=120,completed=120),explicit_render=dict(attempted=1,completed=1))
    r=dict(schema=m.PHYS+'.report',status=m.PSTATUS,rows=rows,actual_operation_receipts=ops,counts=counts,
        errors=[],known_roots=2,fresh_roots=0,P1_admission=False,P3_admission=False,physical_release=False,freshness=False)
    rr=save(tmp/'physical.json',r)
    source=m.ROOT/'experiments/spectral_reachability'/(m.PHYS+'.py')
    s=dict(schema=m.PHYS+'.consume',status=m.PSTATUS,physical_report_sha256=rr['sha256'],rows=deepcopy(rows),counts=deepcopy(counts),
        bindings={rr['path']:rr['sha256'],str(source):m.PINS[m.PHYS]},saved_PNG_verified=1,all_four_physically_executed=True,
        errors=[],known_roots=2,fresh_roots=0,P1_admission=False,P3_admission=False,physical_release=False,freshness=False)
    args=dict(physical_report=rr,physical_receipt=save(tmp/'physical_consume.json',s),history_report=None,history_receipt=None,cp5_tasks=None,cp5_source=None)
    return args,r,s,d


def history(tmp,args,r,d):
    rows=[{} for _ in range(16)];sr=[{} for _ in range(16)];chains=[{} for _ in range(4)]
    for i in (0,2):
        j=4*i;z=np.asarray(d['task']['root_context']['latent_history'][-1],np.float32)
        buf=io.BytesIO();np.save(buf,z,allow_pickle=False);p=tmp/f'z{j}.npy.gz';p.write_bytes(gzip.compress(buf.getvalue()))
        meta=dict(file=p.name,sha256=m.sha(p),array_sha256=m.typed.io.hashlib.sha256(z.tobytes()).hexdigest(),dtype='float32',shape=[192])
        cp=r['rows'][i]['checkpoints'][0];src=deepcopy(cp);src['path']=str(tmp/cp['png'])
        raw=np.zeros((10,2),np.float32);tokens=m.typed.tokens(raw,np.asarray(d['task']['action_mean'],np.float32),np.asarray(d['task']['action_std'],np.float32)).reshape(2,10)
        ctx=deepcopy(d['task']['root_context']);ctx['action_history']=tokens.tolist()
        rows[j]=dict(index=j,plan_index=i,root_slot=i//2,checkpoint=5,arm='endpoints',status='encoded',source=src,
            causal_raw10=raw.tolist(),causal_raw10_bits=raw.tobytes().hex(),normalized_action_history=tokens.tolist(),task_binding=d['task']['binding'],
            model_hash=d['task']['binding']['model_sha256'],latent=meta,observed_anchor192=z.tolist())
        sr[j]=dict(index=j,plan_index=i,checkpoint=5,latent_missing=False,observed_anchor_reencode_MSE=0,actual192=z.tolist())
        chains[i]=dict(plan_index=i,root_slot=i//2,bank='endpoints',checkpoints=[dict(step=5,context_missing=False,context=ctx)])
    hp=save(tmp/'history_prepare.json',dict(bindings={args[k]['path']:args[k]['sha256'] for k in ('physical_report','physical_receipt')}|
        {str(m.ROOT/'experiments/spectral_reachability'/(m.HIST+'.py')):m.PINS[m.HIST]}))
    hr=dict(schema=m.HIST+'.report',status='terminal_encoded_science_blocked',rows=rows,prepared=hp['path'],prepare_sha256=hp['sha256'],
        errors=[],blocked=[],source_verified=True,model_unchanged=True,model_hash_before=d['task']['binding']['model_sha256'],
        model_hash_after=d['task']['binding']['model_sha256'],counts={'encoder':16},model_inventory_before=[dict(path=m.typed.__file__,sha256=m.sha(m.typed.__file__))],
        P1_admission=False,P3_admission=False,physical_release=False)
    args['history_report']=save(tmp/'history.json',hr)
    hs=dict(schema=m.HIST+'.consume',status=m.HSTATUS,report_sha256=args['history_report']['sha256'],physical_report_sha256=args['physical_report']['sha256'],
        physical_receipt_sha256=args['physical_receipt']['sha256'],rows=sr,actual_context_chains=chains,counts=deepcopy(hr['counts']),
        P1_admission=False,P3_admission=False,physical_release=False)
    args['history_receipt']=save(tmp/'history_consume.json',hs)
    return hr,hs


def test_real_shape_physical_projection_blocked_not_ready(tmp_path):
    args,r,s,d=physical(tmp_path);out=m.build(args)
    assert out['status']=='BLOCKED' and out['typed_contract_partial']['authorities']['planner'] is None
    assert out['denominator']==dict(slots=4,known_roots=2,fresh_roots=0)
    assert out['strict_receipt_issued'] is False and not any(out[k] for k in m.typed.FLAGS)
    assert len(out['roots'])==2 and len(out['typed_blocked_manifest']['rows'])==4
    for root in out['roots']:
        assert root['anchor']==root['restore_reference']
        assert [f['absolute_step'] for f in root['frames']]==[5,10,15]
        assert root['anchor']['checkpoint']['plan_step']==5
        assert root['history'] is None
        assert root['source_projection']['checkpoint']['pointer'].endswith('/checkpoints/0')


def test_history_actual_npy_and_tokens_join_not_fresh3(tmp_path):
    args,r,s,d=physical(tmp_path);history(tmp_path,args,r,d);out=m.build(args)
    assert not out['all_three_frames_actual_verified']
    assert all(x['history']['history_tokens']==d['actual_history']['history_tokens'] for x in out['roots'])
    assert 'history.Nash_actual_cp5_tasks_and_frozen_source_join' in out['missing']
    assert 'history.actual_encoder_report_and_full_strict' not in out['missing']


@pytest.mark.parametrize('kind',['status','schema','sha','source','bool_png','census','rows','admission'])
def test_physical_attestation_mutations(tmp_path,kind):
    args,r,s,_=physical(tmp_path)
    if kind=='status':s['status']='passed'
    elif kind=='schema':s['schema']='caller.consume'
    elif kind=='sha':s['physical_report_sha256']='0'*64
    elif kind=='source':s['bindings'][str(m.ROOT/'experiments/spectral_reachability'/(m.PHYS+'.py'))]='0'*64
    elif kind=='bool_png':s['saved_PNG_verified']=True
    elif kind=='census':r['actual_operation_receipts'].pop()
    elif kind=='rows':s['rows'][0]['task_id']='different'
    else:r['P3_admission']=True
    with pytest.raises(ValueError):m.attest_physical(args,r,s)


@pytest.mark.parametrize('kind',['sequence','condition','renderstate','cp5','rawprefix','rawbox','path','signedzero'])
def test_anchor_mutations(tmp_path,kind):
    _,r,_,_=physical(tmp_path);row=r['rows'][0]
    if kind=='sequence':row['prefix51']['frames'][0]['receipt']['step']=1
    elif kind=='condition':row['prefix51']['frames'][0]['receipt']['condition']='on'
    elif kind=='renderstate':row['prefix51']['frames'][0]['post']['x']=1
    elif kind=='cp5':row['checkpoints'][0]['pixel_sha256']='0'*64
    elif kind=='rawprefix':row['plan']['prefix_commands'][0][0]=.5
    elif kind=='rawbox':r['actual_operation_receipts'][0]['raw_bits']=np.asarray([2,0],np.float32).tobytes().hex()
    elif kind=='path':row['checkpoints'][0]['png']='../wrong.png'
    else:row['checkpoints'][0]['state'][0]=-0.
    with pytest.raises(ValueError):m.physical_fragment(r,0,tmp_path)


@pytest.mark.parametrize('kind',['ancestry','tokens','raw','array','context','model','source'])
def test_history_mutations(tmp_path,kind):
    args,r,s,d=physical(tmp_path);hr,hs=history(tmp_path,args,r,d)
    if kind=='ancestry':hs['physical_report_sha256']='0'*64
    elif kind=='tokens':hr['rows'][0]['normalized_action_history'][0][0]+=.25
    elif kind=='raw':hr['rows'][0]['causal_raw10'][0][0]=.25
    elif kind=='array':hr['rows'][0]['latent']['array_sha256']='0'*64
    elif kind=='context':hs['actual_context_chains'][0]['checkpoints'][0]['context']['latent_history'][-1][0]+=.25
    elif kind=='model':hr['model_hash_after']='0'*64
    else:hr['rows'][0]['source']['pixel_sha256']='0'*64
    with pytest.raises(ValueError):
        m.attest_history(args,hr,hs,r,s)
        m.history_fragment(args,hr,hs,m.physical_fragment(r,0,tmp_path),0)


def test_no_arbitrary_cp5_status_is_authority():
    with pytest.raises(ValueError):m.join_cp5({},dict(schema='caller.tasks',status='ready'),[])


def cp5_fixture(tmp,args,r,d):
    hr,hs=history(tmp,args,r,d)
    roots=[m.physical_fragment(r,i,tmp) for i in (0,2)]
    for item,i in zip(roots,(0,2)):item['history']=m.history_fragment(args,hr,hs,item,i)
    p=save(tmp/'shared_prepare.json',dict(schema='agent_p3_shared_panel_20261029_v1.prepare'))
    tasks=[];provenance=[];nodes=[];audits=[]
    for root,item in enumerate(roots):
        t=deepcopy(d['task']);t['root_context']=deepcopy(item['history']['candidate_context'])
        t['binding']['context_sha256']=m.typed.core.digest(t['root_context'])
        witnesses=[]
        for mode in ('fixedH1_endpoints','fixedH1_midpoints','adaptive_endpoints','adaptive_midpoints'):
            eid=f's{root}_{mode}';req=dict(task=deepcopy(t),observation={'fixture':True})
            nodes.append(dict(episode_id=eid,boundary=5,request=req))
            audits.append(dict(episode_id=eid,boundary=5,node_verified=True))
            witnesses.append(dict(episode_id=eid,request_sha256=m.typed.core.digest(req),saved_task_binding=deepcopy(t['binding']),observation=deepcopy(req['observation'])))
        tasks.append(t);provenance.append(dict(source_slot=root,actual_cp5_absolute_steps=[5,10,15],actual_cp5_control_steps=[-5,0,5],
            local_task_sha256=m.typed.core.digest(t),original_task=deepcopy(d['task']),prefix5=deepcopy(item['anchor']['prefix_raw5']),
            raw10=deepcopy(item['history']['raw10']),branch_witnesses=witnesses))
    rr=save(tmp/'shared_report.json',dict(schema='agent_p3_shared_panel_20261029_v1.report',prepared=p['path'],prepare_sha256=p['sha256'],nodes=nodes))
    sr=save(tmp/'shared_consume.json',dict(schema='agent_p3_shared_panel_20261029_v1.consume',prepared=p['path'],prepare_sha256=p['sha256'],
        report_sha256=rr['sha256'],complete=True,status='saved_shared_panel_verified_NOT_admitted',audited=audits))
    source=m.ROOT/'experiments/spectral_reachability'/(m.CP5+'.py')
    args['cp5_source']=dict(path=str(source),sha256=m.sha(source))
    authorities=dict(prepared=p,report=rr,receipt=sr)
    result=dict(schema=m.CP5+'.tasks',status='known2_actual_cp5_saved_join_ready_NOT_admitted',tasks=tasks,provenance=provenance,
        authorities=authorities,bindings={v['path']:v['sha256'] for v in authorities.values()}|{str(source):m.sha(source)})
    return result,roots


def test_nash_actual_schema_join_preserves_planner_null(tmp_path):
    args,r,s,d=physical(tmp_path);result,roots=cp5_fixture(tmp_path,args,r,d)
    assert m.join_cp5(args,result,roots)
    assert all(x['history']['all_three_frames_actual_verified'] for x in roots)
    args['cp5_tasks']=save(tmp_path/'cp5_tasks.json',result)
    out=m.build(args)
    assert out['status']=='BLOCKED' and out['all_three_frames_actual_verified']
    assert out['typed_contract_partial']['authorities']['planner'] is None
    assert 'runtime.new_environment_cp5_pre_step6_guard' in out['missing']


@pytest.mark.parametrize('kind',['source','times','localhash','context','witness','goal','binding','raw','root','upstream'])
def test_nash_join_mutations_fail_closed(tmp_path,kind):
    args,r,s,d=physical(tmp_path);result,roots=cp5_fixture(tmp_path,args,r,d)
    if kind=='source':result['bindings'][args['cp5_source']['path']]='0'*64
    elif kind=='times':result['provenance'][0]['actual_cp5_absolute_steps'][0]=0
    elif kind=='localhash':result['provenance'][0]['local_task_sha256']='0'*64
    elif kind=='context':roots[0]['history']['candidate_context']['latent_history'][0][0]+=.25
    elif kind=='witness':result['provenance'][0]['branch_witnesses'][0]['request_sha256']='0'*64
    elif kind=='goal':roots[0]['candidate_base_task']['goal'][0]+=.25
    elif kind=='binding':result['bindings'].pop(result['authorities']['prepared']['path'])
    elif kind=='raw':result['provenance'][0]['raw10'][0][0]=.25
    elif kind=='root':result['provenance'][0]['source_slot']=1
    else:result['authorities']['receipt']['sha256']='0'*64
    with pytest.raises(ValueError):m.join_cp5(args,result,roots)


def test_path_sha_pair_and_partial_history_rejected(tmp_path):
    args,*_=physical(tmp_path);args['history_report']=args['physical_report']
    with pytest.raises(ValueError,match='all or neither'):m.read(args)


def test_input_byte_mutation(tmp_path):
    args,*_=physical(tmp_path);Path(args['physical_report']['path']).write_bytes(b'{}')
    with pytest.raises(ValueError):m.build(args)


def test_cli_new_only_writer(tmp_path,monkeypatch,capsys):
    args,*_=physical(tmp_path);(tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    # Source authentication remains real; only CLI output containment is rebased.
    result={'schema':m.NAME+'.bridge','status':'BLOCKED'}
    monkeypatch.setattr(m,'build',lambda a:result)
    out=tmp_path/'results'/(m.NAME+'_fixture.json');cli=[]
    for k in ('physical_report','physical_receipt'):
        cli+=['--'+k.replace('_','-'),args[k]['path'],'--'+k.replace('_','-')+'-sha',args[k]['sha256']]
    m.main(cli+['--output',str(out)])
    assert json.loads(out.read_bytes())==result and json.loads(capsys.readouterr().out)['sha256']==m.sha(out)
    with pytest.raises(ValueError,match='exclusive'):m.main(cli+['--output',str(out)])


def test_import_has_no_producer_or_neural_physics():
    code=f'from experiments.spectral_reachability import {m.NAME}; import sys; assert "torch" not in sys.modules and "pymunk" not in sys.modules; assert not any("actual_cp5_task_builder" in s for s in sys.modules)'
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,cwd=m.ROOT)
    assert p.returncode==0,p.stderr
