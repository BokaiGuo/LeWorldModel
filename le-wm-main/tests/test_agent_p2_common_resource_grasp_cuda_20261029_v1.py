"""RAM protocol/array fixtures only; no torch, neural cost or native physics."""
from copy import deepcopy
import gzip
import io
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_common_resource_grasp_cuda_20261029_v1 as m
from test_agent_p2_wallcapped_cuda_20261029_v1 import ram_arm
from test_agent_p2_three_program_20261028_v1 import slot


class Clock:
    def __init__(self):self.now=100.
    def __call__(self):return self.now


def task():
    return dict(root_context=m.grasp.contexts(np.zeros((3,192),np.float32),np.zeros((2,10),np.float32)),
        goal=[0.]*192,action_mean=[0.,0.],action_std=[1.,1.])


def literal_step(c,t,n,label):
    """Synthetic saved buffers, NOT a model/dynamics/cost implementation."""
    z=np.zeros((1,3,192),np.float32);h=np.zeros((1,2,10),np.float32);a=np.zeros((5,2),np.float32)
    c.event(dict(kind='forward_attempt',index=n,label=label,z=m.grasp.pack(z),h=m.grasp.pack(h),raw=m.grasp.pack(a)))
    ae=np.zeros((1,3,192),np.float32);out=np.zeros((3,192),np.float32)
    for k,(name,ins,value) in enumerate((('action_encoder',[np.zeros((1,3,10),np.float32)],ae),
        ('predictor',[z,ae],ae),('pred_proj',[ae.reshape(3,192)],out))):
        if k==1:c.event(dict(kind='predict_attempt',index=n))
        c.event(dict(kind='module_attempt',index=n,resource=name,inputs=[m.grasp.pack(x) for x in ins]))
        c.event(dict(kind='module_return',index=n,resource=name,output=m.grasp.pack(value)))
    c.event(dict(kind='predict_return',index=n,output=m.grasp.pack(ae)))
    c.event(dict(kind='forward_return',index=n,z=m.grasp.pack(z),h=m.grasp.pack(h)))


def ram_grasp(units=1,break_at=None,do_replay=False,root=0,base=100.):
    events=[];clock=Clock();clock.now=base;c=m.GControl(clock,events.append);c.memory(1024,2048)
    x=np.zeros(30,np.float64);v=np.zeros((1,2,192),np.float32);frac=np.asarray([1/3,2/3],np.float32).reshape(1,2,1)
    c.event(dict(kind='initialized',seed=m.grasp.seed(root),fraction=m.grasp.pack(frac),x=m.grasp.pack(x),v=m.grasp.pack(v)))
    mx=m.adam_reference.initial_memory(x);mv=m.adam_reference.initial_memory(v);n=0;completed=0;failure=None;export=None
    try:
        for phase,k,j in m.phases():
            if completed==units:
                if do_replay:break
                clock.now=base+450.
            c.unit()
            if phase=='sync' and j==0:mx=m.adam_reference.initial_memory(x)
            c.event(dict(kind='optimization_attempt',phase=phase,iteration=k,sync_step=j,x=m.grasp.pack(x),v=m.grasp.pack(v)))
            if break_at=='before_forward':raise RuntimeError('RAM injected execution failure')
            for kk in range(3):literal_step(c,task(),n,('lifted/' if phase=='joint' else 'sync/')+str(kk));n+=1
            if break_at=='backward_deadline':clock.now=base+450.
            c.event(dict(kind='backward_attempt',phase=phase,iteration=k,sync_step=j,loss=m.grasp.pack(np.array(0,np.float32)),
                dyn=m.grasp.pack(np.array(0,np.float32)) if phase=='joint' else None,
                dense=m.grasp.pack(np.array(0,np.float32)) if phase=='joint' else None))
            if break_at=='backward':raise RuntimeError('RAM injected backward failure')
            c.event(dict(kind='chart_backward',raw_gradient=m.grasp.pack(np.zeros((3,5,2),np.float32))))
            c.event(dict(kind='backward_return',gx=m.grasp.pack(x),gv=m.grasp.pack(v) if phase=='joint' else None))
            c.event(dict(kind='optimizer_attempt',phase=phase,iteration=k,sync_step=j))
            if break_at=='Adam':raise RuntimeError('RAM injected Adam failure')
            ax=deepcopy(mx);ax['step']+=1;av=deepcopy(mv);av['step']+=1
            c.event(dict(kind='optimizer_return',x_before=mx,v_before=mv if phase=='joint' else None,x_after=ax,
                v_after=av if phase=='joint' else None,proposal=m.grasp.pack(x),x=m.grasp.pack(x),v=m.grasp.pack(v)))
            mx=ax
            if phase=='joint':
                mv=av;c.event(dict(kind='noise_attempt'));c.event(dict(kind='noise',noise=m.grasp.pack(v),v=m.grasp.pack(v)))
            if phase=='sync' and j==9:
                c.event(dict(kind='joint_optimizer_reset',after=k+1,x=m.grasp.pack(x),v=m.grasp.pack(v)))
                mx=m.adam_reference.initial_memory(x);mv=m.adam_reference.initial_memory(v)
            c.finish(m.grasp.pack(x));completed+=1
        if do_replay:
            # Only full632 reaches natural replay under the production schedule.
            c.unit(replay=True)
            c.event(dict(kind='actions_frozen',x=m.grasp.pack(x),commands=m.grasp.pack(np.zeros((15,2),np.float32))))
            for label in ('whole','segmented'):
                for kk in range(3):literal_step(c,task(),n,label+'/'+str(kk));n+=1
            export=dict(logits=x.tolist(),contexts=[deepcopy(task()['root_context']) for _ in range(3)])
            c.event(dict(kind='export_complete',witness_sha256=m.old.digest(export),accepted=True))
    except m.Stop:
        if not c.a.in_unit:
            try:c.unit(replay=True)
            except m.Stop:pass
    except RuntimeError as exc:failure=exc
    account=c.seal(failure)
    result=dict(export=export,error=None if failure is None else dict(type=type(failure).__name__,message=str(failure)))
    return result,account,events


def test_schedule_is_budget_derived_and_not_750_relabel():
    phases=list(m.phases());assert len(phases)==632
    assert sum(p=='joint' for p,_,_ in phases)==532
    assert sum(p=='sync' for p,_,_ in phases)==100
    assert 3*len(phases)+6==1902 and 3*(533+100)+6>1902
    assert phases[-1]==('joint',531,None)
    assert len(m.ORDER)==6 and m.ORDER[3]==[1,'grasp']
    assert m.contract()['equal_FLOPs'] is False and m.contract()['total_FLOPs'] is None


@pytest.mark.parametrize('arm',['joint','composition'])
def test_reuse_original_paid_spectral_algorithm_and_partial_strict(arm):
    row,events,load=ram_arm(slot(),arm,stop_after=46)
    a=m.wall.validate_arm(row,events,slot(),load)
    v=m.resource_vector(arm,a,events)
    assert v['attempted']['predict_API']==a['attempted']['predict']
    assert v['attempted']['predict_samples']==a['attempted']['predict']
    assert v['attempted']['Jacobian']==3
    assert v['attempted']['scalar_backward']==0 and v['engine_internal_backward_calls'] is None


@pytest.mark.parametrize('arm',['joint','composition'])
def test_spectral_complete_schedule_costs_without_neural_calls(arm):
    assert sum(m.wall.protocol.unit_cost(s)['predict'] for s in m.wall.protocol.schedule(arm))==({'joint':1902,'composition':1062}[arm])


@pytest.mark.parametrize('units',[0,1,3,50,61])
def test_saved_grasp_complete_update_prefix_and_deadline(units):
    result,account,events=ram_grasp(units)
    assert m.validate_grasp(task(),0,events,result)==account
    assert account['completed_units']==units and account['stop']=='wall_cap'
    assert result['export'] is None
    v=m.resource_vector('grasp',account,events)
    assert v['attempted']['AD_primal_packet_steps']==3*units
    assert v['attempted']['predict_API']==3*units


@pytest.mark.parametrize('where',['before_forward','backward','Adam','backward_deadline'])
def test_true_partial_never_refunded_or_exported(where):
    result,account,events=ram_grasp(1,where)
    assert m.validate_grasp(task(),0,events,result)==account
    assert account['completed_units']==0 and account['incomplete_unit']
    assert result['export'] is None
    if where=='backward':
        assert account['attempted']['scalar_backward']==1 and account['completed']['scalar_backward']==0
    if where=='Adam':
        assert account['attempted']['Adam_step']==1 and account['completed']['Adam_step']==0
    if where=='backward_deadline':
        assert account['attempted']['packet']==3 and account['attempted']['scalar_backward']==0


def test_full_632_RAM_saved_replay_all1902_calls_and_no_old_strict(monkeypatch):
    monkeypatch.setattr(m.grasp,'consume',lambda *a:pytest.fail('old strict called'))
    monkeypatch.setattr(m.grasp,'validate_root',lambda *a:pytest.fail('old root strict called'))
    result,account,events=ram_grasp(632,do_replay=True)
    assert m.validate_grasp(task(),0,events,result)==account
    assert account['completed']['packet']==account['completed']['predict_API']==1902
    assert account['completed']['scalar_backward']==632 and account['completed']['virtual_noise']==532
    v=m.resource_vector('grasp',account,events)
    assert v['completed']['AD_primal_packet_steps']==1896 and v['completed']['pred_proj_rows']==5706
    assert result['export'] is not None


@pytest.mark.parametrize('mutation',['raw','normalizer','loss','gradient','Adam','commit','extra','clock','account','order'])
def test_resealed_semantic_mutations_rejected(mutation):
    result,account,events=ram_grasp(1)
    find=lambda k:next(e['payload'] for e in events if e['kind']=='g_native' and e['payload']['kind']==k)
    if mutation=='raw':find('forward_attempt')['raw']=m.grasp.pack(np.ones((5,2),np.float32)*.1)
    elif mutation=='normalizer':find('module_attempt')['inputs'][0]=m.grasp.pack(np.ones((1,3,10),np.float32))
    elif mutation=='loss':find('backward_attempt')['loss']=m.grasp.pack(np.array(1,np.float32))
    elif mutation=='gradient':find('backward_return')['gx']=m.grasp.pack(np.ones(30,np.float64))
    elif mutation=='Adam':find('optimizer_return')['x_after']['step']=2
    elif mutation=='commit':next(e for e in events if e['kind']=='g_commit')['x']=m.grasp.pack(np.ones(30,np.float64))
    elif mutation=='extra':events.insert(-1,dict(kind='unknown',now=100.))
    elif mutation=='clock':events[1]['now']=0.
    elif mutation=='account':events[-1]['summary']['attempted']['predict_API']=0
    else:events[3],events[4]=events[4],events[3]
    with pytest.raises((ValueError,KeyError)):m.validate_grasp(task(),0,events,result)


def test_pre_dispatch_cap_and_reserved6_no_padding():
    a=m.GAccount(0.);a.attempted['packet']=1894
    assert a.admit_unit(0.) is False and a.stop=='resource_cap'
    assert a.admit_unit(0.,True) is True
    a.attempted['predict_API']=1902
    assert not a.event(dict(kind='predict_attempt'),0.)
    assert a.attempted['predict_API']==1902


def test_api_and_samples_are_distinct_on_pending_predictor():
    a=m.GAccount(0.);assert a.admit_unit(0.)
    assert a.event(dict(kind='forward_attempt'),0.)
    assert a.event(dict(kind='predict_attempt'),0.)
    assert a.attempted['packet']==1 and a.attempted['predict_API']==1 and a.attempted['predict_samples']==0
    assert a.event(dict(kind='module_attempt',resource='predictor'),0.)
    assert a.attempted['predict_samples']==1 and a.completed['predict_samples']==0


def test_monotone_clock_and_real_memory_bounds():
    a=m.GAccount(10.)
    with pytest.raises(ValueError):a.memory(9.,0,0)
    assert not a.memory(10.,4*1024**3+1,6*1024**3)
    assert a.peak_allocated==4*1024**3+1
    with pytest.raises(ValueError):a.memory(11.,True,10)


def test_inflight_can_overrun_but_no_new_dispatch_or_late_snapshot():
    a=m.GAccount(0.);assert a.admit_unit(0.)
    assert a.event(dict(kind='optimizer_attempt'),1.)
    assert a.event(dict(kind='optimizer_return'),451.)
    assert not a.finish(451.) and not a.admit_unit(451.,True)
    assert a.completed['Adam_step']==1 and a.stop=='wall_cap'


def test_actual_original_task_source_join_and_tamper():
    p=m.grasp.inputs.parse(m.grasp.inputs.read_bound(m.grasp.inputs.UPSTREAM,m.grasp.inputs.UPSTREAM_SHA))
    ts=m.join_tasks(p['slots']);assert len(ts)==2
    p['slots'][0]['packet']['goal'][0]+=.001
    with pytest.raises(ValueError):m.join_tasks(p['slots'])


def test_auth_before_any_read_or_torch(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('read'))
    with pytest.raises(ValueError):m.run_cuda('no','no','no','no')


def test_real_store_grasp_partial_hashchain_disk_and_truncation(tmp_path):
    output=tmp_path/(m.NAME+'_RAM');store=m.Store(output,time.monotonic())
    result,account,events=ram_grasp(1,'Adam')
    for e in events:store.event(dict(kind='arm_event',index=2,payload=e))
    store.put('arm2.json.gz',dict(result=result,accounting=account));store.close()
    saved=list(m.journal(output));assert [x['payload'] for x in saved]==events
    assert m.validate_grasp(task(),0,[x['payload'] for x in saved],result)==account
    with pytest.raises(ValueError):m.Store(output,time.monotonic())
    with pytest.raises(ValueError):store._write('../escape',b'x')
    blob=(output/'events.jsonl.gz').read_bytes();(output/'events.jsonl.gz').write_bytes(blob[:-1])
    with pytest.raises(ValueError):list(m.journal(output))


def test_cold_import_cli_and_no_global_patch():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+' as m; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules; assert m.wall.protocol.WALL==450.; assert m.grasp.STEPS==750; print("inert")'
    out=subprocess.run([sys.executable,'-B','-c',code],check=True,capture_output=True,text=True)
    assert 'inert' in out.stdout
    p=subprocess.run([sys.executable,'-B','-m','experiments.spectral_reachability.'+m.NAME,'--help'],check=True,capture_output=True,text=True)
    assert 'prepare,verify,run,consume' in p.stdout


def test_new_three_files_present():
    assert all(p.is_file() for p in m.own_files())


@pytest.mark.parametrize('name,shapes',[
    ('action_encoder',[(2,3,10)]),('predictor',[(1,1,192),(1,1,192)]),
    ('pred_proj',[(6,192)]),('encoder',[(1,3,224,224)]),
])
def test_batch_abi_rejected_before_dispatch(name,shapes):
    with pytest.raises(ValueError):m.check_tensor_abi(name,shapes)


def saved_six(tmp_path,monkeypatch):
    policy=tmp_path/'policy';policy.mkdir();bindings={};files=[]
    for name in ('config.json','weights.pt'):
        f=policy/name;f.write_bytes(b'RAM bytes only; not loaded');h=m.old.sha(f);bindings[str(f)]=h
        files.append(dict(path=str(f),sha256=h,bytes=f.stat().st_size,hash_buffer_sha256=h,deserialize_buffer_sha256=h,same_buffer=True))
    p=dict(slots=[slot(0),slot(1)],root_arm_order=m.ORDER,expected_model_hash='RAM_MODEL',policy=str(policy),bindings_sha256=bindings)
    prepared=tmp_path/'prepared.json';prepared.write_bytes(m.old.raw({'RAM':'inert authority fixture'}));ph=m.old.sha(prepared)
    output=tmp_path/(m.NAME+'_RAM');store=m.Store(output,time.monotonic())
    store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=ph))
    for f in files:
        store.event(dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']))
        store.event(dict(kind='consumed_buffer_frozen',receipt=f))
    receipt=dict(schema=m.old.loader.NAME+'.loader_receipt',prepare_sha256=ph,adapter_source_sha256=m.old.sha(m.old.loader.__file__),
        legacy_loader_source_sha256=m.old.loader.native.SOURCES['experiments/checkpoint_compat.py'],files=files,device='cuda:0',
        remapped_keys=[],missing=[],unexpected=[],tensor_hash_before='RAM_MODEL',frozen_eval=True)
    store.put('loader.json',receipt);rows=[]
    for i,(root,arm) in enumerate(m.ORDER):
        if arm=='grasp':
            result,account,events=ram_grasp(1,'Adam',root=root,base=100.+1000*i)
            row=dict(index=i,root_slot=root,arm=arm,result=result,accounting=account)
            for e in events:store.event(dict(kind='arm_event',index=i,payload=e))
        else:row,_,_=ram_arm(p['slots'][root],arm,store=store,index=i)
        store.put(f'arm{i}.json.gz',row);store.event(dict(kind='arm_sealed',index=i,row_sha256=m.old.digest(row)))
        rows.append(dict(index=i,root_slot=root,arm=arm,file=f'arm{i}.json.gz'))
    store.close();guard=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,MATH=True,FLASH=False,MEM_EFFICIENT=False,CUDNN=False)
    report=dict(schema=m.NAME+'.report',prepared=str(prepared),prepare_sha256=ph,contract=m.contract(),
        rows=rows,fixed_denominator=6,errors=[],status='records_sealed',model_before='RAM_MODEL',model_after='RAM_MODEL',
        precision_before=guard,precision_after=guard,
        stage_cuda_peak_bytes=dict(allocated=1024,reserved=2048),
        timings=dict(loader_seconds=1.,final_identity_seconds=1.,total_seconds=3000.,initial_prepare_verify_seconds=1.,host_setup_seconds=1.,post_search_persistence_seconds=1.),
        artifact_sha256={f.name:m.old.sha(f) for f in output.iterdir()})
    store.put('report.json',report,final=True)
    monkeypatch.setattr(m,'verify',lambda *a:deepcopy(p))
    monkeypatch.setattr(m,'join_tasks',lambda *a:[task(),task()])
    monkeypatch.setattr(m.old.parent,'check_files',lambda *a:None)
    return output


def test_six_real_files_full_consume_and_resources(tmp_path,monkeypatch):
    output=saved_six(tmp_path,monkeypatch);v=m.consume(output,m.old.sha(output/'report.json'))
    assert v['plans']==[None]*6 and len(v['resources'])==6
    assert v['resources'][2]['attempted']['Adam_step']==1 and v['resources'][2]['completed']['Adam_step']==0
    assert v['fixed_denominator']==6 and v['contract']['P1_admission'] is False
    assert v['inherited_encoding_cohort']['off_allocation'] is None


@pytest.mark.parametrize('change',['model','rows','extra_file','source_hash','precision','time'])
def test_full_consumer_rejects_resealed_bad_report(tmp_path,monkeypatch,change):
    output=saved_six(tmp_path,monkeypatch);r=m.old.parent.overlay.read(output/'report.json')
    if change=='model':r['model_after']='changed'
    elif change=='rows':r['rows']=r['rows'][:-1]
    elif change=='extra_file':(output/'extra').write_bytes(b'bad')
    elif change=='source_hash':r['artifact_sha256']['loader.json']='0'*64
    elif change=='precision':r['precision_after']['matmul_TF32']=True
    else:r['timings']['loader_seconds']=-1.
    (output/'report.json').write_bytes(m.old.raw(r))
    with pytest.raises(ValueError):m.consume(output,m.old.sha(output/'report.json'))


def test_prepare_verify_complete_rebuild_exclusive_metadata(tmp_path,monkeypatch):
    p=dict(schema=m.NAME+'.prepare',protocol_prepared='RAM inert original authority',protocol_sha256='a'*64,
        slots=[slot(),slot(1)],contract=m.contract())
    calls=[]
    def build(a,b):calls.append((a,b));return deepcopy(p)
    monkeypatch.setattr(m,'build',build)
    path=tmp_path/(m.NAME+'_prepare_RAM.json');meta=m.write_new(path,p)
    assert m.verify(path,meta['sha256'])==p and calls==[(p['protocol_prepared'],p['protocol_sha256'])]
    with pytest.raises(ValueError):m.write_new(path,p)
    p['contract']['grasp_joint']=533
    with pytest.raises(ValueError):m.verify(path,meta['sha256'])


def test_actual_cli_prepare_verify_argument_order(tmp_path,monkeypatch,capsys):
    original=tmp_path/'original.json';original.write_bytes(b'RAM protocol bytes')
    p=dict(schema=m.NAME+'.prepare',protocol_prepared=str(original),protocol_sha256=m.old.sha(original),contract=m.contract())
    def build(path,sha):
        assert path==str(original) and sha==m.old.sha(original)
        return deepcopy(p)
    monkeypatch.setattr(m,'build',build)
    new=tmp_path/(m.NAME+'_cli_RAM.json')
    assert m.main(['prepare','--protocol-prepared',str(original),'--protocol-sha',m.old.sha(original),'--output',str(new)])==0
    assert m.main(['verify','--prepared',str(new),'--prepare-sha',m.old.sha(new)])==0
    assert '"verified":true' in capsys.readouterr().out


def test_nonzero_saved_adam_f32_and_f64_without_torch():
    for dtype in (np.float32,np.float64):
        x=np.full(30,.1,dtype);g=np.linspace(-.01,.01,30,dtype=dtype)
        before=m.adam_reference.initial_memory(x);am=m.adam_reference.native_lerp(np.zeros(30,dtype),g);aq=((1-.999)*g*g).astype(dtype)
        after=dict(step=1,m=m.grasp.pack(am),q=m.grasp.pack(aq));lr=.005
        proposal=(x-lr/(1-.9)*am/(np.sqrt(aq)/np.sqrt(1-.999)+1e-8)).astype(dtype)
        m.check_adam(before,after,x,g,proposal,lr)
        with pytest.raises(ValueError):m.check_adam(before,after,x,g,proposal+dtype(.001),lr)


@pytest.fixture(scope='module')
def actual_adam_prefix():
    """Explicitly requested actual arithmetic witnesses, never terminal scores."""
    ref=m.adam_reference;output=ref.ACTUAL_OUTPUT
    assert m.old.sha(output/'report.json')==ref.ACTUAL_REPORT_SHA
    report=m.grasp.readjson(output/'report.json')
    assert m.old.sha(output/'events.jsonl.gz')==report['artifact_sha256']['events.jsonl.gz']
    previous='0'*64;initialized=None;request=None;gradient=None;first=None;cases={}
    for i,line in enumerate(m.grasp.journal_lines(output/'events.jsonl.gz')):
        envelope=m.grasp.inputs.parse(line)
        assert envelope['index']==i and envelope['previous']==previous
        assert envelope['sha256']==m.grasp.digest({k:v for k,v in envelope.items() if k!='sha256'})
        assert m.grasp.raw(envelope)==line+b'\n';previous=envelope['sha256']
        payload=envelope['payload']
        if payload.get('root_slot')!=0:continue
        event=payload['event'];kind=event['kind']
        if kind=='initialized':initialized=event
        elif kind=='optimization_attempt':request=event
        elif kind=='backward_return':gradient=event
        elif kind=='optimizer_return':
            if first is None:first=(event,gradient)
            if request['phase']=='joint' and request['iteration'] in (2,10):
                cases[request['iteration']]=(event,gradient)
                if set(cases)=={2,10}:break
    assert initialized is not None and first is not None and set(cases)=={2,10}
    return initialized,first,cases


def test_actual_negative_zero_empty_memory_and_native_first_adam(actual_adam_prefix):
    init,(op,grad),_=actual_adam_prefix
    v=m.grasp.unpack(init['v'],(1,2,192),np.float32)
    memory=m.adam_reference.initial_memory(v)
    assert np.count_nonzero(np.signbit(m.grasp.unpack(memory['m'])))==185
    assert memory['m']['sha256']=='0b9c761b27a2e0b287c3fedf401af232fba0fae3783087b1f5da0b86568c17bc'
    m.equal(memory,op['v_before'],'actual EMPTY virtual memory exact signs')
    wrong=deepcopy(memory);wrong['m']=m.grasp.pack(np.zeros_like(v))
    with pytest.raises(ValueError):m.equal(wrong,op['v_before'],'signed zero mutation must fail')
    for key,lr in (('x',.005),('v',.1)):
        param=m.grasp.unpack(init[key]);g=m.grasp.unpack(grad['g'+key])
        proposal=m.grasp.unpack(op['proposal'] if key=='x' else op['v'])
        m.check_adam(op[key+'_before'],op[key+'_after'],param,g,proposal,lr)


def test_actual_complement_and_cuda_fma_cases_not_tolerance_repair(actual_adam_prefix):
    _,_,cases=actual_adam_prefix
    for iteration,key,index in ((2,'x',28),(10,'v',176)):
        op,grad=cases[iteration];before=m.grasp.unpack(op[key+'_before']['m']);g=m.grasp.unpack(grad['g'+key])
        after=m.grasp.unpack(op[key+'_after']['m'])
        actual=after.flat[index:index+1];want=m.adam_reference.native_lerp(before,g).flat[index:index+1]
        assert actual.tobytes()==want.tobytes()
        wrong=(before+.1*(g-before) if key=='x' else before+(1-.9)*(g-before)).flat[index:index+1]
        with pytest.raises(ValueError):m.rounding(actual,wrong,64)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_fresh_and_sync_resets_derive_signs_from_current_parameters(dtype):
    first=np.asarray([-2.,-0.,0.,3.],dtype);later=np.asarray([4.,-3.,-2.,-0.],dtype)
    a=m.adam_reference.initial_memory(first);b=m.adam_reference.initial_memory(later)
    assert a!=b and np.array_equal(m.grasp.unpack(a['m']),m.grasp.unpack(b['m']))
    assert m.grasp.unpack(b['m']).tobytes()==(later*dtype(0)).tobytes()


def test_arithmetic_helpers_and_installed_lerp_are_pinned_no_old_consume(monkeypatch):
    for path,sha in m.PINS.items():assert m.old.sha(path)==sha
    assert any(path.endswith('/ATen/native/Lerp.h') for path in m.PINS)
    monkeypatch.setattr(m.adam_reference,'consume',lambda *a:pytest.fail('old consume'))
    monkeypatch.setattr(m.adam_reference,'validate_root',lambda *a:pytest.fail('old validate_root'))
    result,account,events=ram_grasp(1)
    assert m.validate_grasp(task(),0,events,result)==account
