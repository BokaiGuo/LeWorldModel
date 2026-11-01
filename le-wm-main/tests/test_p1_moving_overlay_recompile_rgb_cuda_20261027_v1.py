"""RAM/NumPy arithmetic only: no prepare, torch, model, renderer or physics."""
from copy import deepcopy
import hashlib
import io
import json
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_overlay_recompile_rgb_cuda_20261027_v1 as r


def fixture():
    exports=[]
    for s in (0,1):
        state=[float(s)]*7;goal=[float(s+1)]*7
        root=r.digest(np.asarray(state+goal,dtype=np.float64).tobytes())
        exports.append(dict(source_slot=s,old_root_sha256=root,artifacts={
            c:dict(path=f's{s}_{c}_blind_inputs.npz',sha256=r.digest(b'npz'),goal_pixel_sha256=r.digest(np.full((224,224,3),s*2+i,np.uint8).tobytes()),
                   rgb_history_pixel_sha256=['a'*64,'b'*64,r.digest(np.zeros((224,224,3),np.uint8).tobytes())],
                   raw_action_sha256='c'*64,fields=['rgb_history','raw_action_history','goal_rgb'])
            for i,c in enumerate(('off','on'))}))
    src=dict(schema=r.SOURCE_NAME+'.report',status='paired_moving_overlay_recorded',source_verified=True,
             full_protocol=True,errors=[],blind_exports=exports,
             contract=dict(flags=[False,True,False],no_goal_pose_edit=True))
    plans=[];rows=[];events=[]
    for i,(s,c,a) in enumerate(r.KEYS):
        e=exports[s]['artifacts'][c]
        plan=dict(root_slot=s,condition=c,arm=a,selection_digest=f'{i+1:064x}',task_id=f'{i+20:064x}',
                  commands=np.zeros((10,2),np.float32).tolist(),blind_path='/source/'+e['path'],blind_sha256=e['sha256'],
                  source_report_sha256=r.SOURCE_SHA,planner_report_sha256='d'*64)
        plans.append(plan)
        checkpoints=[dict(step=t,png=f'b{i}_{t}.png',file_sha256=r.digest(b'png'),
                          pixel_sha256=r.digest(np.zeros((224,224,3),np.uint8).tobytes()),
                          state=[float(s)]*7,snapshot=dict(state=[float(s)]*7)) for t in (0,5,10)]
        row=dict(index=i,root_slot=s,condition=c,arm=a,selection_digest=plan['selection_digest'],task_id=plan['task_id'],
                 status='horizon_completed',checkpoints=checkpoints,missing_checkpoints=[],initial_exact=True,
                 root_state=[float(s)]*7,goal=[float(s+1)]*7,goal_pixel_sha256=e['goal_pixel_sha256'],
                 first_error=None,secondary_errors=[])
        rows.append(row)
        for cp in checkpoints:
            events.append(dict(kind='fresh_observation',branch=i,root_slot=s,condition=c,arm=a,
                selection_digest=plan['selection_digest'],receipt=dict(cp,phase='plan',condition=c,overlay_index=1 if c=='on' else 0)))
        events.append(dict(kind='branch_closed',branch=i,root_slot=s,condition=c,arm=a,
                           selection_digest=plan['selection_digest'],status=row['status']))
    counts={k:dict(attempted=0,completed=0) for k in r.PHYSICAL_CAPS}
    prep=dict(schema=r.PHYSICAL_NAME+'.prepare',plans=plans,source_report_sha256=r.SOURCE_SHA,
              planner_report_sha256='d'*64,contract=dict(flags=[False,True,False],no_goal_pose_edit=True,checkpoints=[0,5,10]))
    report=dict(schema=r.PHYSICAL_NAME+'.report',status='moving_overlay_recompile_physical_recorded',rows=rows,
        source_report_sha256=r.SOURCE_SHA,planner_report_sha256='d'*64,fixed_denominator=8,P1_admission=False,
        inputs_final_verified=True,errors=[],counts=counts)
    events.insert(0,dict(kind='run_started',plans=plans,contract=prep['contract']))
    events.append(dict(kind='run_finished',status=report['status'],counts=counts,errors=[]))
    return report,prep,events,src


def assembled():
    args=fixture();p=r.assemble(*args,source_base='/source');p['expected_model_hash']=r.MODEL_TENSOR_SHA
    p['shapes']=r.config_shapes(config())
    return p


def config():
    return dict(encoder=dict(image_size=224,patch_size=14,pretrained=False),
                projector=dict(input_dim=192,output_dim=192))


class MemoryStore(r.Artifacts):
    def __init__(self):self.files={};self.hashes={};self.order=[]
    def check(self,*a,**k):return None
    def write(self,name,raw,final=False):
        assert name not in self.files
        self.files[name]=raw;self.hashes[name]=r.digest(raw);self.order.append(name)
        return self.hashes[name]


def loader(item):
    n=item['root_slot']*2+(item['condition']=='on') if item['kind']=='goal' else 0
    return np.full((224,224,3),n,np.uint8),b'npz' if item['kind']=='goal' else b'png'


def callback(store,fail=None):
    def encode(pixels,index):
        if index==fail:raise ValueError('RAM injected failure')
        value=np.full((192,),index,np.float32)
        receipts=[]
        for kind in ('encoder','projector'):
            shape_in=(1,3,224,224) if kind=='encoder' else (1,192)
            shape_out=(1,257,192) if kind=='encoder' else (1,192)
            a=store.array(f'm{index}_{kind}_in',np.full(shape_in,index,np.float32))
            b=store.array(f'm{index}_{kind}_out',np.full(shape_out,index,np.float32))
            receipts.append(dict(image=index,resource=kind,completed=True,inputs=[a],output=b))
        receipts[0]['preprocessed']=store.array(f'm{index}_transform',np.full((1,1,3,224,224),index,np.float32))
        return value,receipts
    return encode


def finish():return dict(model_hash_before=r.MODEL_TENSOR_SHA,model_hash_after=r.MODEL_TENSOR_SHA,
                         source_verified=True,cuda_verified=False,TF32=False,autocast=False,SDP='MATH',blocked=[])


def test_full_denominator_conditional_goals_and_exact28():
    p=assembled()
    assert len(p['rows'])==8 and len(p['images'])==28
    assert [(x['root_slot'],x['condition']) for x in p['images'][:4]]==[(s,c) for s in (0,1) for c in ('off','on')]
    assert sum(x['kind']=='checkpoint' for x in p['images'])==24
    assert p['contract']['caps']==dict(encoder=28,projector=28)


@pytest.mark.parametrize('mutation',['condition','selection','goal','root','duplicate_row','step_bool','snapshot','missing_lie','event_condition'])
def test_bad_join_rejected(mutation):
    report,prep,events,src=fixture()
    if mutation=='condition':report['rows'][0]['condition']='on'
    elif mutation=='selection':report['rows'][0]['selection_digest']='f'*64
    elif mutation=='goal':report['rows'][0]['goal_pixel_sha256']=src['blind_exports'][0]['artifacts']['on']['goal_pixel_sha256']
    elif mutation=='root':report['rows'][0]['root_state'][0]=999
    elif mutation=='duplicate_row':report['rows'][1]=deepcopy(report['rows'][0])
    elif mutation=='step_bool':report['rows'][0]['checkpoints'][0]['step']=False
    elif mutation=='snapshot':report['rows'][0]['checkpoints'][0]['snapshot']['state'][0]=999
    elif mutation=='missing_lie':report['rows'][0]['missing_checkpoints']=[10]
    else:events[1]['receipt']['condition']='on'
    with pytest.raises(ValueError):r.assemble(report,prep,events,src,source_base='/source')


def test_missing_checkpoint_retains_three_steps_and_all8():
    report,prep,events,src=fixture()
    report['rows'][0]['checkpoints'].pop();report['rows'][0]['missing_checkpoints']=[10]
    events=[e for e in events if not (e['kind']=='fresh_observation' and e['branch']==0 and e['receipt']['step']==10)]
    p=r.assemble(report,prep,events,src,source_base='/source');store=MemoryStore()
    p['shapes']=r.config_shapes(config())
    out=r.evaluate(p,store,callback(store),loader,finish)
    assert len(out['rows'])==8 and len(out['rows'][0]['checkpoints'])==3
    cp=out['rows'][0]['checkpoints'][2]
    assert cp['latent'] is None and cp['Q'] is None and cp['status']=='missing_physical_checkpoint'
    assert len(out['images'])==27


def test_latents_and_raw_modules_sealed_before_q_and_no_dedup():
    p=assembled();store=MemoryStore();out=r.evaluate(p,store,callback(store),loader,finish)
    assert len(out['images'])==28
    assert out['counts']=={k:dict(attempted=28,completed=28) for k in ('encoder','projector')}
    seal=store.order.index('encoding_seal.json');metrics=store.order.index('metrics.json')
    assert seal<metrics and all(store.order.index(i['latent']['file'])<seal for i in out['images'])
    # Four goals are separate even though RAM mock pixels are all equal.
    assert len({i['latent']['file'] for i in out['images'][:4]})==4
    assert out['rows'][0]['checkpoints'][0]['Q']==16.0
    assert out['P1_admission'] is False and out['prediction_status']=='not_recomputed_encoding_only'


def test_failure_keeps_all_rows_and_no_q_or_retry():
    p=assembled();store=MemoryStore();out=r.evaluate(p,store,callback(store,fail=6),loader,finish)
    assert len(out['rows'])==8 and len(out['images'])==28
    assert out['images'][6]['status']=='failed' and out['images'][7]['status']=='not_started'
    assert all(c['Q'] is None for row in out['rows'] for c in row['checkpoints'])
    assert 'encoding_seal.json' in store.files and 'metrics.json' not in store.files


def test_mutated_model_blocks_q():
    p=assembled();store=MemoryStore()
    def drift():return dict(finish(),model_hash_after='f'*64)
    out=r.evaluate(p,store,callback(store),loader,drift)
    assert not out['model_unchanged'] and 'metrics.json' not in store.files


def test_wall_and_budget_pre_dispatch(monkeypatch):
    b=r.Budget(clock=lambda:0,free=lambda:r.FLOOR+r.CAP)
    b.started=-601
    with pytest.raises(ValueError):b.check(1)
    b.started=0;b.used=r.CAP
    with pytest.raises(ValueError):b.check(1)
    b.used=0;b.free=lambda:r.FLOOR-1
    with pytest.raises(ValueError):b.check(1)


def test_no_unauthorized_run_or_fake_future_prepare(monkeypatch):
    monkeypatch.setattr(r,'verify',lambda *a:pytest.fail('not before authorization'))
    with pytest.raises(ValueError):r.run('/no','a'*64,'/new',authorization=None)
    with pytest.raises(ValueError):r.build('/not-created',None)


def test_consume_rejects_ram_only_receipt():
    store=MemoryStore();out=r.evaluate(assembled(),store,callback(store),loader,finish)
    with pytest.raises(ValueError):r.consume_report(out,lambda name:store.files[name])


def test_native_array_receipt_readback_and_tamper():
    store=MemoryStore();a=np.arange(192,dtype=np.float32);receipt=store.array('original',a)
    assert np.array_equal(r.load_array(receipt,lambda name:store.files[name]),a)
    bad=dict(receipt,array_sha256='f'*64)
    with pytest.raises(ValueError):r.load_array(bad,lambda name:store.files[name])


def test_trace_rejects_cpu_and_extra_calls_without_neural_execution():
    store=MemoryStore();trace=r.Trace(None,None,store,lambda:None)
    cpu=type('MockCPU',(),{'device':type('Device',(),{'type':'cpu'})()})()
    with pytest.raises(ValueError):trace.save(cpu,'cpu')
    trace.entries=[dict(image=0,resource='encoder',completed=True)]
    with pytest.raises(ValueError):trace.before('encoder',0)


def test_help_and_no_torch_import(capsys):
    assert 'torch' not in sys.modules
    with pytest.raises(SystemExit):r.main(['--help'])
    assert '--physical-sha' in capsys.readouterr().out


def test_config_derived_257_not197():
    shape=r.config_shapes(config())
    assert shape['hidden']==[1,257,192]
    changed=config();changed['encoder']['patch_size']=16
    with pytest.raises(ValueError):r.config_shapes(changed)


@pytest.mark.parametrize('mutation',['197','cls','projected','dtype','image','transform'])
def test_actual_shape_cls_chain_rejects_fabrication(mutation):
    store=MemoryStore();z,receipts=callback(store)(np.zeros((224,224,3),np.uint8),0)
    if mutation=='197':receipts[0]['output']=store.array('bad_hidden',np.zeros((1,197,192),np.float32))
    elif mutation=='cls':receipts[1]['inputs']=[store.array('bad_cls',np.ones((1,192),np.float32))]
    elif mutation=='projected':receipts[1]['output']=store.array('bad_projected',np.ones((1,192),np.float32))
    elif mutation=='dtype':receipts[0]['output']=store.array('bad_dtype',np.zeros((1,257,192),np.uint8))
    elif mutation=='image':receipts[1]['inputs']=[store.array('bad_shape',np.zeros((192,),np.float32))]
    else:receipts[0]['preprocessed']=store.array('bad_transform',np.ones((1,1,3,224,224),np.float32))
    with pytest.raises(ValueError):r.validate_chain(receipts,z,r.config_shapes(config()),lambda n:store.files[n])


def saved_mock():
    # Schema testing only: this injected attestation is NOT real CUDA evidence.
    store=MemoryStore()
    out=r.evaluate(assembled(),store,callback(store),loader,lambda:dict(finish(),cuda_verified=True))
    return out,store


def test_consume_ram_schema_roundtrip_and28_no_dedup():
    out,store=saved_mock();rows=r.consume_report(out,lambda n:store.files[n])
    assert len(rows)==8 and rows[0]['checkpoints'][0]['Q']==16.
    assert rows[0]['selection_digest']==out['rows'][0]['selection_digest']


@pytest.mark.parametrize('mutation',['counts','cross_arm','false_missing','duplicated_image','entry_count','tf32'])
def test_consume_rejects_inconsistent_saved_schema(mutation):
    out,store=saved_mock()
    if mutation=='counts':out['counts']['encoder']['attempted']=27
    elif mutation=='cross_arm':out['rows'][0]['checkpoints']=deepcopy(out['rows'][1]['checkpoints'])
    elif mutation=='false_missing':out['rows'][0]['checkpoints'][0].update(image=None,latent=None,Q=None)
    elif mutation=='duplicated_image':out['rows'][1]['checkpoints']=deepcopy(out['rows'][0]['checkpoints'])
    elif mutation=='entry_count':
        seal=r.decode(store.files['encoding_seal.json']);seal['entries'].pop();raw=r.canonical(seal)
        store.files['encoding_seal.json']=raw;out['artifact_sha256']['encoding_seal.json']=out['encoding_seal_sha256']=r.digest(raw)
    else:out['TF32']=True
    with pytest.raises(ValueError):r.consume_report(out,lambda n:store.files[n])


def test_deadline_still_allows_bounded_final_error_receipt():
    b=r.Budget(clock=lambda:601,free=lambda:r.FLOOR+r.CAP);b.started=0
    with pytest.raises(ValueError):b.check(1)
    b.check(1,final=True)


def test_initialization_failure_retains8_rows_and_all_null_q():
    store=MemoryStore()
    def failed(_):raise RuntimeError('RAM CUDA initialization failed')
    out=r.evaluate(assembled(),store,callback(store),failed,lambda:{})
    assert len(out['rows'])==8 and all(c['Q'] is None for row in out['rows'] for c in row['checkpoints'])
    assert out['counts']=={k:dict(attempted=0,completed=0) for k in ('encoder','projector')}
    assert out['status']=='partial_or_failed' and 'encoding_seal.json' in store.files


def test_no_checkpoints_retains4_goals_and24_unknowns():
    report,prep,events,src=fixture()
    for row in report['rows']:
        row.update(checkpoints=[],missing_checkpoints=[0,5,10],status='not_started')
        for key in ('root_state','goal','goal_pixel_sha256'):row.pop(key)
    events=[e for e in events if e['kind']!='fresh_observation']
    p=r.assemble(report,prep,events,src,source_base='/source');p['shapes']=r.config_shapes(config())
    store=MemoryStore();out=r.evaluate(p,store,callback(store),loader,finish)
    assert len(out['images'])==4 and len(out['rows'])==8
    assert all(c['latent'] is None and c['Q'] is None for row in out['rows'] for c in row['checkpoints'])


def test_physical_counters_join_without_nested_sum():
    report,prep,events,src=fixture()
    for resource in ('public_step','physics_step'):
        report['counts'][resource]=dict(attempted=1,completed=1)
        events[1:1]=[dict(kind=kind,resource=resource) for kind in ('operation_attempted','operation_completed')]
    p=r.assemble(report,prep,events,src,source_base='/source')
    assert p['physical_resources']['physics_step']['completed']==1
    events.pop(1)
    with pytest.raises(ValueError):r.assemble(report,prep,events,src,source_base='/source')


def test_hashchain_decode_binds_spaces_algorithm_tail_and_index():
    previous=None;lines=[]
    for i,kind in enumerate(('run_started','run_finished')):
        event=dict(index=i,previous_sha256=previous,kind=kind)
        event['event_sha256']=r.digest(json.dumps(event,sort_keys=True,allow_nan=False).encode());previous=event['event_sha256']
        lines.append(r.canonical(event))
    raw=b''.join(lines)
    assert len(r.decode_journal(raw,previous))==2
    with pytest.raises(ValueError):r.decode_journal(raw,'f'*64)
    with pytest.raises(ValueError):r.decode_journal(raw.replace(b'run_started',b'run_mutated'),previous)


def test_latent_q_not_computed_after_deadline_at_seal():
    store=MemoryStore();original=store.write
    def write(name,raw,final=False):
        result=original(name,raw,final)
        if name=='encoding_seal.json':store.check=lambda *a,**k:(_ for _ in ()).throw(ValueError('wall exhausted'))
        return result
    store.write=write
    out=r.evaluate(assembled(),store,callback(store),loader,finish)
    assert 'metrics.json' not in store.files and all(c['Q'] is None for row in out['rows'] for c in row['checkpoints'])


def moving_packet_bytes(raw):
    stream=io.BytesIO()
    np.savez(stream,rgb_history=np.zeros((3,224,224,3),np.uint8),
             goal_rgb=np.zeros((224,224,3),np.uint8),raw_action_history=raw)
    return stream.getvalue()


def fixedraw():
    return np.asarray([[.1,0]]*5+[[0,.1]]*5,dtype=np.float32)


def test_real_nonzero_moving_history_accepted_exactly():
    result=r.packet(moving_packet_bytes(fixedraw()))
    assert result['raw_action_history'].dtype==np.float32
    assert np.array_equal(result['raw_action_history'],fixedraw())


@pytest.mark.parametrize('mutation',['zero','reverse','scale100','scale001','single_step','float64'])
def test_moving_history_zero_order_scale_or_dtype_rejected(mutation):
    raw=fixedraw()
    if mutation=='zero':raw[:]=0
    elif mutation=='reverse':raw=raw[::-1]
    elif mutation=='scale100':raw=raw*np.float32(100)
    elif mutation=='scale001':raw=raw*np.float32(.01)
    elif mutation=='single_step':raw[[4,5]]=raw[[5,4]]
    else:raw=raw.astype(np.float64)
    with pytest.raises(ValueError):r.packet(moving_packet_bytes(raw))
