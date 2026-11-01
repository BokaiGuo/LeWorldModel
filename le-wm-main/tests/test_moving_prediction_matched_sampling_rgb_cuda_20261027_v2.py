"""RAM/mock only; no filesystem prepare, Torch, GPU, simulator or actual labels."""
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import moving_prediction_matched_sampling_rgb_cuda_20261027_v2 as m


def npz(slot=0,raw=None):
    f=BytesIO();np.savez_compressed(f,rgb_history=np.full((3,224,224,3),slot,np.uint8),
        goal_rgb=np.full((224,224,3),slot+4,np.uint8),raw_action_history=m.planner.old.moving.fixed_raw() if raw is None else raw)
    return f.getvalue()


def png(slot=0):
    from PIL import Image
    f=BytesIO();Image.fromarray(np.full((224,224,3),slot,np.uint8)).save(f,format='PNG');return f.getvalue()


def fixture():
    plans=[];rows=[];events=[];tasks=[];goals=[]
    for s in (0,1):
        rgb=np.full((224,224,3),s,np.uint8);goal=np.full((224,224,3),s+4,np.uint8)
        h=m.digest(rgb.tobytes());gh=m.digest(goal.tobytes());state=[float(s)]*7;target=[float(s+1)]*7
        tasks.append(dict(interface=dict(task_id=f'task{s}',gn_task=dict(goal_source_sha256=gh,task_contract_sha256='c'*64)),
            task_export=f'/task{s}',task_sha256=str(s)*64,blind_sha256=m.digest(npz(s)),
            task_contract=dict(pixel_sha256=[h]*3+[gh])))
        goals.append(dict(kind='goal',root_slot=s,path=f'/source/{s}.npz',file_sha256=m.digest(npz(s)),pixel_sha256=gh,
            packet_metadata=dict(rgb_history_pixel_sha256=[h]*3,raw_action_sha256=m.digest(m.planner.old.moving.fixed_raw().tobytes()))))
        for a in m.planner.ARMS:
            i=len(rows);root=m.physical.source.root_hash(np.asarray(state),np.asarray(target))
            plan=dict(root_slot=s,arm=a,task_id=f'task{s}',task_export=f'/task{s}',task_sha256=str(s)*64,blind_sha256=m.digest(npz(s)),
                task_contract_sha256='c'*64,root_sha256=root,commands=np.zeros((10,2),np.float32).tolist())
            plan['selection_digest']=m.physical.source.digest(plan);plans.append(plan)
            scoped=dict(branch=i,root_slot=s,arm=a,selection_digest=plan['selection_digest'])
            warm=m.planner.old.moving.fixed_raw();events.append(dict(kind='branch_started',**scoped,plan=plan,
                raw_warmup=warm.astype(float).tolist(),raw_float32_bits=warm.tobytes().hex()))
            cps=[]
            for phase in ('warmup','plan'):
                for k in range(11):
                    if phase=='warmup' and k:events.append(dict(kind='warmup_return',**scoped,step=k,raw_action=warm[k-1].astype(float).tolist(),raw_float32_bits=warm[k-1].tobytes().hex()))
                    if phase=='plan' and k:events.append(dict(kind='returned_transition',**scoped,phase=phase,step=k,command=plan['commands'][k-1],state=state,
                        terminated=False,truncated=False,returned_goal_state=target,returned_goal_rgb_sha256=gh))
                    c=dict(phase=phase,step=k,png=f'b{i}_{phase}_{k}.png',file_sha256=m.digest(png(s)),pixel_sha256=h,
                        state=deepcopy(state),snapshot=dict(state=deepcopy(state)),observation_state=deepcopy(state))
                    events.append(dict(kind='fresh_observation',**scoped,receipt=c))
                    events.append(dict(kind='guard_receipt',**scoped,phase=phase,step=k,accepted=True,render_preserved_dynamics=True))
                    if phase=='plan' and k in (0,5,10):cps.append(c)
                    if phase=='plan' and k==0:events.append(dict(kind='goal_attached',**scoped,goal=target,goal_rgb_sha256=gh,snapshot=c['snapshot']))
            row=dict(root_slot=s,arm=a,selection_digest=plan['selection_digest'],branch_index=i,selected_plan=plan,status='horizon_completed',
                initial_exact=True,full_horizon=True,own_attempted=10,own_returned=10,warmup_returned=10,first_error=None,secondary_errors=[],
                truncated=False,PNG=22,root_state=state,goal=target,final_state=state,checkpoints=cps,initial_metrics={'native_distance':1.},final_metrics={'native_distance':1.})
            rows.append(row);events.append(dict(kind='branch_closed',receipt=row))
    manifest=dict(schema=m.physical.NAME+'.selection',arms=list(m.planner.ARMS),selection_fixed_before_physics=True,
        physical_outcome_selection=False,wall_seconds=600,sourcepack_sha256=m.physical.SOURCE_SHA,
        roots=[dict(root_slot=s,root_sha256=plans[s*2]['root_sha256']) for s in (0,1)])
    contract=m.physical.contract();counts={k:dict(attempted=v,completed=v) for k,v in contract['caps'].items()}
    p=dict(schema=m.physical.NAME+'.prepare',plans=plans,contract=contract,sourcepack_sha256=m.physical.SOURCE_SHA,source_prepare_id='p')
    report=dict(schema=m.physical.NAME+'.report',status='matched_sampling_physical_recorded_not_admitted',errors=[],fixed_branch_denominator=4,
        P1_admission=False,contract=contract,sourcepack_sha256=m.physical.SOURCE_SHA,source_prepare_id='p',rows=rows,counts=counts)
    actual=[]
    for k,v in contract['caps'].items():
        for _ in range(v):
            pending=dict(operation_index=len(actual),resource=k,status='pending')
            returned=dict(pending,status='returned');actual.append(returned)
            events.extend([dict(kind='operation_dispatch_intent',receipt=pending),
                           dict(kind='operation_returned',receipt=returned)])
    report['actual_operation_receipts']=actual
    events.insert(0,dict(kind='run_started',prepare=p));events.append(dict(kind='run_finished',status=report['status'],counts=counts,errors=[]))
    for i,e in enumerate(events):e['index']=i
    return report,p,manifest,events,tasks,goals


def prepared():
    p=m.assemble(*fixture());p.update(shapes=m.arrays.config_shapes(dict(encoder=dict(image_size=224,patch_size=14,pretrained=False),
        projector=dict(input_dim=192,output_dim=192))),arguments={'external_report_sha':'a'*64},expected_model_hash=m.MODEL_TENSOR_SHA)
    return p


class RAMStore(m.arrays.Artifacts):
    def __init__(self):self.files={};self.hashes={};self.order=[]
    def check(self,*a,**kw):return None
    def write(self,name,raw,final=False):
        assert name not in self.files
        self.files[name]=raw;self.hashes[name]=m.digest(raw);self.order.append(name);return self.hashes[name]
    def blob(self,name):return self.files[name]


def loader(item):
    s=item['root_slot'];raw=npz(s) if item['kind']=='goal' else png(s)
    return np.full((224,224,3),s+4 if item['kind']=='goal' else s,np.uint8),raw


def encode(store,fail=None):
    def callback(pixels,i):
        if i==fail:raise RuntimeError('injected mock encode failure')
        z=np.full((192,),i,np.float32);entries=[]
        for j,kind in enumerate(('encoder','projector')):
            shapein=(1,3,224,224) if j==0 else (1,192);shapeout=(1,257,192) if j==0 else (1,192)
            e=dict(image=i,resource=kind,completed=False,inputs=[store.array(f'{i}_{kind}_input',np.full(shapein,i,np.float32))])
            if j==0:e['preprocessed']=store.array(f'{i}_preprocessed',np.full((1,1,3,224,224),i,np.float32))
            ordinal=2*i+j+1;store.put(f'module_{ordinal:02d}_intent.json',deepcopy(e))
            e.update(completed=True,output=store.array(f'{i}_{kind}_output',np.full(shapeout,i,np.float32)))
            store.put(f'module_{ordinal:02d}_returned.json',e);entries.append(e)
        return z,entries
    return callback


def finish(cuda=False):
    return dict(model_hash_before=m.MODEL_TENSOR_SHA,model_hash_after=m.MODEL_TENSOR_SHA,source_verified=True,cuda_verified=cuda,
        MATH=True,TF32=False,autocast=False,blocked=[])


def test_fixed14_actual_images_shared_goals_not28():
    p=prepared()
    assert len(p['images'])==14 and len(p['rows'])==4 and p['contract']['caps']==dict(encoder=14,projector=14)
    assert [r['goal_image'] for r in p['rows']]==[0,0,1,1]
    assert len({c['image'] for r in p['rows'] for c in r['checkpoints']})==12
    assert p['shapes']['hidden']==[1,257,192]


@pytest.mark.parametrize('mutation',['arm','denominator','missing','duplicate','boolstep','command','goal','history','guard','state','task','count','root'])
def test_source_joins_fail_closed(mutation):
    report,p,manifest,events,tasks,goals=fixture()
    if mutation=='arm':report['rows'][0]['arm']='other'
    elif mutation=='denominator':report['fixed_branch_denominator']=3
    elif mutation=='missing':report['rows'][0]['checkpoints'].pop()
    elif mutation=='duplicate':report['rows'][1]=deepcopy(report['rows'][0])
    elif mutation=='boolstep':report['rows'][0]['checkpoints'][0]['step']=False
    elif mutation=='command':next(e for e in events if e['kind']=='returned_transition')['command']=[.2,.1]
    elif mutation=='goal':goals[0]['pixel_sha256']='f'*64
    elif mutation=='history':next(e for e in events if e['kind']=='branch_started')['raw_float32_bits']='00'
    elif mutation=='guard':next(e for e in events if e['kind']=='guard_receipt')['accepted']=False
    elif mutation=='state':report['rows'][0]['final_state']=[9.]*7
    elif mutation=='task':tasks[0]['task_sha256']='f'*64
    elif mutation=='count':report['counts']['public_step']['completed']-=1
    else:manifest['roots'][0]['root_sha256']='e'*64
    with pytest.raises(ValueError):m.assemble(report,p,manifest,events,tasks,goals)


@pytest.mark.parametrize('raw',[np.zeros((10,2),np.float32),np.asarray([[0,.1]]*5+[[.1,0]]*5,np.float32),
    m.planner.old.moving.fixed_raw().astype(np.float64),m.planner.old.moving.fixed_raw()*100])
def test_zero_reordered_scale_dtype_history_refused(raw):
    with pytest.raises(ValueError):m.arrays.packet(npz(raw=raw))


def test_exact_moving_packet_and_hidden257():
    a=m.arrays.packet(npz());assert np.array_equal(a['raw_action_history'],m.planner.old.moving.fixed_raw())
    c=dict(encoder=dict(image_size=224,patch_size=14,pretrained=False),projector=dict(input_dim=192,output_dim=192))
    assert m.arrays.config_shapes(c)['hidden']==[1,257,192]
    c['encoder']['patch_size']=16
    with pytest.raises(ValueError):m.arrays.config_shapes(c)


def test_goal_not_dedup_by_equal_pixels():
    report,p,manifest,events,tasks,goals=fixture()
    # Same pixel hash, distinct source files: still separately called.
    goals[1]['pixel_sha256']=goals[0]['pixel_sha256'];tasks[1]['interface']['gn_task']['goal_source_sha256']=goals[0]['pixel_sha256']
    # Only exercise goal-provenance policy without making the physical goal joins inconsistent.
    for e in events:
        if e.get('root_slot')==1 and e['kind']=='goal_attached':e['goal_rgb_sha256']=goals[0]['pixel_sha256']
        if e.get('root_slot')==1 and e['kind']=='returned_transition':e['returned_goal_rgb_sha256']=goals[0]['pixel_sha256']
    out=m.assemble(report,p,manifest,events,tasks,goals)
    assert out['contract']['goals']==2 and len(out['images'])==14


def test_seal_before_Q_and_actual_module_counts():
    p=prepared();store=RAMStore();r=m.evaluate(p,store,encode(store),loader,finish)
    assert r['status']=='encoded_sealed' and r['counts']=={k:dict(attempted=14,completed=14) for k in ('encoder','projector')}
    assert store.order.index('encoding_seal.json')<store.order.index('metrics.json')
    assert all(cp['Q'] is not None for row in r['rows'] for cp in row['checkpoints'])
    assert all(x['status']=='encoded' for x in m.decode(store.files['encoding_seal.json'])['images'])
    with pytest.raises(ValueError):m.consume_report(r,p,store.blob)  # RAM cuda_verified false is NOT actual evidence.


@pytest.mark.parametrize('failure',['encode','source','model','blocked'])
def test_failure_keeps_fourrows_Q_null_no_metrics(failure):
    p=prepared();store=RAMStore();meta=finish()
    if failure=='source':meta['source_verified']=False
    if failure=='model':meta['model_hash_after']='f'*64
    if failure=='blocked':meta['blocked']=['predict']
    r=m.evaluate(p,store,encode(store,3 if failure=='encode' else None),loader,lambda:meta)
    assert r['status']=='partial_or_failed' and len(r['rows'])==4 and len(r['images'])==14
    assert all(cp['Q'] is None for row in r['rows'] for cp in row['checkpoints']) and 'metrics.json' not in store.files


def test_pure_saved_complete_consume_and_metric_tamper():
    p=prepared();store=RAMStore();store.put('execution_binding.json',{'prepared':'RAM','prepare_sha256':'a'*64})
    r=m.evaluate(p,store,encode(store),loader,lambda:finish(True))
    # Schema fixture only; injecting CUDA flag is never a host execution claim.
    out=m.consume_report(r,p,store.blob)
    assert len(out['rows'])==4 and not out['P1_admission']
    r=deepcopy(r);r['rows'][0]['checkpoints'][2]['Q']+=.1
    with pytest.raises(ValueError):m.consume_report(r,p,store.blob)


def test_budget_wall_cap_floor():
    clock=[0.];b=m.Budget(clock=lambda:clock[0],free=lambda:m.FLOOR+m.CAP)
    b.start_gate()
    b.check(m.CAP,final=True)
    with pytest.raises(ValueError):b.check(m.CAP)
    clock[0]=601.
    with pytest.raises(ValueError):b.check()
    b.check(final=True);b.free=lambda:m.FLOOR-1
    with pytest.raises(ValueError):b.check(final=True)


def test_start_gate_distinct_from_output_write_cap():
    b=m.Budget(clock=lambda:0,free=lambda:m.FLOOR+m.CAP-1)
    b.check()
    with pytest.raises(ValueError):b.start_gate()


def test_authority_and_run_authorization_fail_before_io():
    with pytest.raises(ValueError):m.run('no_prepare','a'*64,'no_output')
    with pytest.raises(ValueError):m.build('no_report','a'*64,'b'*64,'c'*64)


def test_trace_before_uses14_not28():
    trace=m.Trace(None,{},None,lambda:None,14)
    trace.entries=[dict(image=i,resource='encoder',completed=True) for i in range(14)];trace.image=14
    with pytest.raises(ValueError):trace.before('encoder',14)
    trace.entries=[];trace.fault=True
    with pytest.raises(ValueError):trace.before('encoder',0)


def test_MAIN_actual_planner_report_pinned_not_other_sampling():
    entry=dict(authority=m.planner.NAME,output=str(m.PLANNER_REPORT.parent),report_sha256=m.PLANNER_REPORT_SHA,source_sha256=m.PLANNER_SHA)
    p=dict(plans=[dict(compiler_authority=m.planner.NAME,compiler_report=str(m.PLANNER_REPORT),compiler_report_sha256=m.PLANNER_REPORT_SHA) for _ in range(4)])
    m.check_planner_authority(dict(authorities=[entry]),p)
    entry['report_sha256']='f'*64
    with pytest.raises(ValueError):m.check_planner_authority(dict(authorities=[entry]),p)


@pytest.mark.parametrize('fault',['intent','callback','output','returned','none'])
def test_trace_attempt_return_truth_on_persistence_fault(fault):
    store=RAMStore();shapes=prepared()['shapes'];calls=[]
    fake=NS(is_grad_enabled=lambda:False,cuda=NS(synchronize=lambda:None))
    trace=m.Trace(fake,shapes,store,store.check,14);trace.image=0
    trace.preprocessed=store.array('pre',np.zeros((1,1,3,224,224),np.float32))
    def save(tensor,name):
        if fault=='output' and name.endswith('_output'):raise OSError('output persistence')
        return store.array(name,tensor)
    trace.save=save
    real_put=store.put
    def put(name,value,final=False):
        if fault in ('intent','returned') and name.endswith('_'+fault+'.json'):raise OSError(fault+' persistence')
        return real_put(name,value,final)
    store.put=put
    def callback(x):
        calls.append(1)
        if fault=='callback':raise RuntimeError('native callback')
        return NS(last_hidden_state=np.zeros((1,257,192),np.float32))
    if fault=='none':trace.dispatch('encoder',np.zeros((1,3,224,224),np.float32),callback,np.zeros(1))
    else:
        with pytest.raises((OSError,RuntimeError)):trace.dispatch('encoder',np.zeros((1,3,224,224),np.float32),callback,np.zeros(1))
    assert len(calls)==(0 if fault=='intent' else 1)
    assert len(trace.entries)==len(calls)
    if trace.entries:assert trace.entries[0]['completed']==(fault!='callback')


def test_finalize_restore_failure_no_Q():
    store=RAMStore();p=prepared()
    def final():raise RuntimeError('restore failed')
    r=m.evaluate(p,store,encode(store),loader,final)
    assert r['status']=='partial_or_failed' and 'metrics.json' not in store.files
    assert all(c['Q'] is None for row in r['rows'] for c in row['checkpoints'])
    assert r['errors'][0]['message']=='restore failed'


def test_consume_rejects_unknown_artifact_and_197hidden():
    p=prepared();store=RAMStore();store.put('execution_binding.json',{})
    r=m.evaluate(p,store,encode(store),loader,lambda:finish(True))
    r['artifact_sha256']['unexpected.bin']=store.write('unexpected.bin',b'unknown')
    with pytest.raises(ValueError):m.consume_report(r,p,store.blob)
    del r['artifact_sha256']['unexpected.bin'];r['shapes']['hidden']=[1,197,192]
    with pytest.raises(ValueError):m.consume_report(r,p,store.blob)

def test_final_physical_meter_matches_actual_producer_RAM():
    from test_moving_prediction_matched_sampling_physical_20261027_v1 import execute
    report,store,_=execute()
    p=store.events[0]['prepare']
    assert m.validate_physical_meter(report,p,store.events)==report['counts']


@pytest.mark.parametrize('fault',['obsolete','missing','status','receipt','ordering','counter','invented'])
def test_final_meter_abi_mutations_rejected(fault):
    report,p,_,events,_,_=fixture()
    indices=[i for i,e in enumerate(events) if e['kind']=='operation_dispatch_intent']
    i=indices[0]
    if fault=='obsolete':events[i]=dict(kind='operation_attempted',resource='public_step')
    elif fault=='missing':events.pop(i+1)
    elif fault=='status':events[i+1]['receipt']['status']='pending'
    elif fault=='receipt':events[i+1]['receipt']['resource']='public_step'
    elif fault=='ordering':events[i],events[i+1]=events[i+1],events[i]
    elif fault=='counter':report['counts']['public_step']['completed']-=1
    else:report['actual_operation_receipts'].pop()
    with pytest.raises(ValueError):m.validate_physical_meter(report,p,events)


def test_physical_final_source_pin_is_not_old_mock_pin():
    assert m.PHYSICAL_SHA=='50552d50ce3d97c9a25f938ef2b47e99bc66c88fa1644659611b5bb32843884a'
