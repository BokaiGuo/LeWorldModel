"""RAM hook/tensor and saved-schema tests only; no NN/physics production."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_common_resource_rgb_20261029_v1 as m
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as enc
import test_agent_p2_common_resource_physical_20261029_v1 as phys


def fixture(stops=(15,15,15,15,15,15),missing=()):
    p,_,modelprep,_=phys.fixture(missing);actual=m.producer.rows_for(p)
    for root,slot in enumerate(modelprep['slots']):
        slot.update(blind_path=f'/RAM/blind{root}.npz')
    p['model_slots']=deepcopy(modelprep['slots'])
    for i,plan in enumerate(p['plans']):
        if plan is None:continue
        row=actual[i];stop=stops[i]
        row.update(initial_exact=True,status='horizon_completed' if stop==15 else 'native_stopped',own_returned=stop,
            checkpoints=[],trajectory=[],responses=[],missing_checkpoints=[])
        for step in (5,10,15):
            if step>stop:row['missing_checkpoints'].append(step);continue
            state=deepcopy(plan['native_goal']);state[0]+=step
            cp=dict(step=step,png=f'p{i}_s{step}.png',file_sha256=phys.h(('file',i,step)),
                pixel_sha256=phys.h(('pixels',i,step)),state=deepcopy(state),snapshot=dict(state=deepcopy(state),RAM=True))
            row['checkpoints'].append(cp);row['trajectory'].append(dict(deepcopy(cp),phase='plan',overlay_index=0,condition='off'))
            metric=m.producer.native.metrics(plan['native_goal'],state)
            row['responses'].append(dict(step=step,state=deepcopy(state),native_metrics=metric['native'],
                reward=metric['native7_reward'],native_success=metric['native_success'],terminated=metric['native_success'],truncated=False))
    strict=dict(schema=m.producer.NAME+'.consume',status=m.producer.RECORDED,fixed_denominator=6,known_roots=2,fresh_roots=0,
        errors=[],P1_admission=False,P2_admission=False,physical_release=False,rows=actual,plans=deepcopy(p['plans']),
        planner_authority=deepcopy(p['planner_authority']))
    return p,strict,modelprep

def prepared(stops=(15,15,15,15,15,14),missing=()):
    rows=m.fixed_rows(*fixture(stops,missing),'/RAM/physical');n=sum(x['source'] is not None for x in rows)
    return dict(rows=rows,contract=m.contract(),expected_calls={k:n if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})


def execute(monkeypatch,stops=(15,15,15,15,15,14),tokens=257,forbidden=None,fail_at=None,store=None,missing=()):
    p=prepared(stops,missing);rows=deepcopy(p['rows']);s=store if store is not None else enc.old.Memory()
    model=enc.Model(tokens);torch=enc.fake_torch();errors=[]
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    trace=m.bridge.Trace(model,torch,s,lambda:None,p['expected_calls'])
    guard=m.Ledger(p['expected_calls']['encoder'],lambda e:s.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    original=model.predict
    def encode(pix,row,scope,ordinal):
        if fail_at==ordinal:raise ValueError('RAM injected encode failure')
        guard.image=ordinal
        with m.visual.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(enc.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(enc.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    try:
        with trace,m.bridge.native.guard(model,torch,guard,errors):m.process(p,rows,s,encode)
    except ValueError:
        assert model.predict==original
        if fail_at is not None:return p,rows,s
        raise
    assert model.predict==original and not errors and not guard.blocked
    r=dict(rows=rows,contract=m.contract(),errors=errors,blocked=guard.blocked,modules=trace.entries,guards=guard.entries,
        counts=m.bridge.counts(trace.entries))
    return p,r,s


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)


def test_own_common_schema18_slots17_calls_and18_pairs(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    rows=m.validate(p,r,s.events,s.load);pairs=m.paired(rows);groups=m.grouped(rows)
    assert len(rows)==18 and len(pairs)==18 and len(groups)==6 and len(r['modules'])==34
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=17,completed=17)
    assert all(r['counts'][k]==dict(attempted=0,completed=0) for k in m.bridge.KINDS if k not in ('encoder','projector'))
    assert [x['checkpoint'] for x in rows]==[5,10,15]*6
    assert rows[17]['latent_missing'] and rows[17]['real_Q'] is rows[17]['native_metrics'] is None
    assert rows[17]['actual_returned']==14 and rows[17]['model_Q'] is not None
    assert p['rows'][17]['causal_raw10'] is p['rows'][17]['causal_raw10_bits'] is None
    assert p['rows'][17]['planned_raw10'] is not None
    assert sum(x['real_pair_missing'] for x in pairs)==2
    assert groups[-1]['missing']==dict(joint=True,composition=False,grasp=False)
    assert groups[-1]['model_Q']['joint'] is not None
    assert all('edge' not in x for x in p['rows'])
    assert len(set(x['model_Q'] for x in rows[:3]))==3


def test_exact_three_arm_same_time_pair_arithmetic(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    rows=m.validate(p,r,s.events,s.load);pairs=m.paired(rows)
    for pair in pairs:
        left,right=rows[pair['left_index']],rows[pair['right_index']]
        assert left['root_slot']==right['root_slot'] and left['checkpoint']==right['checkpoint']
        assert pair['model_Q']['delta']==left['model_Q']-right['model_Q']
        if not pair['real_pair_missing']:
            assert pair['real_Q']['delta']==left['real_Q']-right['real_Q']
            assert pair['physical_metrics']['delta']['joint_xy_error']==left['native_metrics']['joint_xy_error']-right['native_metrics']['joint_xy_error']
    for row,spec in zip(rows,p['rows'],strict=True):
        if not row['latent_missing']:
            assert row['real_Q']==m.bridge.mse(s.load(r['rows'][row['index']]['latent']),spec['goal192'])
            assert row['model_Q']==m.bridge.mse(spec['output_context']['latent_history'][-1],spec['goal192'])

@pytest.mark.parametrize('stops',[(4,9,14,14,0,15),(0,0,0,0,0,0)])
def test_more_missing_retains_full_denominators(monkeypatch,stops):
    p,r,s=execute(monkeypatch,stops);rows=m.validate(p,r,s.events,s.load)
    assert len(rows)==18 and len(m.paired(rows))==18
    assert sum(not x['latent_missing'] for x in rows)==p['expected_calls']['encoder']
    assert all(x['source'] is None and x['real_Q'] is None for x in rows if x['latent_missing'])


def test_all_eighteen_images_are_allowed_but_not_nineteenth(monkeypatch):
    p,r,s=execute(monkeypatch,(15,)*6)
    assert r['counts']['encoder']==dict(attempted=18,completed=18)
    assert len(m.validate(p,r,s.events,s.load))==18
    guard=m.Ledger(18,lambda e:None)
    guard.image=18
    with pytest.raises(ValueError):guard.dispatch('encoder',None,lambda:pytest.fail('no callback'))

@pytest.mark.parametrize('bad',['predict','AD','action_encoder'])
def test_zero_prediction_AD_action_encoder(monkeypatch,bad):
    with pytest.raises(ValueError):execute(monkeypatch,forbidden=bad)


def test_original257_hidden_no_shape_relaxation(monkeypatch):
    with pytest.raises(ValueError):execute(monkeypatch,tokens=197)


@pytest.mark.parametrize('fault',['schema','drop','order','task','model','raw','goal','warm','context','history','cpstep','state','response','missing','early','terminalQ','root'])
def test_actual_plan_context_native_binding_mutations(fault):
    p,s,mp=fixture();a=s['rows'][0];plan=p['plans'][0]
    if fault=='schema':s['schema']='agent_p2_wallcapped_physical_20261029_v1.consume'
    elif fault=='drop':s['rows'].pop()
    elif fault=='order':s['rows'].reverse()
    elif fault=='task':a['task_id']='bad'
    elif fault=='model':mp['expected_model_hash']='bad'
    elif fault=='raw':plan['commands'][0][0]=.75
    elif fault=='goal':plan['planner_plan']['goal'][0]+=.1
    elif fault=='warm':plan['planner_plan']['warm_raw10'][0][0]=.1
    elif fault=='context':plan['planner_plan']['contexts'][0]['latent_history'][-1][0]+=.1
    elif fault=='history':plan['planner_plan']['contexts'][0]['action_history'][0][0]+=.1
    elif fault=='cpstep':a['checkpoints'][0]['step']=10
    elif fault=='state':a['checkpoints'][0]['state'][0]+=1
    elif fault=='response':a['responses'][0]['reward']+=1
    elif fault=='missing':a['missing_checkpoints']=[5]
    elif fault=='early':a['own_returned']=4
    elif fault=='root':a['initial_exact']=False
    else:plan['planner_plan']['model_Q']+=1
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,s,mp,'/RAM/o')


@pytest.mark.parametrize('fault',['event','drop_attempt','order','shape','preprocessed','pixel','latent','spec','guard','missing','count','blocked'])
def test_full_saved_tensor_events_mutations(replay,monkeypatch,fault):
    p,original,store=replay;r=deepcopy(original);events=deepcopy(store.events);s=enc.old.Memory();s.files=dict(store.files)
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels);row=r['rows'][0]
    if fault=='event':events.insert(0,dict(kind='UNKNOWN'))
    elif fault=='drop_attempt':events.remove(next(e for e in events if e['kind']=='module_attempt'))
    elif fault=='order':events[2],events[3]=events[3],events[2]
    elif fault=='shape':row['modules'][0]['output']=s.array('bad197',np.zeros((1,197,192),np.float32))
    elif fault=='preprocessed':row['preprocessed']=s.array('badpre',np.zeros((1,1,3,224,224),np.float32))
    elif fault=='pixel':row['pixels']=s.array('badpix',np.zeros((224,224,3),np.uint8))
    elif fault=='latent':row['latent']=s.array('badlatent',np.zeros(192,np.float32))
    elif fault=='spec':row['model_Q']+=1
    elif fault=='guard':r['guards'].pop()
    elif fault=='missing':row['source']=None
    elif fault=='count':r['counts']['encoder']['completed']-=1
    else:r['blocked']=['predict']
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,events,s.load)


def test_partial_no_retry_missing_kept_and_no_later_dispatch(monkeypatch):
    p,rows,s=execute(monkeypatch,fail_at=2)
    assert len(rows)==18 and rows[0]['status']==rows[1]['status']=='encoded'
    assert rows[2]['status']=='attempted' and rows[2]['latent'] is None
    assert all(x['status']=='not_started' for x in rows[3:17]) and rows[17]==p['rows'][17]
    assert len([e for e in s.events if e['kind']=='image_attempt'])==3


def test_auth_before_any_reads(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('no IO'))
    with pytest.raises(ValueError):m.run('/absent','a'*64,'/absent','bad')


def test_real_store_arrays_events_and_consumer_disk(tmp_path,monkeypatch):
    budget=m.bridge.Budget(free=lambda:10**12)
    store=m.Store(tmp_path/(m.NAME+'_RAM'),budget)
    p,r,_=execute(monkeypatch,store=store)
    p.update(arguments=dict(RAM=True),bindings={},expected_model_hash='a'*64,physical_report_sha256='b'*64,
        physical_receipt_sha256='c'*64,inherited_physical_counts={},inherited_model_accounts=[],inherited_model_timings=dict(total_seconds=1.),inherited_model_resources=[],runtime_report_sha256='d'*64,runtime_receipt_sha256='e'*64,runtime_prepare_sha256='f'*64)
    prepared=tmp_path/(m.NAME+'_prepare_RAM.json');m.write(prepared,p)
    store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=m.sha(prepared)));store.put('initial_roster.json',p['rows'])
    seal={k:r[k] for k in ('rows','modules','guards','blocked','errors')};h=store.put('native_seal.json',seal,final=True)
    r.update(schema=m.NAME+'.report',status='common_resource_RGB_encoded_NOT_science',prepared=str(prepared),prepare_sha256=m.sha(prepared),
        model_unchanged=True,source_verified=True,model_hash_before='a'*64,model_hash_after='a'*64,
        P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False,elapsed_seconds=1.,
        peak_allocated_bytes=1024,peak_reserved_bytes=2048,native_seal_sha256=h,event_count=store.index,
        artifact_sha256=deepcopy(store.hashes))
    store.put('report.json',r,final=True)
    monkeypatch.setattr(m,'verify',lambda *a:p)  # only own prep boundary, no real authority claim
    receipt=m.consume(store.path,m.sha(store.path/'report.json'))
    assert len(receipt['pairs'])==18 and receipt['missing_slots']==[17] and receipt['real_pairs_available']==16
    assert receipt['counts']['encoder']['completed']==17
    (store.path/'extra.json').write_text('{}')
    with pytest.raises(ValueError):m.consume(store.path,m.sha(store.path/'report.json'))


def test_own_metadata_codec_and_no_overwrite(tmp_path):
    assert m.producer.digest([1,2])!=m.producer.planner_digest([1,2])
    path=tmp_path/(m.NAME+'_RAM.json');m.write(path,dict(RAM=True))
    with pytest.raises(ValueError):m.write(path,{})
    assert m.META==128*1024**2 and m.contract()['max_calls']['predict_API']==0


@pytest.mark.parametrize('missing',[(2,),(0,2,4),(0,1,2,3,4,5)])
def test_null_runtime_exports_preserve_all18_and_all_three_arm_joins(monkeypatch,missing):
    p,r,s=execute(monkeypatch,(15,)*6,missing=missing)
    rows=m.validate(p,r,s.events,s.load)
    assert len(rows)==18 and len(m.grouped(rows))==6 and len(m.paired(rows))==18
    assert r['counts']['encoder']==dict(attempted=18-3*len(missing),completed=18-3*len(missing))
    for plan in missing:
        for row in rows[3*plan:3*plan+3]:
            assert row['source'] is row['model_Q'] is row['real_Q'] is row['response'] is None
            assert row['latent_missing'] and row['missing_reason']=='null_runtime_export'
            spec=p['rows'][row['index']]
            assert spec['full15'] is spec['prefix_actions'] is spec['causal_raw10'] is None
    assert all(x['root_slot'] in (0,1) for x in rows)


@pytest.mark.parametrize('field',['task','goal','Q','condition','root_context','duplicate','wrong_time'])
def test_three_arm_same_time_domain_join_refuses_mismatch(replay,monkeypatch,field):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    rows=m.validate(p,r,s.events,s.load);rows=deepcopy(rows)
    if field=='task':rows[6]['task_sha256']='0'*64
    elif field=='goal':rows[6]['goal192'][0]+=.01
    elif field=='Q':rows[6]['Q_sha256']='0'*64
    elif field=='condition':rows[6]['condition']='on'
    elif field=='root_context':rows[6]['root_context']['latent_history'][0][0]+=.01
    elif field=='duplicate':rows[6]['arm']='joint'
    else:rows[6]['checkpoint']=10
    with pytest.raises(ValueError):m.grouped(rows)


def test_no_old_guard_12_image_ceiling_and_exact18_not_20():
    from types import SimpleNamespace as NS
    events=[];guard=m.Ledger(18,events.append)
    # This is ONLY a metadata device/shape stand-in; callback performs no ops.
    x=NS(device=NS(type='cuda'),shape=(1,3,224,224));z=NS(device=NS(type='cuda'),shape=(1,192))
    for i in range(18):
        guard.image=i;guard.dispatch('encoder',x,lambda:None);guard.dispatch('projector',z,lambda:None)
    assert len(guard.entries)==36 and all(e['completed'] for e in guard.entries)
    guard.image=18
    with pytest.raises(ValueError):guard.dispatch('encoder',x,lambda:pytest.fail('19th must not run'))


def test_build_auth_source_before_consume(monkeypatch):
    from types import SimpleNamespace as NS
    monkeypatch.setattr(m,'sha',lambda path:'a'*64)
    fake=NS(**vars(m.producer));fake.consume=lambda *a:pytest.fail('source seal before strict')
    monkeypatch.setattr(m,'producer',fake)  # only new-module injection, not frozen producer globals
    with pytest.raises(ValueError,match='physical source'):
        m.build('/RAM/out','a'*64,'/RAM/receipt','a'*64,'b'*64)


def test_cold_import_no_neural_or_physics():
    import sys
    import subprocess
    code='''
import sys
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','gymnasium','stable_worldmodel','pymunk'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from experiments.spectral_reachability import agent_p2_common_resource_rgb_20261029_v1 as m
assert m.contract()['fixed_denominator']==18 and m.contract()['max_calls']['encoder']==18
'''
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,text=True,capture_output=True,timeout=20)
    assert r.returncode==0,r.stderr


def test_build_complete_own_physical_runtime_receipt_join_in_RAM(monkeypatch):
    from types import SimpleNamespace as NS
    p,strict,mp=fixture((15,15,15,15,15,14));p['bindings']={};strict['bindings']={};strict['counts']={'RAM':True}
    mp['policy']=str(m.bridge.rgb.POLICY)
    physical_output='/RAM/physical';receipt='/RAM/receipt.json';physical_prepare='/RAM/physical_prepare.json'
    runtime_report=str(Path(p['arguments']['planner_output'])/'report.json');runtime_prepare='/RAM/runtime_prepare.json'
    physical_report=dict(execution_binding=dict(prepared=physical_prepare,prepare_sha256=phys.h('pp')))
    runtime=dict(prepared=runtime_prepare,prepare_sha256=phys.h('rp'),timings=dict(total_seconds=12.))
    p['bindings'].update({runtime_report:p['arguments']['planner_report_sha'],runtime_prepare:runtime['prepare_sha256']})
    mapping={str(Path(physical_output)/'report.json'):physical_report,receipt:strict,runtime_report:runtime,runtime_prepare:mp}
    pins={str(Path(m.producer.__file__)):phys.h('source'),str(Path(physical_output)/'report.json'):phys.h('report'),
          receipt:phys.h('receipt'),**p['bindings']}
    calls=[];real=m.producer;fake=NS(**vars(real))
    def consume(path,expected):
        calls.append(('consume',str(path),expected));return deepcopy(strict)
    def verify(path,expected):
        calls.append(('verify',str(path),expected));assert path==physical_prepare;return deepcopy(p),{}
    fake.consume=consume;fake.verify=verify
    monkeypatch.setattr(m,'producer',fake)
    monkeypatch.setattr(m,'sha',lambda path:pins.get(str(path),phys.h(('source',str(path)))))
    monkeypatch.setattr(m,'read',lambda path:deepcopy(mapping[str(path)]))
    shim=NS(**vars(m.bridge));shim.source_files=lambda:[];shim.load_pixels=enc.pixels;shim.verify_bindings=lambda _:None
    shim.rgb=NS(**vars(m.bridge.rgb));shim.rgb.read=lambda _:dict(encoder=dict(image_size=224,patch_size=14,pretrained=False),projector=dict(input_dim=192,output_dim=192))
    monkeypatch.setattr(m,'bridge',shim)
    result=m.build(physical_output,phys.h('report'),receipt,phys.h('receipt'),phys.h('source'))
    assert [x[0] for x in calls]==['consume','verify']
    assert len(result['rows'])==18 and result['expected_calls']['encoder']==17
    assert result['runtime_receipt_sha256']==p['arguments']['planner_receipt_sha']
    assert result['runtime_report_sha256']==p['arguments']['planner_report_sha']
    assert result['runtime_prepare_sha256']==runtime['prepare_sha256']
    assert result['inherited_model_resources']==p['planner_authority']['resources']
    mapping[receipt]=deepcopy(strict);mapping[receipt]['fixed_denominator']=5
    with pytest.raises(ValueError,match='entire own physical.consume'):m.build(physical_output,phys.h('report'),receipt,phys.h('receipt'),phys.h('source'))

