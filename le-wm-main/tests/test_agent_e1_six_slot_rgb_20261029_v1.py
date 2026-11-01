from copy import deepcopy
import json
import sys
import types
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e1_six_slot_rgb_20261029_v1 as m


def fixture():
    plans=[];rows=[];tasks=[];replays=[]
    for i,(root,arm) in enumerate(m.native.producer.core.SLOTS):
        target=[float(i)]*192;goal=[float(root)]*192;z=[0.0]*192
        plan=dict(root_slot=root,arm=arm,target=target,commands=[[0.0,0.0]]*10,selection_digest=str(i),task_id=str(root))
        points=[dict(step=step,phase='plan',state=[0]*7,observation_state=[0]*7,snapshot={'state':[0]*7}) for step in m.CP]
        plans.append(plan);rows.append(dict(branch_index=i,root_slot=root,arm=arm,selected_plan=deepcopy(plan),checkpoints=points))
        tasks.append(dict(goal=goal,root_context=dict(latent_history=[z,z,z])))
        state=lambda x:dict(latent_history=[z,z,[x]*192],action_history=[[0.]*10,[0.]*10])
        replays.append(dict(raw=plan['commands'],whole=[state(1.),state(2.)],segmented=[state(1.),state(2.5)]))
    return plans,rows,tasks,replays


def test_all_slots_separate_whole_and_segmented():
    roster=m.assemble(*fixture());assert len(roster)==18
    out=m.panel(roster,[np.zeros(192,np.float32)]*18)
    assert out['fixed_cells']==18 and out['winner'] is None and out['FLOPs'] is None
    assert out['rows'][2]['prediction_error']==4 and out['rows'][2]['whole_segmented_prediction_error']==0.25


def test_missing_preserved():
    plans,rows,tasks,replays=fixture();rows[1]['checkpoints'].pop()
    r=m.assemble(plans,rows,tasks,replays);actual=[np.zeros(192,np.float32) if x['checkpoint'] else None for x in r]
    out=m.panel(r,actual);assert len(out['rows'])==18 and out['rows'][5]['actual_tracking_Q'] is None


@pytest.mark.parametrize('kind',['plan','raw','goal','duplicate','state'])
def test_invalid_join_rejected(kind):
    p,r,t,x=fixture()
    if kind=='plan':r[0]['selected_plan']['selection_digest']='changed'
    if kind=='raw':x[0]['raw']=[[1.,0.]]*10
    if kind=='goal':t[1]['goal']=[5.]*192
    if kind=='duplicate':r[0]['checkpoints'].append(deepcopy(r[0]['checkpoints'][0]))
    if kind=='state':r[0]['checkpoints'][0]['state']=[1]*7
    with pytest.raises(ValueError):m.assemble(p,r,t,x)


def test_complete_tempdisk_fixture_and_tamper(tmp_path,monkeypatch):
    rows=m.assemble(*fixture());pixels=np.zeros((224,224,3),np.uint8)
    for row in rows:row.update(status='pending',latent=None,pixels=None,preprocessed=None,modules=[],pixel_source={'fixture':True})
    p=dict(schema=m.NAME+'.prepare',rows=rows,contract=m.contract(),expected_model_hash='fixture',bindings={},
           expected_calls={k:18 if k in ('encoder','projector') else 0 for k in m.bridge.KINDS},
           imported_native_receipt={'counts':{}},imported_controller_receipt={'fixture':True},controller_report_sha256='fixture')
    monkeypatch.setattr(m.bridge,'load_pixels',lambda _:pixels.copy())
    store=m.Store(tmp_path/(m.NAME+'_fixture'),m.shared.Budget())
    store.put('binding.json',dict(prepared='/fixture/prepare',prepare_sha256='fixture'))
    store.put('initial_roster.json',p['rows']);out=deepcopy(rows);entries=[]
    def encode(pix,row,scope):
        pre=m.bridge.pixels.preprocess_pixels(pix);row['preprocessed']=store.array(f'r{row["index"]}_pre',pre)
        z=np.asarray(row['predicted192'],np.float32);hidden=np.repeat(z[None,None],257,axis=1);mods=[]
        for kind,inp,value in [('encoder',pre.reshape(1,3,224,224),hidden),('projector',hidden[:,0],z[None])]:
            e=dict(index=len(entries),scope=deepcopy(scope),resource=kind,completed=False)
            store.event(dict(kind='module_attempt',entry=deepcopy(e)))
            e['inputs']=[store.array(f'm{e["index"]}_in',inp)];e['output']=store.array(f'm{e["index"]}_out',value);e['completed']=True
            store.event(dict(kind='module_return',entry=deepcopy(e)));entries.append(e);mods.append(e)
        row['modules']=deepcopy(mods);return z
    m.process(p,out,store,encode)
    seal=dict(rows=out,modules=entries,counts=m.bridge.counts(entries),errors=[])
    h=store.put('native_seal.json',seal)
    r=dict(schema=m.NAME+'.report',contract=m.contract(),completed=True,**seal,model_before='fixture',model_after='fixture',
           gpu_peaks=dict(allocated=1,reserved=1,guard_checks=1),host=dict(device='cuda:0'),prepared='/fixture/prepare',prepare_sha256='fixture',
           resources=m.accounting(p,out,entries),native_seal_sha256=h,artifact_sha256=deepcopy(store.hashes),
           artifact_order=list(store.order),event_count=store.index,wall_before_report=1)
    store.put('report.json',r,final=True);rh=m.sha(store.path/'report.json')
    store.put('completion.json',dict(schema=m.NAME+'.completion',report_sha256=rh,complete=True,wall_inclusive=2,boundary=m.BOUNDARY),final=True)
    monkeypatch.setattr(m,'verify',lambda *args:p)
    result=m.consume(store.path,rh)
    assert result['fixed_slots']==6 and result['fixed_cells']==18 and result['resources']['encoder_samples']==18
    (store.path/'event_0000.json').write_bytes(b'{}')
    with pytest.raises(ValueError):m.consume(store.path,rh)


def test_run_authorization_before_verify(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('verify before authorization'))
    with pytest.raises(ValueError,match='authorization'):m.run('missing','missing',tmp_path/'x')


def test_prepare_verify_with_real_fixture_files(tmp_path,monkeypatch):
    plans,rows,tasks,replays=fixture();folder=tmp_path/'physical';folder.mkdir()
    source=tmp_path/'task.json';source.write_bytes(b'fixture task')
    task_contract={'pixel_sha256':['a','b','px']};contract_sha=m.native.old.source.digest(task_contract)
    for i,plan in enumerate(plans):
        plan.update(task_export=str(source),task_sha256=m.sha(source),task_contract_sha256=contract_sha)
        tasks[i]['fixture_id']=plan['task_id'];rows[i]['selected_plan']=deepcopy(plan)
        for cp in rows[i]['checkpoints']:
            f=folder/f'b{i}_{cp["step"]}.png';f.write_bytes(b'fixture PNG')
            cp.update(png=f.name,file_sha256=m.sha(f),pixel_sha256='px')
    native_prepare={'plans':plans,'bindings':{str(source):m.sha(source)}}
    receipt={'status':'six_slot_native_recorded','rows':rows,'counts':{}}
    receipt_path=tmp_path/'receipt.json';receipt_path.write_bytes(m.raw(receipt))
    report_path=folder/'report.json';report_path.write_bytes(m.raw({'prepared':str(source),'prepare_sha256':m.sha(source)}))
    monkeypatch.setattr(m,'REPORT_SHA',m.sha(report_path));monkeypatch.setattr(m,'RECEIPT_SHA',m.sha(receipt_path))
    monkeypatch.setattr(m.native,'consume',lambda *a:deepcopy(receipt))
    monkeypatch.setattr(m.native,'verify',lambda *a:deepcopy(native_prepare))
    controller_path=tmp_path/'controller.json'
    controller_path.write_bytes(m.raw({'report_sha256':'fixture-controller','status':'all_six_saved_controller_and_replays_verified'}))
    monkeypatch.setattr(m.native,'RECEIPT',controller_path);monkeypatch.setattr(m.native,'RECEIPT_SHA',m.sha(controller_path))
    monkeypatch.setattr(m.native,'REPORT_SHA','fixture-controller')
    producer=m.native.producer;loader=producer.core.old.sampling.old
    monkeypatch.setattr(producer,'read_bound',lambda *a:b'fixture')
    monkeypatch.setattr(producer,'parse_canonical',lambda _:dict(model_hash_before=m.bridge.native.MODEL_TENSOR_SHA,model_hash_after=m.bridge.native.MODEL_TENSOR_SHA))
    monkeypatch.setattr(producer,'read_store',lambda *a:({f'slot_{i}.replay.z':replays[i] for i in range(6)},[]))
    calls=[0]
    def load(*args):
        task=tasks[calls[0]%6];calls[0]+=1;return {'gn_task':deepcopy(task)}
    monkeypatch.setattr(loader,'load_export',load)
    # All synthetic plans for the same root have the same task id; alternate
    # source slots are represented by the current fixture invocation counter.
    monkeypatch.setattr(loader,'read',lambda *a:{'source_slot':plans[(calls[0]-1)%6]['root_slot'],
        'task_id':plans[(calls[0]-1)%6]['task_id'],'task_contract':task_contract})
    monkeypatch.setattr(producer.core.old.core,'task_id',lambda task:task['fixture_id'])
    monkeypatch.setattr(m.shared,'source_files',lambda:[])
    monkeypatch.setattr(m.bridge.rgb,'MODELS',{})
    monkeypatch.setattr(m.bridge,'load_pixels',lambda _:np.zeros((224,224,3),np.uint8))
    output=tmp_path/(m.NAME+'_prepare.json')
    arguments=dict(native_output=folder,native_report_sha=m.REPORT_SHA,native_receipt=receipt_path,native_receipt_sha=m.RECEIPT_SHA)
    p=m.build(**arguments);ref=m.write_metadata(output,p)
    assert len(m.verify(output,ref['sha256'])['rows'])==18
    receipt_path.write_bytes(b'changed')
    with pytest.raises(ValueError):m.verify(output,ref['sha256'])


def test_failure_retained_after_fake_cuda_preflight(tmp_path,monkeypatch):
    fake=types.ModuleType('torch');fake.__path__=[]
    fake.cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda:None)
    fake.backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
                                        cudnn=types.SimpleNamespace(allow_tf32=False))
    fake.are_deterministic_algorithms_enabled=lambda:False
    fake.is_deterministic_algorithms_warn_only_enabled=lambda:False
    fake.use_deterministic_algorithms=lambda *a,**kw:None
    fake.device=lambda x:x
    nn=types.ModuleType('torch.nn');nn.__path__=[]
    att=types.ModuleType('torch.nn.attention');att.sdpa_kernel=lambda *a:None;att.SDPBackend=types.SimpleNamespace(MATH=0)
    utils=types.ModuleType('torch.utils');utils.__path__=[]
    dispatch=types.ModuleType('torch.utils._python_dispatch');dispatch.TorchDispatchMode=type('TorchDispatchMode',(),{})
    for k,v in [('torch',fake),('torch.nn',nn),('torch.nn.attention',att),('torch.utils',utils),('torch.utils._python_dispatch',dispatch)]:monkeypatch.setitem(sys.modules,k,v)
    def fail(*a,**kw):raise RuntimeError('fixture load failure')
    preflight=types.ModuleType('experiments.gpu_preflight');preflight.require_cuda_allocation=lambda _:dict(device='cuda:0')
    compat=types.ModuleType('experiments.checkpoint_compat');compat.load_compatible_pretrained=fail
    p0=types.ModuleType('experiments.spectral_reachability.preflight_pusht_d2p_p0')
    for k,v in [('experiments.gpu_preflight',preflight),('experiments.checkpoint_compat',compat),
                ('experiments.spectral_reachability.preflight_pusht_d2p_p0',p0)]:monkeypatch.setitem(sys.modules,k,v)
    prepared=tmp_path/'prepared.json';prepared.write_bytes(b'fixture')
    p={'rows':[],'bindings':{},'policy':'fixture','expected_model_hash':'fixture',
       'expected_calls':{k:0 for k in m.bridge.KINDS}}
    monkeypatch.setattr(m,'verify',lambda *_:p)
    output=tmp_path/(m.NAME+'_failed')
    with pytest.raises(ValueError,match='partial failed artifacts'):m.run(prepared,m.sha(prepared),output,main_authorized=True)
    report=json.loads((output/'report.json').read_bytes());completion=json.loads((output/'completion.json').read_bytes())
    assert report['completed'] is False and report['errors'] and completion['complete'] is False
    assert report['resources']['encoder_samples']==0 and (output/'native_seal.json').is_file()
    with pytest.raises(ValueError):m.consume(output,m.sha(output/'report.json'))


def test_actual_saved_replay_receipt_bitwise_abi_only():
    p=m.native.RECEIPT
    assert m.sha(p)==m.native.RECEIPT_SHA
    value=m.native.old.read(p,m.META)
    assert value['status']=='all_six_saved_controller_and_replays_verified' and len(value['scores'])==6
    for row in value['scores']:
        assert len(row['checkpoints'])==2
        for cp in row['checkpoints']:
            for k in ('action_history','latent_history'):
                assert cp[k]['bitwise_equal'] is True and cp[k]['max_abs']==0 and cp[k]['max_ulp']==0
