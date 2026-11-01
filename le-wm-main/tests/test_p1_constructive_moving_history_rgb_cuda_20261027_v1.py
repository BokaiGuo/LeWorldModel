"""RAM-only moving schema/encoder mocks; never simulator/model/actual prepare."""
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_moving_history_rgb_cuda_20261027_v1 as m
from test_p1_constructive_moving_history_physical_20261027_v1 import fixture as physical_fixture,execute
from test_p1_constructive_moving_history_encode_cuda_20261027_v1 import protocol
from test_p1_constructive_selected_rgb_cuda_20261027_v1 import RAMStore


def fixture():
    p,data=physical_fixture();tasks=[];selected={}
    p.update(schema=m.physical.NAME+'.prepare',manifest='/RAM/manifest.json',manifest_sha256='a'*64,
        source_slots=[dict(root_slot=s,root_sha256=ref['root_sha256']) for s,ref in enumerate(data['references'])])
    for s,ref in enumerate(data['references']):
        packet=dict(rgb_history=ref['history'],raw_action_history=ref['raw_history'],goal_rgb=ref['goal_rgb'])
        pr=protocol();slot=pr['slots'][s]
        slot.update(blind_sha256=ref['blind_sha256'],pixel_sha256=[m.moving.digest(a.tobytes()) for a in [*packet['rgb_history'],packet['goal_rgb']]])
        export=m.moving.task_export(pr,'d'*64,slot,packet,[np.full(192,i,np.float32) for i in range(4)])
        interface=m.moving.task_interfaces(export);t=interface['gn_task']
        tasks.append(dict(source_slot=s,task_export=m.EXPORTS[s],task_sha256=m.EXPORT_SHAS[s],
            blind_sha256=ref['blind_sha256'],task_contract=export['task_contract'],interface=interface))
        for a in m.ARMS:
            raw=np.zeros((10,2),np.float32).tolist();tokens=m.moving.core.normalized_tokens(t,raw)
            before=deepcopy(t['root_context'])
            for k,z in enumerate((2.5,2.75)):
                before=dict(before,latent_history=before['latent_history'][1:]+[[z]*192],
                    action_history=[before['action_history'][-1],tokens[k].tolist()])
            body=dict(schema=m.planner.NAME+'.model_plan',source_slot=s,order=s,arm=a,blind_sha256=ref['blind_sha256'],
                task_export=m.EXPORTS[s],task_sha256=m.EXPORT_SHAS[s],task_id=interface['task_id'],
                task_contract_sha256=t['task_contract_sha256'],with_target=False,physical_execution=False,
                selection_uses_physics=False,P1_admission=False,logits=np.zeros((2,10)).tolist(),commands=raw,
                prediction=before,model_terminal_cost=m.moving.core.model_cost(t,before['latent_history'][-1]))
            selected[s,a]=dict(body,selection_sha256=m.moving.core.digest(body),whole_run_report_sha256=m.PLANNER_REPORT_SHA)
    for b,plan in enumerate(p['plans']):
        s=b//2;a=m.ARMS[b%2]
        plan.update(arm=a,commands=deepcopy(selected[s,a]['commands']),compiler_report='/RAM/handoff.json',
            compiler_report_SHA='b'*64,compiler_schema='moving_history_planner_handoff.v1',compiler_status='commands_verified_before_physics',
            commands_pointer=f'/data/plans/{b}/commands',source_slot_pointer=f'/data/plans/{b}/source_slot',
            blind_sha_pointer=f'/data/plans/{b}/blind_sha256',arm_pointer=f'/data/plans/{b}/arm')
        plan.pop('selection_digest');plan['selection_digest']=m.physical.source.digest(plan)
    report,store,_=execute(p=p,data=data)
    assert report['status']=='moving_history_physical_recorded'
    return report,store.events,p,tasks,selected


def test_all_eight_images_from_moving_schema_without_source_slot_rows():
    data=fixture();before=deepcopy(data);images=m.assemble(*data)
    assert data==before and len(images)==8 and all('source_slot' not in r for r in data[0]['rows'])
    assert [(i['source_slot'],i['arm'],i['role']) for i in images]==[(s,a,r) for s in (0,1) for a in m.ARMS for r in ('prefix5','terminal10')]
    assert all(len(i['predicted'])==192 for i in images)


@pytest.mark.parametrize('bad',('schema','partial','missing5','missing10','arm','digest','boolslot','counts','command','goal',
    'guard','snapshot','history','raw','no_close','truncated','tail','firsterror','duplicate5'))
def test_saved_contract_failclosed_no_missing_image_substitution(bad):
    r,e,p,t,s=fixture();row=r['rows'][0]
    if bad=='schema':r['schema']=m.rgb.old.PHYSICAL+'.report'
    elif bad=='partial':row['full_horizon']=False
    elif bad.startswith('missing'):row['checkpoints']=[c for c in row['checkpoints'] if c['step']!=int(bad[7:])]
    elif bad=='arm':row['arm']='old_GN'
    elif bad=='digest':row['selection_digest']='0'*64
    elif bad=='boolslot':row['root_slot']=False
    elif bad=='counts':r['counts']['physics_step']['completed']-=1
    elif bad=='command':next(x for x in e if x['kind']=='returned_transition' and x['phase']=='plan')['command']=[.5,0.]
    elif bad=='goal':t[0]['interface']['gn_task']['goal_source_sha256']='f'*64
    elif bad=='guard':next(x for x in e if x['kind']=='guard_receipt' and x['phase']=='plan')['accepted']=False
    elif bad=='snapshot':row['checkpoints'][1]['snapshot']['state'][0]+=1.
    elif bad=='history':t[0]['task_contract']['pixel_sha256'][1]='0'*64
    elif bad=='raw':next(x for x in e if x['kind']=='warmup_return')['raw_float32_bits']='bad'
    elif bad=='no_close':e[:]=[x for x in e if x['kind']!='branch_closed']
    elif bad=='truncated':row['truncated']=True
    elif bad=='tail':r['journal_chain_tail']='f'*64
    elif bad=='firsterror':row['first_error']=dict(message='x')
    else:row['checkpoints'].append(deepcopy(row['checkpoints'][1]))
    with pytest.raises(ValueError):m.assemble(r,e,p,t,s)


@pytest.mark.parametrize('damage',('none','commands','scalar','task','omit_extra','projection_source'))
def test_projected_handoff_uses_original_sealed_prediction(damage):
    _,_,p,_,selected=fixture();plan=p['plans'][0];chosen=selected[0,m.ARMS[0]]
    authority=deepcopy({k:v for k,v in chosen.items() if k!='prediction'})
    h=dict(data=dict(plans=[authority]),projection=dict(omitted_plan_fields=['prediction'],source_report_sha256=m.PLANNER_REPORT_SHA))
    if damage=='commands':authority['commands'][0][0]=.5
    elif damage=='scalar':authority['model_terminal_cost']+=1.
    elif damage=='task':authority['task_sha256']='0'*64
    elif damage=='omit_extra':h['projection']['omitted_plan_fields'].append('commands')
    elif damage=='projection_source':h['projection']['source_report_sha256']='0'*64
    if damage=='none':m.handoff_plan(h,plan,chosen,m.PLANNER_REPORT_SHA)
    else:
        with pytest.raises(ValueError):m.handoff_plan(h,plan,chosen,m.PLANNER_REPORT_SHA)


def build_fixture(monkeypatch):
    r,e,p,t,selected=fixture();h='a'*64;calls=[]
    p['sourcepack_sha256']=r['sourcepack_sha256']=m.SOURCEPACK_SHA
    e[0]['prepare']=deepcopy(p)
    manifest=dict(schema=m.physical.NAME+'.selection',arms=m.ARMS,plans=deepcopy(p['plans']),roots=p['source_slots'],sourcepack_sha256=m.SOURCEPACK_SHA)
    handoff=dict(data=dict(plans=[{k:v for k,v in selected[s,a].items() if k!='prediction'} for s in (0,1) for a in m.ARMS]),
        projection=dict(omitted_plan_fields=['prediction'],source_report_sha256=m.PLANNER_REPORT_SHA))
    saved={str(m.PHYSICAL_REPORT):r,str(m.PHYSICAL_REPORT.parent/'execution_binding.json'):
        dict(prepared='/RAM/prepared.json',prepare_sha256=m.PHYSICAL_PREPARE_SHA),
        '/RAM/prepared.json':p,'/RAM/manifest.json':manifest,'/RAM/handoff.json':handoff,
        str(m.PLANNER_REPORT):dict(prepared='/RAM/planner_prepare.json',prepare_sha256=h,artifact_sha256={})}
    for s,path in enumerate(m.EXPORTS):
        saved[path]=dict(source_slot=s,blind_sha256=t[s]['blind_sha256'],blind_source_sha256=t[s]['blind_sha256'],task_contract=t[s]['task_contract'])
    def read(path,expected=None):
        calls.append(('read',str(path)));assert 'sourcepack' not in str(path) and 'certificate' not in str(path)
        return deepcopy(saved[str(path)])
    def sha(path):
        return {str(m.physical.__file__):m.PHYSICAL_SHA,str(m.moving.__file__):m.MOVING_SHA,str(m.planner.__file__):m.PLANNER_SHA,
            str(m.rgb.__file__):m.RGB_SHA,str(m.rgb.old.__file__):m.rgb.OLD_SHA,str(m.PHYSICAL_REPORT):m.PHYSICAL_REPORT_SHA,
            str(m.PLANNER_REPORT):m.PLANNER_REPORT_SHA,'/RAM/prepared.json':m.PHYSICAL_PREPARE_SHA,
            '/RAM/handoff.json':'b'*64,**dict(zip(m.EXPORTS,m.EXPORT_SHAS)),**m.enc.MODELS,
            str(m.PHYSICAL_REPORT.parent/'journal.jsonl'):r['journal_sha256']}.get(str(path),'x'*64 if str(path).endswith('.png') else h)
    # Real ArrayEnv PNG hashes differ per file. Resolve the saved receipt map, not a model forward.
    png_hashes={str(m.PHYSICAL_REPORT.parent/c['png']):c['file_sha256'] for row in r['rows'] for c in row['checkpoints']}
    monkeypatch.setattr(m,'sha',lambda path:png_hashes.get(str(path),sha(path)))
    monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(m.planner,'consume',lambda *a:dict(schema=m.planner.NAME+'.consume',status='moving_commands_verified_not_admitted',
        plans=list(selected.values()),new_physics=0,P1_admission=False))
    monkeypatch.setattr(m.planner,'verify',lambda *a:dict(bindings_sha256={}))
    def load_task(path,expected):
        calls.append(('moving.load_task',path,expected));return deepcopy(t[m.EXPORTS.index(path)]['interface'])
    monkeypatch.setattr(m.moving,'load_task',load_task)
    monkeypatch.setattr(m.rgb.planner,'load_export',lambda *a:pytest.fail('forbidden old zero-only loader'))
    monkeypatch.setattr(m.physical,'selection',lambda *a:(deepcopy(p['plans']),{'/RAM/handoff.json':'b'*64}))
    monkeypatch.setattr(m.physical,'journal',lambda *a:deepcopy(e))
    monkeypatch.setattr(m.physical,'load_source',lambda *a:pytest.fail('no evaluator loader'))
    monkeypatch.setattr(m.rgb.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    monkeypatch.setattr(m.moving,'source_files',lambda:[])
    return saved,p,calls


def test_build_uses_moving_load_task_and_explicit_physical_chain(monkeypatch):
    _,_,calls=build_fixture(monkeypatch);p=m.build()
    assert len(p['images'])==8 and p['contract']['encoder']==p['contract']['projector']==8
    assert [c for c in calls if c[0]=='moving.load_task']==[('moving.load_task',f,h) for f,h in zip(m.EXPORTS,m.EXPORT_SHAS)]
    assert p['contract']['predict']==p['contract']['physics']==p['contract']['AD']==0


@pytest.mark.parametrize('damage',('prepare','manifest','handoff','source','loader'))
def test_build_failure_before_any_encoding(damage,monkeypatch):
    saved,p,_=build_fixture(monkeypatch)
    if damage=='prepare':saved[str(m.PHYSICAL_REPORT.parent/'execution_binding.json')]['prepare_sha256']='0'*64
    elif damage=='manifest':saved['/RAM/manifest.json']['arms']=['GN','D2P']
    elif damage=='handoff':saved['/RAM/handoff.json']['data']['plans'][0]['model_terminal_cost']+=1.
    elif damage=='source':p['sourcepack_sha256']='0'*64
    else:
        original=m.sha
        monkeypatch.setattr(m,'sha',lambda path:'0'*64 if str(path)==str(m.moving.__file__) else original(path))
    with pytest.raises(ValueError):m.build()


def test_eight_encodes_no_dedup_and_float32_NPY(monkeypatch):
    images=m.assemble(*fixture())
    for item in images:item['png_path']='/RAM/same.png'
    monkeypatch.setattr(m.rgb.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    store=RAMStore();ledger=m.enc.Ledger(2,store.event);rows=[dict(status='not_started') for _ in images]
    def callback(a):
        for resource,shape in (('encoder',(1,3,224,224)),('projector',(1,192))):
            ledger.dispatch(resource,NS(device=NS(type='cuda'),shape=shape),lambda:None)
        return np.full(192,4,np.float32)
    m.encode_images(dict(images=images),callback,ledger,store,rows)
    assert ledger.counts()=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')}
    assert len(store.files)==8 and all(r['status']=='encoded' for r in rows)
    assert all(np.load(BytesIO(raw),allow_pickle=False).dtype==np.float32 for raw in store.files.values())
    d=rows[-1]['metrics'];assert d['actual_Q']==1 and d['root_Q']==1 and d['predicted_Q']==.0625
    assert d['actual_minus_predicted_Q']==.9375 and d['predicted_vs_real_mse']==1.5625


def test_partial_attempt_no_retry_or_advancing(monkeypatch):
    images=m.assemble(*fixture())
    for item in images:item['png_path']='/RAM/a.png'
    monkeypatch.setattr(m.rgb.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    store=RAMStore();ledger=m.enc.Ledger(2,store.event);rows=[dict(status='not_started') for _ in images]
    def fail(a):
        ledger.dispatch('encoder',NS(device=NS(type='cuda'),shape=(1,3,224,224)),lambda:None)
        raise RuntimeError('mock projector')
    with pytest.raises(RuntimeError):m.encode_images(dict(images=images),fail,ledger,store,rows)
    assert rows[0]['status']=='attempted' and all(r['status']=='not_started' for r in rows[1:]) and not store.files
    assert ledger.counts()['encoder']==dict(attempted=1,completed=1)


def test_run_verify_before_store_or_neural(monkeypatch):
    def reject(*a):raise ValueError('binding')
    monkeypatch.setattr(m,'verify',reject);monkeypatch.setattr(m.rgb,'Store',lambda *a:pytest.fail('before verify'))
    with pytest.raises(ValueError,match='binding'):m.run('x','x','x')


def test_verify_full_rebuild(monkeypatch):
    p=dict(arguments=dict(),images=[])
    monkeypatch.setattr(m,'read',lambda *a:deepcopy(p));monkeypatch.setattr(m,'build',lambda **k:deepcopy(p))
    assert m.verify('RAM','a'*64)==p
    monkeypatch.setattr(m,'build',lambda **k:dict(p,changed=True))
    with pytest.raises(ValueError):m.verify('RAM','a'*64)


def test_prepare_disk_floor_before_write(monkeypatch):
    class Path:
        parent='RAM'
        def resolve(self):return self
        def exists(self):return False
        def open(self,*a):pytest.fail('no writes with insufficient disk')
    monkeypatch.setattr(m,'build',lambda **k:dict(images=[]))
    monkeypatch.setattr(m,'Path',lambda *a:Path())
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:NS(free=m.rgb.FLOOR))
    with pytest.raises(ValueError,match='200MiB'):m.prepare('RAM')


def test_source_seals_small_files_only():
    assert m.sha(m.physical.__file__)==m.PHYSICAL_SHA
    assert m.sha(m.moving.__file__)==m.MOVING_SHA
    assert m.sha(m.planner.__file__)==m.PLANNER_SHA
