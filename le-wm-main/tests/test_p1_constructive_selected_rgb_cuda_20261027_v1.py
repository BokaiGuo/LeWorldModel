"""Pure NumPy/RAM provenance, encoding-meter and failure tests; no neural/physics."""
from copy import deepcopy
from io import BytesIO
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_selected_rgb_cuda_20261027_v1 as m


def fixture():
    spec=importlib.util.spec_from_file_location('selected_rgb_gn_fixture',Path(__file__).with_name('test_p1_line_search_gn_cuda_20261027_v1.py'))
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    t=g.task();t['goal']=np.asarray(t['goal'],np.float32).astype(float).tolist()
    i=dict(gn_task=t,task_id=m.enc.core.task_id(t));raw=np.zeros((10,2),np.float32)
    paths=g.Backend().path(t,raw)
    base=dict(task_id=i['task_id'],task_export='/mock/task_00.json.gz',task_sha256='a'*64,commands=raw.tolist(),logits=np.zeros((2,10)).tolist(),prediction=paths[-1],model_terminal_cost=m.enc.core.model_cost(t,paths[-1]['latent_history'][-1]))
    state=[100.,100.,200.,200.,0.,0.,0.];goal=[110.,110.,220.,220.,.1,0.,0.]
    root_hash=m.enc.digest(np.ascontiguousarray(state+goal,dtype=np.float64).tobytes())
    snapshots=dict(state=state,block_velocity=[0.,0.],block_angular_velocity=0.,block_angle=0.)
    counts={k:dict(attempted=v,completed=v) for k,v in dict(reset=8,public_step=80,explicit_render=88,goal_render=8,set_state=16,physics_step=816,goal_bind=4).items()}
    contract=dict(with_target=False,own_horizon=10,branches=4)
    prepared=dict(schema=m.PHYSICAL+'.prepare',contract=contract,sourcepack_sha256='c'*64,plans=[],
        source_slots=[dict(root_slot=r,root_sha256=root_hash) for r in range(2)])
    physical=dict(schema=m.PHYSICAL+'.report',status='physical_calibration_recorded',errors=[],P1_admission=False,
        contract=contract,sourcepack_sha256='c'*64,counts=counts,rows=[])
    events=[];plans=[];tasks=[]
    for b,(r,a) in enumerate((r,a) for r in range(2) for a in m.planner.ARMS):
        plan=dict(base,arm=a);plans.append(plan);tasks.append(deepcopy(i))
        selection=dict(root_slot=r,arm=a,commands=raw.tolist());selection['selection_digest']=m.enc.core.digest(selection);prepared['plans'].append(selection)
        checkpoints=[dict(phase='plan',step=step,png=f'b{b}_plan_{step:02d}.png',file_sha256='1'*64,pixel_sha256='2'*64,
            state=state,snapshot=snapshots,observation_state=state) for step in (0,5,10)]
        row=dict(root_slot=r,arm=a,status='horizon_completed',first_error=None,secondary_errors=[],initial_exact=True,
            full_horizon=True,own_attempted=10,own_returned=10,warmup_returned=10,truncated=False,
            selection_digest=selection['selection_digest'],checkpoints=checkpoints,root_state=state,final_state=state,goal=goal)
        physical['rows'].append(row)
        events.append(dict(kind='branch_started',branch=b,plan=selection))
        events.append(dict(kind='goal_attached',branch=b,goal=goal,goal_rgb_sha256=t['goal_source_sha256'],snapshot=snapshots))
        for step in range(1,11):
            events.append(dict(kind='returned_transition',branch=b,phase='plan',step=step,command=raw[step-1].tolist(),state=state,
                returned_goal_state=goal,returned_goal_rgb_sha256=t['goal_source_sha256']))
        for c in checkpoints:
            events.append(dict(kind='fresh_observation',branch=b,receipt=c))
            events.append(dict(kind='guard_receipt',branch=b,phase='plan',step=c['step'],accepted=True,png=c['png'],file_sha256=c['file_sha256']))
        events.append(dict(kind='branch_closed',receipt=row))
    for k,c in counts.items():
        for _ in range(c['completed']):
            events.extend([dict(kind='operation_attempted',resource=k),dict(kind='operation_completed',resource=k)])
    events.insert(0,dict(kind='run_started',prepare=prepared));events.append(dict(kind='run_finished',status=physical['status'],counts=counts,errors=[]))
    # Break shared fixture aliases, as in a parsed saved journal.
    return deepcopy((physical,events,prepared,{},plans,tasks))


def test_all_four_branches_exact_eight_images_no_candidate_access():
    images=m.assemble(*fixture())
    assert [(i['root_slot'],i['arm'],i['role']) for i in images]==[(r,a,role) for r in range(2) for a in m.planner.ARMS for role in ('prefix5','terminal10')]
    assert len(images)==8 and all(i['Q']=='identity192' for i in images)


@pytest.mark.parametrize('bad', ['partial','missing5','missing10','duplicate5','state','snapshot','root','goal','goalRGB','commands','selection','guard','journal','census','wrong_arm'])
def test_provenance_and_partial_fail_closed(bad):
    p,e,prep,pr,plans,tasks=fixture();row=p['rows'][0]
    if bad=='partial':row['full_horizon']=False;row['own_returned']=7
    elif bad.startswith('missing'):row['checkpoints']=[c for c in row['checkpoints'] if c['step']!=int(bad[7:])]
    elif bad=='duplicate5':row['checkpoints'].append(deepcopy(row['checkpoints'][1]))
    elif bad=='state':row['checkpoints'][1]['observation_state']=[9.]*7
    elif bad=='snapshot':row['checkpoints'][1]['snapshot']['state']=[9.]*7
    elif bad=='root':prep['source_slots'][0]['root_sha256']='f'*64
    elif bad=='goal':row['goal']=[9.]*7
    elif bad=='goalRGB':tasks[0]['gn_task']['goal_source_sha256']='f'*64
    elif bad=='commands':plans[0]['commands'][0][0]=.1
    elif bad=='selection':row['selection_digest']='f'*64
    elif bad=='guard':next(x for x in e if x['kind']=='guard_receipt' and x['step']==5)['accepted']=False
    elif bad=='journal':e.pop(next(j for j,x in enumerate(e) if x['kind']=='returned_transition'))
    elif bad=='census':p['counts']['physics_step']['completed']-=1
    elif bad=='wrong_arm':row['arm']='random'
    with pytest.raises(ValueError):m.assemble(p,e,prep,pr,plans,tasks)


def test_selected_context_chronology_not_just_terminal_cost():
    *_,plans,tasks=fixture();p=deepcopy(plans[0]);p['prediction']['latent_history'][0][0]+=1
    with pytest.raises(ValueError):m.selected_task(p,tasks[0])


def test_comparison_identity_q_and_reversal():
    item=dict(predicted=[.5]*192,root=[1.]*192,goal=[0.]*192)
    metrics=m.compare(item,np.full(192,2,np.float32))
    assert metrics['predicted_Q']==.25 and metrics['actual_Q']==4 and metrics['root_Q']==1
    assert metrics['Q_improvement_sign_reversed'] is True and metrics['predicted_vs_real_mse']==2.25


class RAMStore:
    def __init__(self):self.events=[];self.files={}
    def event(self,e):self.events.append(deepcopy(e))
    def latent(self,index,z):
        b=BytesIO();np.save(b,z,allow_pickle=False);self.files[index]=b.getvalue()
        return dict(file=str(index),sha256=m.enc.digest(b.getvalue()))


def encode_fixture(monkeypatch,callback_error=False):
    images=m.assemble(*fixture())
    for item in images:item['png_path']='/mock/'+item['checkpoint']['png']
    monkeypatch.setattr(m,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    store=RAMStore();ledger=m.enc.Ledger(2,store.event);rows=[dict(status='not_started') for _ in images]
    tensor=lambda shape:NS(device=NS(type='cuda'),shape=shape)
    def callback(rgb):
        assert rgb.shape==(224,224,3) and rgb.dtype==np.uint8
        ledger.dispatch('encoder',tensor((1,3,224,224)),lambda:None)
        if callback_error:raise RuntimeError('between encoder/projector')
        ledger.dispatch('projector',tensor((1,192)),lambda:None)
        return np.zeros(192,np.float32)
    return dict(images=images),callback,ledger,store,rows


def test_eight_encodes_actual_mock_dispatch_and_lossless_arrays(monkeypatch):
    p,cb,l,s,rows=encode_fixture(monkeypatch);m.encode_images(p,cb,l,s,rows)
    assert l.counts()=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')}
    assert len(s.files)==8 and all(r['status']=='encoded' for r in rows)
    for raw in s.files.values():assert np.array_equal(np.load(BytesIO(raw),allow_pickle=False),np.zeros(192,np.float32))


def test_partial_encoder_failure_no_next_image_or_fabricated_array(monkeypatch):
    p,cb,l,s,rows=encode_fixture(monkeypatch,True)
    with pytest.raises(RuntimeError):m.encode_images(p,cb,l,s,rows)
    assert l.counts()['encoder']==dict(attempted=1,completed=1)
    assert l.counts()['projector']==dict(attempted=0,completed=0)
    assert not s.files and rows[1]['status']=='not_started'


def test_no_unmetered_callback_success(monkeypatch):
    p,cb,l,s,rows=encode_fixture(monkeypatch)
    with pytest.raises(ValueError):m.encode_images(p,lambda rgb:np.zeros(192,np.float32),l,s,rows)
    assert not s.files


def test_native_dtype_and_path_guards():
    with pytest.raises(ValueError):m.compare(dict(predicted=[0.]*192,root=[0.]*192,goal=[0.]*192),np.zeros(192,np.float64))
    for name in ('../certificate.npz','/tmp/foo','..'):
        with pytest.raises(ValueError):m.local(Path('/mock'),name)


def test_png_no_color_conversion_and_pixel_provenance(monkeypatch):
    from PIL import Image
    rgb=np.zeros((224,224,3),np.uint8);b=BytesIO();Image.fromarray(rgb).save(b,format='PNG');data=b.getvalue()
    original=Image.open
    monkeypatch.setattr(m,'sha',lambda p:'1'*64)
    monkeypatch.setattr(Path,'stat',lambda p:NS(st_size=len(data)))
    monkeypatch.setattr(Image,'open',lambda p:original(BytesIO(data)))
    assert np.array_equal(m.png('/mock/a.png','1'*64,m.enc.digest(rgb.tobytes())),rgb)
    with pytest.raises(ValueError):m.png('/mock/a.png','1'*64,'f'*64)


@pytest.mark.parametrize('damage',['chain','torn','tail','file_sha'])
def test_physical_journal_fail_closed_RAM(monkeypatch,damage):
    events=[dict(index=0,previous_sha256=None,kind='run_started',prepare={})]
    events[0]['event_sha256']=m.enc.core.digest(events[0])
    e=dict(index=1,previous_sha256=events[0]['event_sha256'],kind='run_finished')
    e['event_sha256']=m.enc.core.digest(e);events.append(e)
    if damage=='chain':events[1]['previous_sha256']='f'*64
    if damage=='tail':events[1]['kind']='operation_attempted'
    raw=b''.join(m.canonical(e) for e in events)
    if damage=='torn':raw=raw[:-1]
    monkeypatch.setattr(m,'sha',lambda path:'1'*64)
    monkeypatch.setattr(Path,'stat',lambda path:NS(st_size=len(raw)))
    monkeypatch.setattr(Path,'read_bytes',lambda path:raw)
    with pytest.raises(ValueError):m.journal('/mock/journal.jsonl','2'*64 if damage=='file_sha' else '1'*64)


def test_encoder_sync_fault_retains_attempt_without_completion():
    l=m.enc.Ledger(2,lambda e:None,sync=lambda:(_ for _ in ()).throw(RuntimeError('async failed')))
    l.image=0;t=NS(device=NS(type='cuda'),shape=(1,3,224,224))
    with pytest.raises(RuntimeError):l.dispatch('encoder',t,lambda:None)
    assert l.counts()['encoder']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError):l.dispatch('encoder',t,lambda:None)


def test_forbidden_predict_no_callback_dispatch():
    l=m.enc.Ledger(2,lambda e:None)
    with pytest.raises(ValueError):l.forbid('predict')()
    assert l.blocked==['predict'] and l.counts()['encoder']['attempted']==0


def test_no_neural_simulator_imports():
    assert not {'torch','gymnasium','pymunk','stable_worldmodel'} & set(sys.modules)
