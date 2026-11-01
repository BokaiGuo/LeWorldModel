"""Numerical mocks need no actual physical fixture or torch/model import."""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import p1_line_search_gn_rgb_cuda_20261027_v1 as m


def fixture(folder):
    rows=[]; plans=[]; tasks=[]; predictions=[]
    for index in range(2):
        task=dict(root_context=dict(latent_history=np.zeros((3,192)).tolist()),goal=np.ones(192).tolist(),projection=np.eye(192).tolist())
        taskid=m.hashlib_task(task); commands=np.zeros((10,2)).tolist()
        plan=dict(root_sha256=str(index)*64,selection_sha256=str(index+2)*64,arm=m.ARM,commands=commands,task_id=taskid)
        pred=dict(commands=commands,contexts=[dict(latent_history=np.full((3,192),i).tolist()) for i in (1,2)],task_id=taskid)
        checkpoints=[dict(step=s,path=str(folder/f'{index}_{s}.png'),file_sha256='a'*64,rgb_sha256='b'*64,state=[s]*7) for s in (0,5,10)]
        trace=[dict(step=s,attempted=True,returned=True,command=commands[s-1],actual_state=[s]*7) for s in range(1,11)]
        row=dict(root_sha256=plan['root_sha256'],selection_sha256=plan['selection_sha256'],arm=m.ARM,
                 status='completed',full_horizon=True,initial_exact=True,first_error=None,secondary_errors=[],
                 returned_steps=10,attempted_steps=10,checkpoints=checkpoints,trace=trace)
        rows.append(row); plans.append(plan); tasks.append(task); predictions.append(pred)
    return rows,plans,tasks,predictions


def packet(folder):
    frames=m.frames_from_rows(*fixture(folder),folder,check_pixels=lambda *args:None)
    metrics=[]
    for index in range(2):
        f=frames[index*3]; metrics.append({k:deepcopy(f[k]) for k in ('root_index','root_sha256','task_id','q','goal')})
    for f in frames: del f['q']; del f['goal']
    return dict(frames=frames,root_metrics=metrics)


def tensor(kind='encoder',device='cuda'):
    return NS(device=NS(type=device),shape=(1,3,224,224) if kind=='encoder' else (1,192))


def encode_with(ledger):
    def encode(f):
        ledger.dispatch('encoder',tensor(),lambda:None)
        ledger.dispatch('projector',tensor('projector'),lambda:None)
        return np.full(192,f['step'],np.float32)
    return encode


def test_full_six_duplicates_paid_and_raw_NPY_saved_before_analysis(tmp_path):
    p=packet(tmp_path); store=m.Store(tmp_path/'out'); events=[]; ledger=m.Ledger(events.append)
    rows=[]; values=[]; m.encode_frames(p,encode_with(ledger),store,ledger,rows,values)
    assert ledger.counts()=={k:dict(attempted=6,completed=6) for k in ('encoder','projector')}
    assert len(rows)==len(values)==6 and all(r['status']=='saved' for r in rows)
    assert len({f['pixel_sha256'] for f in p['frames']})==1  # all input hashes same, still six calls/NPYs
    for row,z in zip(rows,values):
        saved=np.load(row['latent']['path'],allow_pickle=False)
        assert saved.dtype==np.float32 and saved.shape==(192,) and np.array_equal(saved,z)
        assert m.sha(row['latent']['path'])==row['latent']['sha256']
    analysis=m.analyze(p['frames'],values,p['root_metrics'])
    assert len(analysis['rows'])==6 and not analysis['P1_admission']
    assert analysis['rows'][0]['actual_Q_mse']==1. and analysis['rows'][2]['actual_Q_mse']==81.


@pytest.mark.parametrize('field', ['partial','wrongarm','wrongroot','steps','historytask','command','checkpoint','state','path','order'])
def test_frame_contract_rejects_bad_or_old_physical_inputs(tmp_path,field):
    rows,plans,tasks,predictions=fixture(tmp_path)
    if field=='partial': rows[0]['full_horizon']=False
    elif field=='wrongarm': rows[0]['arm']='B8'
    elif field=='wrongroot': rows[0]['root_sha256']='f'*64
    elif field=='steps': rows[0]['returned_steps']=9
    elif field=='historytask': tasks[0]['goal'][0]+=1
    elif field=='command': rows[0]['trace'][0]['command']=[.1,0.]
    elif field=='checkpoint': rows[0]['checkpoints'].pop()
    elif field=='state': rows[0]['checkpoints'][1]['state'][0]+=1
    elif field=='path': rows[0]['checkpoints'][1]['path']=str(tmp_path/'old'/'endpoint.png')
    elif field=='order': rows[0]['checkpoints'].reverse()
    with pytest.raises(ValueError): m.frames_from_rows(rows,plans,tasks,predictions,tmp_path,check_pixels=lambda *args:None)


def test_pixels_checked_for_all_six_not_unique_hash(tmp_path):
    calls=[]
    def check(*args): calls.append(args)
    m.frames_from_rows(*fixture(tmp_path),tmp_path,check_pixels=check)
    assert len(calls)==6


def test_projector_fault_keeps_encoder_return_and_no_retry(tmp_path):
    p=packet(tmp_path); ledger=m.Ledger(lambda e:None); store=m.Store(tmp_path/'out'); rows=[]; values=[]
    def fail(): raise RuntimeError('projector fault')
    def encode(f):
        ledger.dispatch('encoder',tensor(),lambda:None)
        ledger.dispatch('projector',tensor('projector'),fail)
    with pytest.raises(RuntimeError,match='projector fault'): m.encode_frames(p,encode,store,ledger,rows,values)
    assert ledger.counts()==dict(encoder=dict(attempted=1,completed=1),projector=dict(attempted=1,completed=0))
    assert len(rows)==1 and rows[0]['status']=='failed' and not values
    ledger.frame=0
    with pytest.raises(ValueError): ledger.dispatch('encoder',tensor(),lambda:None)


def test_sync_failure_is_uncompleted():
    def fail(): raise OSError('sync')
    ledger=m.Ledger(lambda e:None,sync=fail); ledger.frame=0
    with pytest.raises(OSError): ledger.dispatch('encoder',tensor(),lambda:None)
    assert ledger.counts()['encoder']==dict(attempted=1,completed=0)


@pytest.mark.parametrize('reason',['cpu','wrongshape','grad','outoforder','cap'])
def test_predispatch_checks(reason):
    ledger=m.Ledger(lambda e:None); ledger.frame=0; t=tensor(); kind='encoder'
    if reason=='cpu': t=tensor(device='cpu')
    elif reason=='wrongshape': t.shape=(2,3,224,224)
    elif reason=='grad': ledger.no_grad=lambda:False
    elif reason=='outoforder': kind='projector'; t=tensor(kind)
    elif reason=='cap': ledger.entries=[dict(frame=i//2,resource=['encoder','projector'][i%2],completed=True) for i in range(12)]
    count=len(ledger.entries)
    with pytest.raises(ValueError): ledger.dispatch(kind,t,lambda:None)
    assert len(ledger.entries)==count


@pytest.mark.parametrize('bad',[np.zeros(192,dtype=np.float64),np.zeros((1,192),np.float32),np.full(192,np.nan,np.float32)])
def test_no_silent_cast_or_invalid_latent(bad):
    with pytest.raises(ValueError): m.native(bad)


def test_six_root_order_and_fixed_Q_no_crossmix(tmp_path):
    p=packet(tmp_path); values=[np.zeros(192,np.float32) for _ in range(6)]
    p['root_metrics'][0]['root_sha256']='bad'
    with pytest.raises(ValueError): m.analyze(p['frames'],values,p['root_metrics'])
    p=packet(tmp_path); p['frames'].reverse()
    with pytest.raises(ValueError): m.analyze(p['frames'],values,p['root_metrics'])


def test_NPY_fsync_failure_preserves_file_no_false_save(tmp_path,monkeypatch):
    p=packet(tmp_path); store=m.Store(tmp_path/'out'); ledger=m.Ledger(lambda e:None); rows=[]; values=[]
    real=m.os.fsync; calls=[0]
    def fsync(fd):
        calls[0]+=1
        if calls[0]==2: raise OSError('NPY fsync fault')  # frame_attempt fsync then NPY fsync
        real(fd)
    monkeypatch.setattr(m.os,'fsync',fsync)
    with pytest.raises(OSError): m.encode_frames(p,encode_with(ledger),store,ledger,rows,values)
    assert rows[0]['status']=='failed' and not values and (store.output/'frame_00.native192.npy').exists()
    assert ledger.counts()=={k:dict(attempted=1,completed=1) for k in ('encoder','projector')}


def test_forbidden_predict_or_AD_never_runs():
    events=[]; ledger=m.Ledger(events.append)
    with pytest.raises(ValueError): ledger.forbid('AD.grad')()
    assert ledger.blocked==['AD.grad'] and events[-1]['kind']=='forbidden'


def test_restore_failure_does_not_replace_primary():
    def old(*args,**kwargs): return None
    class Module:
        def __init__(self): self.forward=old
    class Sticky(Module):
        def __setattr__(self,name,value):
            if name=='forward' and hasattr(self,'forward') and value is old: raise RuntimeError('restore fault')
            object.__setattr__(self,name,value)
    model=NS(encoder=Sticky(),projector=Module(),predict=old,predictor=Module(),action_encoder=Module(),pred_proj=Module())
    torch=NS(autograd=NS(grad=old,backward=old,functional=NS(**{k:old for k in ('jacobian','hessian','jvp','vjp','hvp','vhp')})),
             Tensor=NS(backward=old),func=NS(**{k:old for k in ('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value')}))
    errors=[]
    with pytest.raises(RuntimeError,match='primary'):
        with m.guard(model,torch,m.Ledger(lambda e:None),errors): raise RuntimeError('primary')
    assert len(errors)==1 and errors[0]['stage']=='restore_forward'
    assert model.projector.forward is old


def test_actual_saved_pair_and_small_metadata_only():
    # Added after MAIN supplied actual hashes; all other unit tests need no actual fixture.
    p=m.collect()
    assert p['arguments']['physical_report_sha']==m.PHYSICAL_SHA
    assert p['arguments']['physical_prepare_sha']==m.PHYSICAL_PREPARE_SHA
    assert len(p['frames'])==6 and len(p['root_metrics'])==2 and len(m.canonical(p))<512*1024
    assert p['contract']['encoder_cap']==p['contract']['projector_cap']==6
    assert [f['step'] for f in p['frames']]==[0,5,10,0,5,10]
    assert not p['contract']['new_Q'] and p['contract']['physics_cap']==0
