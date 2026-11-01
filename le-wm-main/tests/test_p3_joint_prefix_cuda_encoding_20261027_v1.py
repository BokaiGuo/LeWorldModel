"""Saved-data and interface fakes only; no model/tensor/environment execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p3_joint_prefix_cuda_encoding_20261027_v1 as m


def tensor(kind='encoder',device='cuda'):
    return SimpleNamespace(device=SimpleNamespace(type=device),shape=(1,3,224,224) if kind=='encoder' else (1,192))


@pytest.fixture(scope='module')
def protocol():return m.collect()


def test_actual_all_branch_census_and_chronology(protocol):
    frames=protocol['frames']
    assert len(frames)==17 and len({(x['root_sha256'],x['arm']) for x in frames})==16
    assert sum(x['primitive_step']==5 for x in frames)==16
    extra=[x for x in frames if x['primitive_step']==10]
    assert len(extra)==1 and extra[0]['root_index']==1 and extra[0]['arm']=='goal_patch_Q__uniform'
    assert extra[0]['predicted_history_index']==1
    for x in frames:
        assert x['predicted_history_index']==3-x['horizon']+x['primitive_step']//5-1
    assert protocol['contract']['encoder_cap']==protocol['contract']['existing_projector_cap']==17
    assert protocol['contract']['predict_cap']==protocol['contract']['AD_cap']==protocol['contract']['physics_cap']==0
    assert protocol['contract']['remaining_disk_reserve_bytes']==200*1024*1024


def tiny():
    return dict(frames=[dict(frame_id='frame_'+str(i),root_sha256='root',arm='arm',primitive_step=5) for i in range(17)])


def test_full_nonrefundable_dispatch():
    p=tiny();events=[];ledger=m.Ledger(p['frames'],events.append)
    for f in p['frames']:
        ledger.active=f['frame_id']
        for kind in ('encoder','existing_projector'):assert ledger.dispatch(kind,tensor(kind),lambda:7,(),{})==7
    assert ledger.counts()=={k:dict(attempted=17,completed=17) for k in ('encoder','existing_projector')}
    assert len(events)==68
    with pytest.raises(ValueError):ledger.dispatch('encoder',tensor(),lambda:0,(),{})


@pytest.mark.parametrize('damage',['cpu','shape','wrong_frame','projector_first'])
def test_reject_before_callback(damage):
    ledger=m.Ledger(tiny()['frames'],lambda e:None);ledger.active='frame_0';x=tensor();kind='encoder';calls=[]
    if damage=='cpu':x.device.type='cpu'
    elif damage=='shape':x.shape=(2,3,224,224)
    elif damage=='wrong_frame':ledger.active='frame_1'
    else:kind='existing_projector';x=tensor(kind)
    with pytest.raises(ValueError):ledger.dispatch(kind,x,lambda:calls.append(1),(),{})
    assert not calls and not ledger.entries and ledger.blocked


@pytest.mark.parametrize('kind',['predict','predictor','action_encoder','pred_proj','AD.grad','AD.jvp','AD.func.jacrev'])
def test_forbidden_entries(kind):
    ledger=m.Ledger(tiny()['frames'],lambda e:None)
    with pytest.raises(RuntimeError):ledger.forbid(kind)()
    assert ledger.blocked[0]['kind']==kind


def test_failed_call_and_failed_error_journal_preserve_first_error():
    count=0
    def journal(e):
        nonlocal count
        count+=1
        if count==2:raise OSError('secondary journal failure')
    ledger=m.Ledger(tiny()['frames'],journal);ledger.active='frame_0'
    def fail():raise RuntimeError('first callback error')
    with pytest.raises(RuntimeError,match='first callback'):ledger.dispatch('encoder',tensor(),fail,(),{})
    assert ledger.entries[0]['error']['message']=='first callback error'
    assert ledger.journal_errors[0]['message']=='secondary journal failure'
    with pytest.raises(ValueError):ledger.dispatch('encoder',tensor(),lambda:0,(),{})


def test_attempt_journal_failure_never_dispatches():
    def journal(e):raise OSError('attempt unavailable')
    calls=[];ledger=m.Ledger(tiny()['frames'],journal);ledger.active='frame_0'
    with pytest.raises(OSError):ledger.dispatch('encoder',tensor(),lambda:calls.append(1),(),{})
    assert not calls and len(ledger.entries)==1 and not ledger.entries[0]['completed']


def test_async_failure_not_completed():
    def sync():raise RuntimeError('asynchronous CUDA error fake')
    ledger=m.Ledger(tiny()['frames'],lambda e:None,sync);ledger.active='frame_0'
    with pytest.raises(RuntimeError):ledger.dispatch('encoder',tensor(),lambda:1,(),{})
    assert not ledger.entries[0]['completed']


def test_pipeline_saves_all_native_arrays_first(tmp_path):
    p=tiny();events=[];ledger=m.Ledger(p['frames'],events.append);rows=[];errors=[]
    def encode(frame):
        for kind in ('encoder','existing_projector'):ledger.dispatch(kind,tensor(kind),lambda:None,(),{})
        return np.ones(192,np.float32)
    m.encode_frames(p,tmp_path,encode,ledger,events.append,rows,errors)
    assert not errors and len(rows)==17 and all(r['status']=='encoded_and_saved' for r in rows)
    for r in rows:
        a=np.load(r['latent_path'],allow_pickle=False);assert a.dtype==np.float32 and a.shape==(192,)
        assert m.sha(r['latent_path'])==r['latent_sha256']
    assert len(list(tmp_path.glob('*.attempt.json')))==len(list(tmp_path.glob('*.result.json')))==17


@pytest.mark.parametrize('damage',['encode','invalid','save','result_journal'])
def test_partial_census_stops_without_retry(tmp_path,monkeypatch,damage):
    p=tiny();ledger=m.Ledger(p['frames'],lambda e:None);rows=[];errors=[]
    def encode(frame):
        if damage=='encode':raise RuntimeError('first encode')
        return np.zeros(192,np.float64 if damage=='invalid' else np.float32)
    if damage=='save':
        def bad(*args):raise OSError('save failure')
        monkeypatch.setattr(m,'save_native',bad)
    def journal(e):
        if damage=='result_journal' and e['event']=='endpoint_saved':raise OSError('result journal failure')
    m.encode_frames(p,tmp_path,encode,ledger,journal,rows,errors)
    assert len(rows)==17 and errors and all(r['status']=='blocked_not_attempted' for r in rows[1:])
    if damage=='result_journal':assert Path(rows[0]['latent_path']).is_file() and rows[0]['latent_sha256']


def test_restore_attempts_all_and_retains_errors():
    class Broken:
        value=1
        def __setattr__(self,key,value):
            if value==1:raise RuntimeError('restore unavailable')
            object.__setattr__(self,key,value)
    obj=Broken();other=SimpleNamespace(value=3);hooks=m.Hooks();hooks.install(other,'value',4);hooks.install(obj,'value',2)
    errors=[];hooks.restore(errors)
    assert other.value==3 and errors[0]['stage']=='hook_restore'


def test_png_pixel_and_file_hash(tmp_path):
    from PIL import Image
    path=tmp_path/'frame.png';Image.fromarray(np.zeros((224,224,3),np.uint8)).save(path)
    rgb,h=m.pixels(path);assert rgb.shape==(224,224,3)
    with pytest.raises(ValueError):m.pixels(path,'wrong',h)
    with pytest.raises(ValueError):m.pixels(path,m.sha(path),'wrong')


def test_numerical_costs_not_cross_Q():
    z=np.ones(192,np.float32);pred=np.zeros(192);g=np.zeros(192);qs={'old_Q':np.eye(192),'goal_patch_Q':np.diag([1.]+[0.]*191)}
    a=m.numeric(z,pred,g,qs)
    assert a['actual_Q_costs']==dict(old_Q=1.,goal_patch_Q=1/192)
    assert a['native_prediction_mse']==1 and a['projected_prediction_mse']==a['actual_Q_costs']
    assert 'new_Q_minus_old_Q' not in a


def test_prepare_exclusive_source_seal(tmp_path,monkeypatch):
    src=tmp_path/'source.py';src.write_text('fixture = True\n')
    p=dict(schema=m.NAME+'.protocol',contract=m.contract(),bindings={},model_bindings={})
    monkeypatch.setattr(m,'collect',lambda:deepcopy(p));monkeypatch.setattr(m,'source_files',lambda:[src])
    out=tmp_path/'prepared';m.prepare(out);h=m.sha(out/'prepare.json');assert m.verify(out,h)['contract']==m.contract()
    with pytest.raises(ValueError):m.prepare(out)
    with pytest.raises(ValueError):m.verify(out,'wrong')
    src.write_text('drift = True\n')
    with pytest.raises(ValueError):m.verify(out,h)


def test_import_no_neural_or_old_execution():
    subprocess.run([sys.executable,'-c','import sys; from experiments.spectral_reachability import '+m.NAME+
        '; assert not {"torch","gymnasium","pymunk"}.intersection(sys.modules); '
        'assert "experiments.spectral_reachability.p3_joint_horizon_goal_metric_physical_20261027_v1" not in sys.modules'],check=True)


@pytest.fixture(scope='module')
def witnesses():
    from experiments.spectral_reachability import p3_quality_cost_audit_20261027_v2 as io
    cp=m.read(m.REPO/'results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_prepared01/protocol.json')
    pp=m.read(m.REPO/'results/p3_joint_horizon_goal_metric_physical_20261027_v1_prepared01/protocol.json')
    report=m.read(m.PLANNER/'report.json');roots=io.index(report['roots'],('root_sha256',));records={x['packet']['root_sha256']:x for x in cp['records']}
    plan=next(x for x in pp['plans'] if x['planned_horizon_tokens']==3)
    return plan,roots[plan['root_sha256'],]['arms'][plan['method']],records[plan['root_sha256']],cp['normalization']


@pytest.mark.parametrize('damage',['swap_states','history','segment_order','terminal','layout','actions'])
def test_chronology_rejected_not_blind_offsets(witnesses,damage):
    plan,source,record,norm=deepcopy(witnesses)
    if damage=='swap_states':
        a=source['winner']['context']['latent_history'];a[0],a[1]=a[1],a[0]
    elif damage=='history':source['segments'][0]['action_history'][0][0]+=.01
    elif damage=='segment_order':source['segments'][0],source['segments'][1]=source['segments'][1],source['segments'][0]
    elif damage=='terminal':source['segments'][0]['latent_history'][-1]=source['segments'][-1]['latent_history'][-1]
    elif damage=='layout':source['segments'][0]['layout']='wrong'
    else:plan['commands'][0][0]+=.001
    with pytest.raises(ValueError):m.prefix_vectors(plan,source,record,norm)


def test_saved_only_consumer_complete_fixture(protocol,tmp_path,monkeypatch):
    folder=tmp_path/'fake_cuda';folder.mkdir();events=[];ledger=m.Ledger(protocol['frames'],events.append);rows=[];errors=[]
    def encode(frame):
        for kind in ('encoder','existing_projector'):ledger.dispatch(kind,tensor(kind),lambda:None,(),{})
        return np.zeros(192,np.float32)
    m.encode_frames(protocol,folder,encode,ledger,events.append,rows,errors)
    assert not errors
    journal=folder/'encoding.events.jsonl'
    journal.write_text(''.join(m.io.canonical(e)+'\n' for e in events))
    receipt=dict(schema=m.NAME+'.derived_receipt',status='all_17_prefix_latents_saved',errors=[],first_error=None,blocked=[],journal_errors=[],
        model_unchanged=True,model_hash_before='FAKE-TEST-HASH',model_hash_after='FAKE-TEST-HASH',source_verified=True,
        physical_report_sha256=m.PHYSICAL_SHA,planner_report_sha256=m.PLANNER_SHA,prepare_sha256='fake-test-prepare',
        goal_q_sha256=protocol['goal_q_sha256'],model_bindings=protocol['model_bindings'],counts=ledger.counts(),
        new_predict_calls=0,new_AD_calls=0,new_physical_steps=0,training=False,P1_admission=False,P3_admission=False,efficacy_claim=False,
        journal_sha256=m.sha(journal),rows=rows,entries=ledger.entries)
    m.write(folder/'receipt.json',receipt);m.write(folder/'execution_binding.json',dict(prepared='fake-test-only',prepare_sha256='fake-test-prepare'))
    monkeypatch.setattr(m,'verify',lambda *a:protocol)
    out=tmp_path/'numbers';r=m.consume(folder,m.sha(folder/'receipt.json'),out)
    assert len(r['rows'])==17 and r['status']=='saved_prefix_numerics_verified' and r['new_neural_calls']==0
    assert len(r['trajectories'])==16 and r['inherited_terminal_encoder_samples']==16
    for t in r['trajectories']:
        assert [x['primitive_step'] for x in t['checkpoints']]==([5,10] if t['horizon']==2 else [5,10,15])
        assert t['checkpoints'][-1]['source']=='inherited_original_terminal_encoding'
    # Same fake receipt cannot hide altered native arrays or terminal substitution.
    path=Path(rows[0]['latent_path']);np.save(path,np.ones(192,np.float32),allow_pickle=False)
    with pytest.raises(ValueError):m.consume(folder,m.sha(folder/'receipt.json'),tmp_path/'tampered')


def test_main_setup_failure_receipt_is_durable_and_not_model_verified(tmp_path,monkeypatch):
    p=dict(tiny(),goal_q_sha256='fake-goalQ',model_bindings={})
    monkeypatch.setattr(m,'verify',lambda *a:p)
    def no_cuda(*a):raise RuntimeError('FAKE preflight first error, no CUDA ran')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'experiments.checkpoint_compat',SimpleNamespace(load_compatible_pretrained=lambda *a,**k:None))
    monkeypatch.setitem(sys.modules,'experiments.gpu_preflight',SimpleNamespace(require_cuda_allocation=no_cuda))
    monkeypatch.setitem(sys.modules,'experiments.spectral_reachability.preflight_pusht_d2p_p0',SimpleNamespace())
    prepared=tmp_path/'prepare';prepared.mkdir();out=tmp_path/'run'
    with pytest.raises(RuntimeError,match='incomplete'):m.execute_cuda(prepared,'fake',out)
    receipt=m.read(out/'receipt.json')
    assert receipt['first_error']['message'].startswith('FAKE preflight first error')
    assert receipt['model_unchanged'] is None and receipt['model_hash_before'] is None
    assert len(receipt['rows'])==17 and all(x['status']=='blocked_not_attempted' for x in receipt['rows'])
    assert receipt['counts']['encoder']['attempted']==0
    # Launch claims are separate new files, not edits to a sealed prepare.
    assert not list(prepared.iterdir()) and prepared.with_name(prepared.name+'.launch.json').is_file()
    with pytest.raises(FileExistsError):m.execute_cuda(prepared,'fake',tmp_path/'retry')


def test_reserve_refuses_before_any_encode(tmp_path,monkeypatch):
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=200*1024*1024))
    p=tiny();ledger=m.Ledger(p['frames'],lambda e:None);rows=[];errors=[];called=[]
    m.encode_frames(p,tmp_path,lambda e:called.append(e),ledger,lambda e:None,rows,errors)
    assert not called and errors and len(rows)==17 and rows[1]['status']=='blocked_not_attempted'
