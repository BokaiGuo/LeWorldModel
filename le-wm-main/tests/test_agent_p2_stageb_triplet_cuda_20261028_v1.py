"""RAM native-ABI stand-ins + read-only saved inputs; never neural or physics."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from experiments.spectral_reachability import agent_p2_stageb_triplet_cuda_20261028_v1 as m


class T:
    def __init__(self,a):
        self.value = np.asarray(a,np.float32); self.shape = self.value.shape
        self.device = SimpleNamespace(type='cuda'); self.dtype = 'torch.float32'


def protocol(folder,stage):
    stats = dict(mean=[.5,.5,.5],std=[.25,.25,.25]); jobs = []
    folder.mkdir(exist_ok=True)
    for i in range(8):
        frames = []
        for n in range(3):
            rgb = np.full((224,224,3),i*3+n,np.uint8); path = folder/f'job{i}.frame{n}.png'
            Image.fromarray(rgb).save(path)
            frames.append(dict(path=str(path),file_sha256=m.sha(path),pixel_sha256=m.inv.pixel_sha(rgb)))
        teacher = dict(m.mid.META,latent_history=np.zeros((3,192),np.float32).tolist(),
                       action_history=[np.zeros(10,np.float32).tolist(),[.125,-.25]*5])
        j = dict(job_id=f'RAM_job{i}',root_slot=i//2,arm=m.ARMS[i%2],root_sha256=f'RAM_root{i//2}',
            frames=frames,paid_mid_context=teacher,teacher_mid_context=teacher,raw10=[[.125,-.25]]*10,
            action_normalization=dict(mean=[0.,0.],std=[1.,1.]),goal=[0.]*192,projection=np.eye(192).tolist(),
            actual_terminal_z=[2.]*192,metric_id='RAM_original_Q')
        if stage == 'compare':
            receipt = m.mid.save_array(folder,f'job{i}.cached_full3',np.full((1,3,192),i/8,np.float32))
            j['cache_full3'] = dict(folder=str(folder),receipt=receipt)
        j['job_binding_sha256'] = m.digest(j); jobs.append(j)
    return dict(schema=m.NAME+'.protocol',contract=m.contract(stage),jobs=jobs,image_statistics=stats,
        encoder_abi=dict(tokens=257,encoder_dim=192),expected_model_state_sha256='RAM_frozen',
        model_bindings={'RAM':'not_weights'},inherited_resources={'RAM':'not_science'})


def fake_adapter(ledger,p,*,manual_delta=0.,fail_on=None):
    def adapter(q,b):
        rid = q['run_id']
        if rid == fail_on: raise RuntimeError('RAM injected adapter failure')
        def dispatch(k,fn,*args,**kw): return ledger.dispatch(k,fn,args,kw)
        if q['route'] == 'cache':
            def encode(info):
                def encoder(x,**kwargs):
                    assert kwargs == dict(interpolate_pos_encoding=True)
                    output = np.zeros((3,p['encoder_abi']['tokens'],192),np.float32)
                    output[:,0] = x.value[:,0,0,0,None]
                    return SimpleNamespace(last_hidden_state=T(output))
                e = dispatch('encoder',encoder,T(info['pixels'].value[0]),interpolate_pos_encoding=True)
                z = dispatch('projector',lambda x:T(x.value.copy()),T(e.last_hidden_state.value[:,0]))
                info['emb'] = T(z.value[None]); return info
            out = dispatch('encode',encode,{'pixels':T(b['normalized_pixels'])})
            return dict(full3=out['emb'].value)
        def prediction(lat,ae):
            delta = manual_delta if q['route'] == 'manual' else 0.
            pred = dispatch('predictor',lambda x,y:T(x.value+y.value+delta),lat,ae)
            proj = dispatch('pred_proj',lambda x:T(x.value.copy()),T(pred.value[0]))
            return T(proj.value[None])
        def predict(lat,actions):
            ae = dispatch('action_encoder',lambda a:T(np.repeat(a.value.mean(-1,keepdims=True),192,axis=-1)),actions)
            return dispatch('predict',prediction,lat,ae)
        if q['route'] == 'public_rollout':
            def rollout(info,actions,*,history_size):
                assert history_size == 3 and actions.value.shape == (1,1,3,10)
                pred = predict(T(info['emb'].value[:,0]),T(actions.value[:,0]))
                info['action'] = actions
                info['predicted_emb'] = T(np.concatenate((info['emb'].value,pred.value[:,None,-1:]),axis=2))
                return info
            out = dispatch('rollout',rollout,{'pixels':T(b['normalized_pixels'][:,None]),'emb':T(b['full3'][:,None])},
                           T(b['outgoing_tokens'][:,None]),history_size=3)
            return dict(public_predicted_emb=out['predicted_emb'].value)
        out = predict(T(b['full3']),T(b['outgoing_tokens']))
        return dict(manual_predictions=out.value)
    return adapter


def ram_run(tmp_path,stage,**adapter_kw):
    p = protocol(tmp_path/'inputs',stage); output = tmp_path/'output'; output.mkdir()
    m.write(output/'execution_binding.json',dict(prepared='RAM',prepare_sha256='a'*64,authority=m.AUTH,**m.FLAGS))
    rows,errors = [],[]
    with (output/'events.jsonl').open('x') as f:
        journal = m.mid.Journal(f)
        ledger = m.Ledger(p,journal,lambda name,t:m.mid.save_array(output,name,t.value),precision=m.mid.precision_contract)
        m.process(p,output,fake_adapter(ledger,p,**adapter_kw),ledger,journal,rows,errors)
    r = dict(schema=m.NAME+'.report',stage=stage,status='completed' if not errors else 'partial_or_failed',contract=m.contract(stage),
        rows=rows,entries=ledger.entries,counts=ledger.counts(),errors=errors,blocked=ledger.blocked,journal_errors=ledger.journal_errors,
        host=dict(device='cuda:0'),model_hash_before='RAM_frozen',model_hash_after='RAM_frozen',source_verified=True,
        model_bindings=p['model_bindings'],precision_before=m.mid.precision_contract(),precision_after=m.mid.precision_contract(),
        root_denominator=4,arm_denominator=8,route_denominator=len(m.plans(p)),total_wall_seconds=1.,prepare_sha256='a'*64,
        journal_sha256=m.sha(output/'events.jsonl'),journal_tail_sha256=journal.previous,new_AD_calls=0,new_physics_steps=0,
        artifact_sha256={f.name:m.sha(f) for f in output.iterdir()},**m.FLAGS)
    m.write(output/'report.json',r)
    return p,r,output


@pytest.mark.parametrize('stage',['cache','compare'])
def test_real_callable_ram_adapters_full_saved_closure(tmp_path,stage):
    p,r,output = ram_run(tmp_path,stage)
    values = m.validate_run(output,r,p); rows = m.summary(p,values)
    assert len(rows) == 8 and len(r['rows']) == (8 if stage == 'cache' else 16)
    assert not r['errors'] and m.contract(stage)['planned_returns'] is None
    if stage == 'cache':
        assert r['counts']['encoder']['attempted'] == 8 and r['counts']['encoder']['input_axis0_samples'] == 24
        assert r['counts']['projector']['input_temporal_rows'] == 24 and r['counts']['predict']['attempted'] == 0
    else:
        assert r['counts']['predict']['attempted'] == 16 and r['counts']['rollout']['attempted'] == 8
        assert r['counts']['pred_proj']['input_axis0_samples'] == 48
        assert r['counts']['encoder']['attempted'] == 0
        assert all(x['all_native_buffers_exact_equal'] and x['first_unequal_buffer'] is None for x in rows)


def test_numeric_route_discrepancy_retained_not_filtered_or_failclosed(tmp_path):
    p,r,output = ram_run(tmp_path,'compare',manual_delta=.125)
    rows = m.summary(p,m.validate_run(output,r,p))
    assert len(rows) == 8 and all(not x['all_native_buffers_exact_equal'] for x in rows)
    assert all(x['first_unequal_buffer'] == 'predictor.output_output' for x in rows)
    assert r['status'] == 'completed' and r['errors'] == []


@pytest.mark.parametrize('stage',['cache','compare'])
def test_failed_first_row_keeps_full_denominator_and_no_hidden_returns(tmp_path,stage):
    p = protocol(tmp_path/'inputs',stage)
    fail = m.plans(p)[0]['run_id']
    # A second independent temp fixture, never saved research assets.
    sub = tmp_path/'run'; sub.mkdir()
    p,r,out = ram_run(sub,stage,fail_on=fail)
    assert len(r['rows']) == (8 if stage == 'cache' else 16)
    assert r['rows'][0]['status'] == 'failed' and all(x['status'] == 'blocked_not_attempted' for x in r['rows'][1:])
    assert all(x['returned'] == 0 for x in r['counts'].values())
    with pytest.raises(ValueError): m.validate_run(out,r,p)


@pytest.mark.parametrize('stage',['cache','compare'])
@pytest.mark.parametrize('damage',['denominator','counter','flags','source','tokens','precision','model','row','journal','extra','native','goal'])
def test_mutation_rejected_without_neural_execution(tmp_path,stage,damage):
    p,r,out = ram_run(tmp_path,stage)
    if damage == 'denominator': r['arm_denominator'] = 6
    if damage == 'counter': r['counts']['predict']['returned'] += 1
    if damage == 'flags': r['P2_admission'] = True
    if damage == 'source': r['source_verified'] = False
    if damage == 'tokens': p['encoder_abi']['tokens'] = 197 if stage == 'cache' else 257; p['jobs'][0]['raw10'][5][0] = .5
    if damage == 'precision': r['precision_before']['matmul_tf32'] = True
    if damage == 'model': r['model_hash_after'] = 'changed'
    if damage == 'row': r['rows'].pop()
    if damage == 'journal': r['journal_tail_sha256'] = 'bad'
    if damage == 'extra': (out/'undeclared.txt').write_text('RAM')
    if damage == 'native':
        receipt = r['entries'][0]['arrays'][next(iter(r['entries'][0]['arrays']))]
        (out/receipt['file']).write_bytes(b'corrupt RAM array')
    if damage == 'goal': p['jobs'][0]['goal'][0] = 1.
    with pytest.raises((ValueError,KeyError)): m.validate_run(out,r,p)


def test_zero_encoder_compare_missing_cache_is_explicit_block(tmp_path,monkeypatch):
    p = protocol(tmp_path/'inputs','cache')
    monkeypatch.setattr(m,'actual_triplets',lambda:deepcopy(p))
    with pytest.raises(ValueError,match='full3 triplet cache prerequisite'): m.build('compare')


def test_missing_or_mismatching_cache_array_never_defaulted(tmp_path):
    p = protocol(tmp_path/'inputs','compare'); j = p['jobs'][0]
    j['cache_full3']['receipt']['shape'] = [1,1,192]
    with pytest.raises(ValueError): m.inputs_for(j,p['image_statistics'],'compare')


def test_manual_raw_normalization_exact_not_changed(tmp_path):
    p = protocol(tmp_path/'inputs','compare'); j = p['jobs'][0]
    b = m.inputs_for(j,p['image_statistics'],'compare')
    assert b['outgoing_tokens'].shape == (1,3,10)
    assert np.array_equal(b['outgoing_tokens'][0,0],np.zeros(10,np.float32))
    assert np.array_equal(b['outgoing_tokens'][0,1:],np.array([[.125,-.25]*5]*2,np.float32))
    j['action_normalization']['std'][0] = 0.
    with pytest.raises(ValueError): m.inputs_for(j,p['image_statistics'],'compare')


def test_stale_prefix_token_rejected_not_silently_regenerated(tmp_path):
    p = protocol(tmp_path/'inputs','compare'); j = p['jobs'][0]
    j['paid_mid_context']['action_history'][1] = [0.]*10
    with pytest.raises(ValueError,match='paid A3'): m.inputs_for(j,p['image_statistics'],'compare')


@pytest.mark.parametrize('op',['encoder','projector','predict'])
def test_wrong_operation_order_blocked_before_native(tmp_path,op):
    p = protocol(tmp_path/'inputs','compare'); calls = []; events = []
    ledger = m.Ledger(p,events.append,lambda *args:None); ledger.active = m.plans(p)[0]['run_id']
    with pytest.raises(ValueError): ledger.dispatch(op,lambda:calls.append(1),(),{})
    assert calls == [] and ledger.entries == [] and ledger.blocked


def test_runtime_requires_MAIN_authority_before_verify_import_or_output(tmp_path):
    with pytest.raises(ValueError,match='MAIN HOST CUDA'): m.run_cuda('missing','bad',tmp_path/'out','no')
    assert not (tmp_path/'out').exists()


@pytest.fixture(scope='module')
def saved_actual(): return m.actual_triplets()


def test_actual_warmup_strict_reconsume_and_all8_genuine_frames(saved_actual):
    p = saved_actual
    assert len(p['jobs']) == 8 and p['cached_actual_triplet_full3_receipt_available_in_bound_prior_assets'] is False
    assert [(j['root_slot'],j['arm']) for j in p['jobs']] == [(i,a) for i in range(4) for a in m.ARMS]
    for j in p['jobs']:
        assert j['frame_roles'] == ['warmup10','warmup15','actualstep5']
        assert len({f['path'] for f in j['frames']}) == len({f['observation_id'] for f in j['frames']}) == 3
        assert all(f['actual_observation'] is True for f in j['frames'])
    assert p['encoder_abi']['tokens'] == 257
    assert p['bindings'][str(m.RECON_STRICT)] == m.RECON_STRICT_SHA


def test_actual_saved_cache_prepare_verify_no_cuda(tmp_path,saved_actual,monkeypatch):
    # Reuse the fully validated actual input snapshot; build/verify still hash it.
    monkeypatch.setattr(m,'actual_triplets',lambda:deepcopy(saved_actual))
    p = m.build('cache'); path = tmp_path/'prepare.json'; m.write(path,p)
    assert m.verify(path,m.sha(path)) == p
    assert p['contract']['planned_calls']['encoder'] == 8 and p['contract']['encoder_samples'] == 24
    with pytest.raises(ValueError): m.verify(path,'0'*64)


def test_compare_prepare_relative_paths_canonicalized(tmp_path,monkeypatch):
    base = protocol(tmp_path/'inputs','cache')
    base.update(bindings={},sources_sha256={},model_bindings={})
    cp = deepcopy(base)
    monkeypatch.setattr(m,'actual_triplets',lambda:deepcopy(base))
    monkeypatch.setattr(m,'verify',lambda *args:cp)
    cache = tmp_path/'cache'; cache.mkdir()
    prep = tmp_path/'cache_prepare.json'; m.write(prep,{})
    m.write(cache/'execution_binding.json',dict(prepared=str(prep),prepare_sha256=m.sha(prep)))
    (cache/'events.jsonl').write_text('RAM fixture')
    report = cache/'report.json'
    m.write(report,dict(rows=[dict(arrays={'full3':{'file':'RAM.npy','sha256':'RAM','shape':[1,3,192],'dtype':'float32'}}) for _ in range(8)],
                       counts={},artifact_sha256={}))
    receipt = dict(stage='cache',status='all8_saved_verified_not_admitted')
    strict = tmp_path/'strict.json'; m.write(strict,receipt)
    monkeypatch.setattr(m,'consume',lambda *args:receipt); monkeypatch.chdir(tmp_path)
    a = m.build('compare','cache/report.json',m.sha(report),'strict.json',m.sha(strict))
    b = m.build('compare',report,m.sha(report),strict,m.sha(strict))
    assert a == b
    assert all(Path(path).is_absolute() for path in a['bindings'])


def test_entered_native_exception_is_charged_and_no_retry(tmp_path):
    p = protocol(tmp_path/'inputs','cache'); events = []
    ledger = m.Ledger(p,events.append,lambda name,t:dict(shape=list(t.shape)),precision=m.mid.precision_contract)
    ledger.active = m.plans(p)[0]['run_id']
    def broken(info): raise RuntimeError('RAM native failure')
    args = ({'pixels':T(np.zeros((1,3,3,224,224),np.float32))},)
    with pytest.raises(RuntimeError): ledger.dispatch('encode',broken,args,{})
    c = ledger.counts()['encode']
    assert c['attempted'] == c['entered'] == 1 and c['returned'] == c['saved'] == 0
    assert c['input_temporal_rows'] == 3
    assert events[-1]['event'] == 'return_or_error'
    with pytest.raises(ValueError): ledger.dispatch('encode',broken,args,{})
    assert len(ledger.entries) == 1


def test_cold_import_and_saved_contract_no_new_runtime_imports():
    code = '''
import sys
class Guard:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','gym','gymnasium','pymunk','stable_worldmodel','stable_pretraining'}:
   raise AssertionError('forbidden runtime import '+fullname)
sys.meta_path.insert(0,Guard())
from experiments.spectral_reachability import agent_p2_stageb_triplet_cuda_20261028_v1 as m
assert m.contract('compare')['planned_calls']['predict'] == 16
assert m.contract('compare')['planned_calls']['encoder'] == 0
assert m.contract('cache')['encoder_samples'] == 24
'''
    proc = subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True,timeout=60,
                          env={**os.environ,'OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
    assert proc.returncode == 0,proc.stdout+proc.stderr
