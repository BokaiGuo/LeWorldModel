"""Saved metadata + synthetic NumPy module interfaces only; never a neural run."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_midpoint_teacher_suffix_20261028_v1 as m


def paid_context():
    return dict(m.mid.META, latent_history=np.arange(576, dtype=np.float32).reshape(3,192).tolist(),
                action_history=np.zeros((2,10), np.float32).tolist())


def norm(): return dict(mean=[0.,0.], std=[1.,1.])


def test_only_last_latent_replaced_not_history_or_actions():
    old = paid_context()
    raw = np.full((10,2), .125, np.float32).tolist()
    teacher, buffers = m.intervention(old, [7.]*192, raw, norm())
    assert teacher['latent_history'][:2] == old['latent_history'][:2]
    assert teacher['latent_history'][-1] == [7.]*192 and teacher['action_history'] == old['action_history']
    assert old['latent_history'][-1] != [7.]*192
    assert np.array_equal(buffers['outgoing_tokens'][:2], np.asarray(old['action_history'], np.float32))
    assert np.array_equal(buffers['outgoing_tokens'][-1], np.full(10, .125, np.float32))


@pytest.mark.parametrize('damage', ['null_z5','wrong_shape','nan','bool','wrong_context','zero_std','bad_raw','not_float32'])
def test_intervention_missing_values_rejected(damage):
    c, z, raw, normalization = paid_context(), [7.]*192, [[.125,.125]]*10, norm()
    if damage == 'null_z5': z = None
    if damage == 'wrong_shape': z = z[:-1]
    if damage == 'nan': z[0] = float('nan')
    if damage == 'bool': z[0] = True
    if damage == 'wrong_context': c['context_length'] = 1
    if damage == 'zero_std': normalization['std'][0] = 0.
    if damage == 'bad_raw': raw[0] = [2.,0.]
    if damage == 'not_float32': z[0] = .1
    with pytest.raises(ValueError): m.intervention(c,z,raw,normalization)


@pytest.fixture(scope='module')
def actual(): return m.collect()


def test_actual8_source_contexts_and_saved_z5(actual):
    assert len(actual['jobs']) == 8 and len(actual['bindings']) == 222
    assert [(j['root_slot'],j['arm']) for j in actual['jobs']] == [(i,a) for i in range(4) for a in m.ARMS]
    assert actual['unique_midpoint_images'] == 7
    for job in actual['jobs']:
        z = m.mid.load_array(job['actual_z5_folder'], job['actual_z5_receipt'])
        assert job['teacher_mid_context']['latent_history'][-1] == z.tolist()
        assert job['teacher_mid_context']['latent_history'][:2] == job['paid_mid_context']['latent_history'][:2]
        assert job['teacher_mid_context']['action_history'] == job['paid_mid_context']['action_history']
        assert job['raw_suffix5'] == job['raw10'][5:]
    assert actual['jobs'][0]['actual_z5_receipt']['sha256'] == actual['jobs'][1]['actual_z5_receipt']['sha256']


def test_config_real_native_api_abi(actual):
    config = m.read(Path(actual['policy'])/'config.json')
    abi = m.native_abi(config)
    assert abi['outgoing_tokens'] == [1,3,10] and abi['predict_output'] == [1,3,192] and abi['pred_proj_flat'] == [3,192]
    config['predictor']['num_frames'] = 1
    with pytest.raises(ValueError): m.native_abi(config)


def tensor(value):
    a = np.asarray(value, np.float32)
    return SimpleNamespace(shape=a.shape, device=SimpleNamespace(type='cuda'), value=a)


def ram_protocol():
    jobs = []
    for i in range(8):
        paid, raw = paid_context(), [[.125,-.25]]*10
        teacher, buffers = m.intervention(paid, [1.]*192, raw, norm())
        job = dict(job_id='job_'+str(i), root_slot=i//2, root_sha256='RAM_root_'+str(i//2), arm=m.ARMS[i%2],
                   paid_mid_context=paid, paid_mid_context_sha256=m.digest(paid), teacher_mid_context=teacher,
                   raw10=raw, action_normalization=norm(), goal=[0.]*192, projection=np.eye(192).tolist(),
                   actual_terminal_z=[2.]*192, original_terminal_latent=[3.]*192,
                   original_terminal_prediction=dict(Q_cost=9., physical_teacher_Q=4.),
                   buffer_digests={k:m.digest(v.tolist()) for k,v in buffers.items()})
        job['job_binding_sha256'] = m.digest(job)
        jobs.append(job)
    return dict(jobs=jobs, contract=m.contract(), model_bindings={'RAM':'not_weights'}, expected_model_state_sha256='RAM', inherited_resources='RAM',
                inherited_midpoint_encoding_counts='RAM')


def fake_adapter(ledger):
    def adapter(job, buffers):
        lat, outgoing = tensor(buffers['teacher_latents'][None]), tensor(buffers['outgoing_tokens'][None])
        ae = ledger.dispatch('action_encoder', lambda x: tensor(np.full((1,3,192), .5, np.float32)), (outgoing,), {})
        def predict(l, a):
            predicted = ledger.dispatch('predictor', lambda x,y: tensor(np.ones((1,3,192), np.float32)), (l,a), {})
            proj = ledger.dispatch('pred_proj', lambda x: tensor(x.value.copy()), (tensor(predicted.value.reshape(3,192)),), {})
            return tensor(proj.value.reshape(1,3,192))
        return ledger.dispatch('predict', predict, (lat,ae), {}).value
    return adapter


def ram_run(folder, protocol=None):
    p = ram_protocol() if protocol is None else protocol
    rows, errors = [], []
    with (folder/'events.jsonl').open('x') as stream:
        journal = m.mid.Journal(stream)
        def persist(name,t): return m.mid.save_array(folder,name,t.value)
        ledger = m.Ledger(p['jobs'], journal, persist, precision=m.mid.precision_contract)
        m.predict_jobs(p,folder,fake_adapter(ledger),ledger,journal,rows,errors)
        assert not errors
        report = dict(schema=m.NAME+'.report', status='all8_suffix_predictions_saved', contract=m.contract(), errors=[], blocked=[],
                      journal_errors=[], first_error=None, host=dict(device='cuda:0'), model_hash_before='RAM', model_hash_after='RAM',
                      model_unchanged=True, source_verified=True, model_bindings=p['model_bindings'],
                      precision_before=m.mid.precision_contract(), precision_after=m.mid.precision_contract(),
                      new_encoder_calls=0, new_projector_calls=0, new_AD_calls=0, new_physics_steps=0,
                      P1_admission=False, P2_admission=False, scientific_admission=False, rows=rows, entries=ledger.entries,
                      counts=ledger.counts(), total_wall_seconds=1., journal_sha256=m.sha(folder/'events.jsonl'),
                      journal_tail_sha256=journal.previous)
    return p,report


def test_full_ram_call_tree_arrays_and_comparisons(tmp_path):
    p,r = ram_run(tmp_path)
    z = m.validate_run(tmp_path,r,p)
    rows,pairs = m.comparisons(p,z)
    assert len(rows) == len(z) == 8 and len(pairs) == 4
    assert rows[0]['teacher_forced_suffix']['predicted_Q'] == 1.
    assert rows[0]['teacher_forced_suffix']['teacher_Q'] == 4.
    assert all(c['attempted'] == c['entered'] == c['returned'] == c['saved'] == 8 for c in r['counts'].values())
    assert len(r['entries']) == 32


@pytest.mark.parametrize('damage', ['missing_row','duplicate_root','order','entry_count','CPU','unknown_hash','model',
                                    'encoder','AD','precision','counts','array_hash','journal','history','z5','goal'])
def test_ram_strict_mutations_failclosed(tmp_path,damage):
    p,r = ram_run(tmp_path)
    if damage == 'missing_row': r['rows'].pop()
    if damage == 'duplicate_root': p['jobs'][-1]['root_slot'] = 0
    if damage == 'order': r['rows'].reverse()
    if damage == 'entry_count': r['entries'].pop()
    if damage == 'CPU': r['host']['device'] = 'cpu'
    if damage == 'unknown_hash': r['model_hash_before'] = r['model_hash_after'] = None
    if damage == 'model': r['model_hash_after'] = 'changed'
    if damage == 'encoder': r['new_encoder_calls'] = 1
    if damage == 'AD': r['new_AD_calls'] = 1
    if damage == 'precision': r['precision_after']['sdpa_flash'] = True
    if damage == 'counts': r['counts']['predict']['entered'] = 7
    if damage == 'array_hash': r['entries'][0]['arrays']['input_0']['sha256'] = 'bad'
    if damage == 'journal': r['journal_sha256'] = 'bad'
    if damage == 'history': p['jobs'][0]['teacher_mid_context']['latent_history'][0][0] += 1
    if damage == 'z5': p['jobs'][0]['teacher_mid_context']['latent_history'][-1][0] += 1
    if damage == 'goal': p['jobs'][0]['goal'][0] += 1
    with pytest.raises(ValueError): m.validate_run(tmp_path,r,p)


@pytest.mark.parametrize('damage',['CPU','shape','predictor_without_API','wrong_job'])
def test_guard_before_native_callback(damage):
    p = ram_protocol(); calls=[]
    ledger = m.Ledger(p['jobs'],lambda e:None,lambda *args:None)
    ledger.active = p['jobs'][0]['job_id']
    x,kind = tensor(np.zeros((1,3,10))), 'action_encoder'
    if damage == 'CPU': x.device.type='cpu'
    if damage == 'shape': x.shape=(1,1,10)
    if damage == 'predictor_without_API': kind='predictor'
    if damage == 'wrong_job': ledger.active=p['jobs'][1]['job_id']
    with pytest.raises(ValueError): ledger.dispatch(kind,lambda *args:calls.append(1),(x,),{})
    assert not calls and not ledger.entries and ledger.blocked


@pytest.mark.parametrize('kind',['encoder','projector','encode','rollout','AD.grad','AD.vjp'])
def test_forbidden_entries(kind):
    p = ram_protocol(); ledger=m.Ledger(p['jobs'],lambda e:None,lambda *args:None)
    with pytest.raises(RuntimeError): ledger.forbid(kind)()
    assert ledger.blocked


def test_failed_nested_call_keeps_original_error_and8_denominator(tmp_path):
    p=ram_protocol(); rows=[]; errors=[]
    ledger=m.Ledger(p['jobs'],lambda e:None,lambda name,t:m.mid.save_array(tmp_path,name,t.value),precision=m.mid.precision_contract)
    def fail(job,buffers): raise RuntimeError('original RAM error')
    m.predict_jobs(p,tmp_path,fail,ledger,lambda e:None,rows,errors)
    assert len(rows)==8 and rows[0]['status']=='failed'
    assert all(r['status']=='blocked_not_attempted' for r in rows[1:])
    assert errors[0]['message']=='original RAM error'


def test_attempt_persistence_failure_no_native_entry():
    p=ram_protocol(); calls=[]
    def fail(event): raise OSError('journal unavailable')
    ledger=m.Ledger(p['jobs'],fail,lambda *args:None)
    ledger.active=p['jobs'][0]['job_id']
    with pytest.raises(OSError): ledger.dispatch('action_encoder',lambda x:calls.append(1),(tensor(np.zeros((1,3,10))),),{})
    assert not calls and not ledger.entries[0]['native_entered'] and ledger.counts()['action_encoder']['attempted']==1


def test_nested_failure_nonrefundable_and_no_retry(tmp_path):
    p=ram_protocol(); ledger=m.Ledger(p['jobs'],lambda e:None,
        lambda name,t:m.mid.save_array(tmp_path,name,t.value),precision=m.mid.precision_contract)
    ledger.active=p['jobs'][0]['job_id']
    ae=ledger.dispatch('action_encoder',lambda x:tensor(np.zeros((1,3,192))), (tensor(np.zeros((1,3,10))),),{})
    def fail_predictor(x,y): raise RuntimeError('original nested failure')
    def predict(x,y): return ledger.dispatch('predictor',fail_predictor,(x,y),{})
    with pytest.raises(RuntimeError,match='original nested failure'):
        ledger.dispatch('predict',predict,(tensor(np.zeros((1,3,192))),ae),{})
    assert not ledger.stack and ledger.counts()['predict']['entered']==1 and ledger.counts()['predictor']['entered']==1
    assert ledger.counts()['predict']['returned']==ledger.counts()['predictor']['returned']==0
    with pytest.raises(ValueError): ledger.dispatch('pred_proj',lambda x:x,(tensor(np.zeros((3,192))),),{})


def test_nested_return_journal_failure_preserves_native_error():
    p=ram_protocol()
    def journal(event):
        if event['event']=='return_or_error': raise OSError('secondary journal')
    ledger=m.Ledger(p['jobs'],journal,lambda name,x:{'shape':list(x.shape)},precision=m.mid.precision_contract)
    ledger.active=p['jobs'][0]['job_id']
    def fail(x): raise RuntimeError('primary native error')
    with pytest.raises(RuntimeError,match='primary native error'):
        ledger.dispatch('action_encoder',fail,(tensor(np.zeros((1,3,10))),),{})
    assert ledger.journal_errors and ledger.entries[0]['error']['message']=='primary native error'


def test_public_prepare_verify_ram_only(tmp_path,monkeypatch):
    p=ram_protocol(); p['bindings']={}; p['model_bindings']={}
    monkeypatch.setattr(m,'collect',lambda:deepcopy(p))
    monkeypatch.setattr(m,'source_files',lambda:[Path(m.__file__).resolve()])
    output=tmp_path/'prepared'; receipt=m.prepare(output)
    assert len(m.verify(output,receipt['prepare_sha256'])['jobs'])==8
    with pytest.raises(ValueError): m.verify(output,'0'*64)
    with pytest.raises(ValueError): m.prepare(output)


def test_public_consume_ram_only(tmp_path,monkeypatch):
    folder=tmp_path/'run'; folder.mkdir()
    p,r=ram_run(folder)
    monkeypatch.setattr(m,'verify',lambda *args:p)
    r.update(prepare_sha256='RAM',source_midpoint_report_sha256=m.INPUTS['run'][1])
    m.write(folder/'report.json',r)
    m.write(folder/'execution_binding.json',dict(prepared='RAM',prepare_sha256='RAM',source_midpoint_report_sha256=m.INPUTS['run'][1]))
    output=tmp_path/'consume.json'; result=m.consume(folder,m.sha(folder/'report.json'),output)
    assert m.read(output)['scientific_admission'] is False and result['sha256']==m.sha(output)
    assert len(m.read(output)['rows'])==8
    with pytest.raises(ValueError): m.consume(folder,'0'*64,tmp_path/'none.json')
    with pytest.raises(FileExistsError): m.consume(folder,m.sha(folder/'report.json'),output)


def test_cold_saved_no_torch_and_explicit_run_authority():
    code='''
import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','gymnasium','pymunk','stable_worldmodel','requests'}:
   raise AssertionError('forbidden import '+fullname)
sys.meta_path.insert(0,Guard())
from experiments.spectral_reachability import agent_p2_midpoint_teacher_suffix_20261028_v1 as m
assert len(m.collect()['jobs'])==8
assert 'torch' not in sys.modules
'''
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,env=env,capture_output=True,check=True,timeout=45)
    r=subprocess.run([sys.executable,'-B','-m','experiments.spectral_reachability.'+m.NAME,'run'],cwd=m.ROOT,
                     env=env,capture_output=True,text=True,timeout=15)
    assert r.returncode!=0 and 'MAIN HOST CUDA authorization' in r.stderr
