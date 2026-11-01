"""RAM callback topology/array mutations, never torch/model/physics execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_teacher_library40_api_equivalence_20261028_v1 as m


class T:
    def __init__(self,x):
        self.value = np.asarray(x,np.float32)
        self.device = SimpleNamespace(type='cuda'); self.dtype = 'torch.float32'


def protocol():
    jobs = []
    for i in range(40):
        task,rem = divmod(i,10); role,word = divmod(rem,5)
        j = dict(index=i,job_id=f'slot{i:02d}',root_slot=task//2,condition=('off','on')[task%2],role=('Z','S')[role],
            library_index=word,library_label=m.teacher.LIBRARY[word],frames=[None]*3,actual192=None if i == 12 else [0.]*192,
            goal192=[0.]*192,actual_Q=None if i == 12 else 0.,goal_binding=dict(task=task),anchor_id=str(i//5),
            raw_suffix_bits=m.history.bits(np.zeros((5,2),np.float32)+word),missing_reason='missing12' if i == 12 else None,
            compiler_folder='RAM_ONLY',old_prediction_receipt=dict(index=i))
        j['job_binding_sha256'] = m.digest(j); jobs.append(j)
    return dict(jobs=jobs,contract=m.contract(),image_statistics={},expected_model_state_sha256='RAM_NOT_WEIGHTS',model_bindings={'RAM':'not_model'})


RGB = np.zeros((3,224,224,3),np.uint8)
PIXELS = np.zeros((1,3,3,224,224),np.float32)


def source(j,stats):
    return dict(rgb=RGB,normalized_pixels=PIXELS,full3=np.full((1,3,192),j['index']/40,np.float32),
                outgoing_tokens=np.full((1,3,10),j['library_index']/8,np.float32),goal192=np.zeros(192,np.float32))


def fake_adapter(ledger,delta=0.,fail=None):
    def adapter(q,b):
        if q['run_id'] == fail: raise RuntimeError('RAM injected failure, not execution')
        def dispatch(k,fn,*a,**kw): return ledger.dispatch(k,fn,a,kw)
        def predict(lat,act):
            ae = dispatch('action_encoder',lambda x:T(np.repeat(x.value.mean(-1,keepdims=True),192,axis=-1)),act)
            def api(x,y):
                pr = dispatch('predictor',lambda a,c:T(a.value+c.value+(delta if q['route'] == 'cached_suffix' else 0.)),x,y)
                pp = dispatch('pred_proj',lambda a:T(a.value),T(pr.value[0]))
                return T(pp.value[None])
            return dispatch('predict',api,lat,ae)
        if q['route'] == 'public_rollout':
            def rollout(info,actions,*,history_size):
                assert history_size == 3
                pred = predict(T(info['emb'].value[:,0]),T(actions.value[:,0]))
                info['action'] = actions
                info['predicted_emb'] = T(np.concatenate((info['emb'].value,pred.value[:,None,-1:]),axis=2))
                return info
            out = dispatch('rollout',rollout,dict(pixels=T(b['normalized_pixels'][:,None]),emb=T(b['full3'][:,None])),T(b['outgoing_tokens'][:,None]),history_size=3)
            return dict(public_predicted_emb=out['predicted_emb'].value)
        return dict(cached_predictions=predict(T(b['full3']),T(b['outgoing_tokens'])).value)
    return adapter


def ram_run(monkeypatch,delta=0.,fail=None):
    p = protocol(); buffers = {}; payloads = []; rows = []; errors = []
    def save(folder,name,value):
        a = np.asarray(value); key = name+'.npy'; buffers[key] = a
        return dict(file=key,shape=list(a.shape),dtype=str(a.dtype))
    monkeypatch.setattr(m,'inputs_for',source)
    monkeypatch.setattr(m.mid,'save_array',save)
    monkeypatch.setattr(m.mid,'write',lambda path,value:None)
    journal = lambda e:payloads.append(deepcopy(e))
    ledger = m.Ledger(p,journal,lambda name,t:save(None,name,t.value),precision=m.mid.precision_contract)
    m.process(p,Path('RAM_ONLY'),fake_adapter(ledger,delta,fail),ledger,journal,rows,errors)
    r = dict(schema=m.NAME+'.report',status='completed' if not errors else 'partial_or_failed',contract=m.contract(),
        rows=rows,entries=ledger.entries,counts=ledger.counts(),errors=errors,blocked=ledger.blocked,journal_errors=ledger.journal_errors,
        host=dict(device='cuda:0'),source_verified=True,model_hash_before='RAM_NOT_WEIGHTS',model_hash_after='RAM_NOT_WEIGHTS',
        model_bindings=p['model_bindings'],precision_before=m.mid.precision_contract(),precision_after=m.mid.precision_contract(),
        fixed_slots=40,route_slots=80,total_wall_seconds=1.,new_AD_calls=0,new_physics_steps=0,equal_FLOPs=None,equal_total_compute=None,**m.FLAGS)
    return p,r,payloads,buffers,ledger


def validate(p,r,events,buffers):
    return m.validate_payload(p,r,events,lambda receipt:buffers[receipt['file']],source)


def test_real_callback_wiring_fixed40_not_P2_eight(monkeypatch):
    p,r,events,arrays,ledger = ram_run(monkeypatch)
    outputs = validate(p,r,events,arrays)
    assert len(outputs) == 80 and len(r['entries']) == 360 and len(r['rows']) == 80
    assert [q['route'] for q in m.plans(p)][:4] == list(m.ROUTES)*2
    assert [r['index'] for r in r['rows']] == [i for i in range(40) for _ in range(2)]
    assert p['jobs'][12]['actual192'] is None and r['rows'][24]['status'] == r['rows'][25]['status'] == 'saved'
    assert r['counts']['rollout']['returned'] == 40
    assert r['counts']['predict']['input_temporal_rows'] == 240
    assert r['counts']['pred_proj']['input_axis0_samples'] == 240
    assert r['counts']['encoder']['entered'] == 0 and ledger.order[-1] == ('slot39.cached_suffix','pred_proj')


def test_route_inequality_retained_not_filtered(monkeypatch):
    p,r,events,arrays,_ = ram_run(monkeypatch,delta=.125)
    outputs = validate(p,r,events,arrays)
    monkeypatch.setattr(m.teacher,'load_array',lambda base,receipt:outputs[2*receipt['index']]['predictions'][0])
    result = m.summarize(p,outputs)
    assert len(result['rows']) == 40 and result['all40_routes_exact_equal'] is False
    assert result['rows'][0]['first_unequal_buffer'] is not None
    for route in m.ROUTES:
        a = result['route_rank_errors'][route]
        assert a['rows'][12]['actual_Q'] is None
        assert a['within_role_summary']['unknown_pairs'] == 4
        assert a['same_word_ZS_summary']['unknown_pairs'] == 1
        assert a['rank_rows'] == dict(fixed=40,known=35,unknown=5)


@pytest.mark.parametrize('mutation',['drop_slot','duplicate_slot','P2_roster','swap_condition','fill_missing','alter_job_digest'])
def test_fixed_roster_rejects_mutations(mutation):
    p = protocol()
    if mutation == 'drop_slot': p['jobs'].pop()
    elif mutation == 'duplicate_slot': p['jobs'][-1] = deepcopy(p['jobs'][38])
    elif mutation == 'P2_roster': p['jobs'] = p['jobs'][:8]
    elif mutation == 'swap_condition': p['jobs'][0]['condition'] = 'on'
    elif mutation == 'fill_missing': p['jobs'][12]['actual192'] = [0.]*192
    else: p['jobs'][0]['goal192'][0] = 1.
    with pytest.raises(ValueError): m.validate_jobs(p['jobs'])


@pytest.mark.parametrize('mutation',['drop_row','swap_row','pending','lost_entry','lost_return','invent_call','counts','axes','equalflops','model','precision','CPU','flags','extra_AD','journal_drop','journal_reorder','fake_source'])
def test_full_actual_denominator_and_topology_rejected(monkeypatch,mutation):
    p,r,events,arrays,_ = ram_run(monkeypatch)
    if mutation == 'drop_row': r['rows'].pop()
    elif mutation == 'swap_row': r['rows'][0],r['rows'][1] = r['rows'][1],r['rows'][0]
    elif mutation == 'pending': r['rows'][25]['status'] = 'blocked_not_attempted'
    elif mutation == 'lost_entry': r['entries'].pop()
    elif mutation == 'lost_return': r['entries'][3]['native_returned'] = False
    elif mutation == 'invent_call': r['entries'][0]['kind'] = 'encode'
    elif mutation == 'counts': r['counts']['predict']['returned'] = 79
    elif mutation == 'axes': r['counts']['pred_proj']['input_axis0_samples'] = 80
    elif mutation == 'equalflops': r['equal_FLOPs'] = True
    elif mutation == 'model': r['model_hash_after'] = 'changed'
    elif mutation == 'precision': r['precision_after']['matmul_tf32'] = True
    elif mutation == 'CPU': r['host']['device'] = 'cpu'
    elif mutation == 'flags': r['P1_admission'] = True
    elif mutation == 'extra_AD': r['new_AD_calls'] = 1
    elif mutation == 'journal_drop': events.pop()
    elif mutation == 'journal_reorder': events[1],events[2] = events[2],events[1]
    else: r['source_verified'] = False
    with pytest.raises(ValueError): validate(p,r,events,arrays)


@pytest.mark.parametrize('key',['action_encoder.input_0','predict.input_0','predict.input_1','predictor.input_0','predictor.input_1',
    'pred_proj.input_0','predict.output_output','rollout.emb','rollout.action_sequence','rollout.output_predicted_emb'])
def test_actual_tensor_mutations_rejected(monkeypatch,key):
    p,r,events,arrays,_ = ram_run(monkeypatch)
    name = 'slot00.public_rollout.'+key+'.npy'; x = arrays[name].copy(); x.flat[0] += .125; arrays[name] = x
    with pytest.raises(ValueError): validate(p,r,events,arrays)


@pytest.mark.parametrize('key',['emb','action_sequence','pixels'])
def test_public_shape_not_relaxed(monkeypatch,key):
    p,r,events,arrays,_ = ram_run(monkeypatch)
    name = 'slot00.public_rollout.rollout.'+key+'.npy'; arrays[name] = arrays[name].reshape(-1)
    with pytest.raises(ValueError): validate(p,r,events,arrays)


def test_failure_retains_all80_and_spent_calls_no_retry(monkeypatch):
    p,r,events,arrays,ledger = ram_run(monkeypatch,fail='slot12.public_rollout')
    assert len(r['rows']) == 80 and r['rows'][24]['status'] == 'failed'
    assert all(row['status'] == 'blocked_not_attempted' for row in r['rows'][25:])
    assert r['counts']['predict']['returned'] == 24 and len(ledger.entries) == 108
    with pytest.raises(ValueError): validate(p,r,events,arrays)


def test_forbidden_encode_never_enters(monkeypatch):
    p,r,events,arrays,ledger = ram_run(monkeypatch)
    ledger.active = 'slot39.cached_suffix'
    called = []
    with pytest.raises(ValueError): ledger.dispatch('encode',lambda:called.append(1),(),{})
    assert not called and ledger.blocked


def test_durable_journal_chain(tmp_path):
    path = tmp_path/'events.jsonl'
    with path.open('x') as stream:
        j = m.Journal(stream); j(dict(event='RAM_only')); j(dict(event='done'))
    lines = path.read_bytes().splitlines(keepends=True)
    es = [m.rgb.decode(line) for line in lines]
    assert es[0]['previous_sha256'] is None and es[1]['previous_sha256'] == es[0]['event_sha256']
    assert es[1]['event_sha256'] == j.previous and len(lines) == j.index == 2


def test_bad_authority_before_model_import(tmp_path):
    before = set(sys.modules)
    with pytest.raises(ValueError): m.run_cuda('nonexistent','0'*64,tmp_path/'run','not_MAIN')
    assert set(sys.modules)-before == set()
    assert not (tmp_path/'run').exists()


def test_signed_zero_literal_same_is_not_relaxed():
    with pytest.raises(ValueError): m.same(np.asarray([0.],np.float32),np.asarray([-0.],np.float32),'signedzero')


def test_inputs_real_normalizer_and_cache_reconstruction(monkeypatch):
    warm = np.tile(np.asarray([0.,.1],'<f4'),(5,1)); prefix = np.zeros((5,2),'<f4'); suffix = -warm
    raw = np.concatenate((warm,prefix)); z = np.arange(576,dtype=np.float32).reshape(3,192)
    tokens = np.concatenate((m.teacher.kernel.normalized_history(raw),m.teacher.kernel.normalized_history(suffix)))
    j = dict(frames=[dict(path='RAM',file_sha256='RAM',pixel_sha256='RAM')]*3,compiler_folder='RAM',latent_receipt={},
        full3=z.tolist(),raw_history_bits=m.history.bits(raw),raw_suffix_bits=m.history.bits(suffix),outgoing_tokens=tokens.tolist(),goal192=[0.]*192)
    monkeypatch.setattr(m.teacher,'rgb',lambda *a:RGB[0])
    monkeypatch.setattr(m.teacher,'load_array',lambda *a:z)
    out = m.inputs_for(j,dict(mean=[.5]*3,std=[.25]*3))
    assert out['normalized_pixels'].shape == (1,3,3,224,224)
    assert np.array_equal(out['full3'][0],z) and np.array_equal(out['outgoing_tokens'][0],tokens)
    j['outgoing_tokens'][1][0] += 1
    with pytest.raises(ValueError): m.inputs_for(j,dict(mean=[.5]*3,std=[.25]*3))
