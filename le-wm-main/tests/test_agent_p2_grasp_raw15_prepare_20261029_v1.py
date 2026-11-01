"""RAM arithmetic / pinned input metadata only; no torch, NN, physics or outcomes."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_grasp_raw15_prepare_20261029_v1 as m


@pytest.fixture(scope='module')
def original():
    # Real gzip export regression, not a fake plain JSON schema. No report,
    # selection, saved outcome, NPZ, image or checkpoint is opened.
    p=m.parse(m.read_bound(m.UPSTREAM,m.UPSTREAM_SHA))
    es=[m.parse(m.read_bound(s['task_file'],s['task_sha256'],compressed=True)) for s in p['slots']]
    return p,es


def test_exact_saved_identity_and_no_H2_relabel(original):
    p,es=original;r=m.make_request(p,es)
    assert [(s['root_slot'],s['condition']) for s in r]==[(0,'off'),(1,'off')]
    for s,old,e in zip(r,p['slots'],es,strict=True):
        assert s['model_input']['root_context']==old['packet']['root_context']
        assert s['model_input']['goal']==old['packet']['goal']
        assert s['task_id']==m.core.task_id(e['task'])
        assert s['source_export_horizon']==e['task']['horizon']==2
        assert s['target_future_packets']==3 and s['target_raw_steps']==15
        assert s['blind_pointer']['opened'] is False and s['observed_warmup_steps']==[0,5,10]
        assert np.asarray(s['first_call_preview']['first_latent_input']).shape==(3,192)
        assert np.asarray(s['first_call_preview']['first_normalized_actions']).shape==(3,10)
        assert s['first_call_preview']['future_contexts'] is None


def test_native_normalized_zero_is_not_raw_zero(original):
    _,es=original
    x=m.initial_coordinates(es[0]['task'])
    raw=np.asarray(x['p2_zero_logit_raw15'],np.float32)
    norm=np.asarray(x['p2_zero_logit_normalized3'],np.float32)
    assert not raw.any() and norm.any() and x['initializations_equal'] is False
    assert np.array_equal(norm,np.tile(-np.asarray(m.normalization()['mean'],np.float32)/
                           np.asarray(m.normalization()['std'],np.float32), (3,5)))
    assert np.array_equal(np.asarray(x['official_zero_normalized_inverse_raw15']),
                          np.tile(es[0]['task']['action_mean'],(15,1)))


def test_real_raw15_packet_arithmetic_not_double_division(original):
    p,es=original;s=m.make_request(p,es)[0]['model_input']
    u=np.linspace(-.4,.4,30,dtype=np.float64)
    r=m.raw15_inputs(s,u)
    raw=np.tanh(u).astype(np.float32).reshape(15,2)
    expected=np.concatenate([m.core.normalized_tokens(es[0]['task'],raw[k:k+5]) for k in (0,5,10)])
    assert np.array_equal(r['normalized_packets'],expected)
    assert r['raw15_sha256']==m.sha(raw.tobytes())
    assert np.array_equal(r['first_normalized_actions'][:2],s['root_context']['action_history'])
    assert np.array_equal(r['first_normalized_actions'][2],expected[0])
    assert not np.array_equal(expected,expected/100)


@pytest.mark.parametrize('change',[
    lambda p,e:p['slots'].reverse(),
    lambda p,e:p['slots'].pop(),
    lambda p,e:p['slots'][0].update(root_slot=True),
    lambda p,e:p['slots'][0].update(condition='on'),
    lambda p,e:p['contract'].update(raw_steps=10),
    lambda p,e:p['contract'].update(horizon=2),
    lambda p,e:p['contract'].update(known_roots=4),
    lambda p,e:p['contract'].update(logit_cap=99.),
    lambda p,e:p['contract'].update(fresh=True),
    lambda p,e:p['contract'].update(P1_admission=True),
    lambda p,e:p.update(expected_model_hash='f'*64),
    lambda p,e:p['slots'][0].update(task_sha256='f'*64),
    lambda p,e:p['slots'][0]['packet'].update(native_success=True),
    lambda p,e:p['slots'][0].update(reward=0.),
    lambda p,e:e[0].update(schema='old.raw15.fake'),
    lambda p,e:e[0].update(task_id='f'*64),
    lambda p,e:e[0]['task'].update(horizon=3),
    lambda p,e:e[0]['task'].update(action_mean=[0.,0.]),
    lambda p,e:e[0]['task'].update(action_std=[1.,1.]),
    lambda p,e:e[0]['task']['root_raw_action_history'].__setitem__(0,[0.,0.]),
    lambda p,e:e[0]['task']['root_context']['latent_history'].pop(0),
    lambda p,e:e[0]['task']['root_context']['action_history'].__setitem__(0,[0.]*10),
    lambda p,e:e[0]['task_contract'].update(condition='on'),
    lambda p,e:e[0]['task_contract'].update(source_report_sha256='f'*64),
    lambda p,e:e[0]['task_contract'].update(goal_pose_edited=True),
])
def test_mutations_refused_without_access_to_outcomes(original,change):
    p,es=deepcopy(original);change(p,es)
    with pytest.raises((ValueError,KeyError)):m.make_request(p,es)


@pytest.mark.parametrize('u',[[0.]*20,[0.]*31,[float('nan')]*30,[float('inf')]*30,[1.]*30])
def test_bad_shape_and_outside_domain_never_repaired(original,u):
    p,es=original;r=m.make_request(p,es)[0]['model_input']
    with pytest.raises(ValueError):m.raw15_inputs(r,u)


def test_public_request_rejects_evaluator_fields(original):
    p,es=original;r=deepcopy(m.make_request(p,es)[0]['model_input']);r['state7']=[0.]*7
    with pytest.raises(ValueError,match='whitelist'):m.raw15_inputs(r,np.zeros(30))


def test_budget_uses_actual_official_loop_semantics_not_H2():
    b=m.budget();assert b['sync_after']==list(range(50,751,50))
    assert b['official_solver_predict_per_root']==750*3+15*10*3==2700
    assert b['official_solver_backward_per_root']==900
    assert b['prospective_with_dual_replay_predict_per_root']==2706
    assert b['prospective_with_dual_replay_predict_two_roots']==5412
    assert b['actual_attempts'] is b['actual_returns'] is b['FLOPs'] is b['actual_wall'] is None
    assert b['equal_API'] is b['equal_total_compute'] is False
    assert 'OLDER' in b['P2_standalone_scope']
    current=b['current_common_cap_MAIN_notice']
    assert current['common_predict_cap']==1902 and current['wall_cap_seconds']==450
    assert current['joint_actual_predict']==1902 and current['composition_actual_predict']==1062
    assert current['J_each']==85 and current['use_as_certified_saved_authority'] is False


def test_source_facts_have_real_lines_and_untouched_foreign_text():
    f=m.official_facts()
    assert f['repositories']['LeWM_GRASP']['commit']=='94be3c42aa2f0867c6cb1e38bda498b293b6921e'
    assert f['configuration']['n_steps']==750 and f['seed_binding']=='${seed}'
    assert len(f['anchors'])==7
    for x in f['anchors']:
        assert x['start_line_1based']>0 and x['end_line_1based']>=x['start_line_1based']
        assert m.sha(x['excerpt'].encode())==x['excerpt_sha256']
    assert f['official_frozen_process'] is None and f['paper_table_code_revision'] is None


@pytest.mark.parametrize('fake',[{},dict(official_same_input_ready=True,production_dispatch=True),
                              dict(differences=[],P1_admission=True)])
def test_forged_ready_or_empty_gap_cannot_dispatch(fake):
    with pytest.raises(ValueError,match='no dispatch'):m.official_execution_request(fake)


@pytest.mark.parametrize('payload',[b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}'])
def test_noncanonical_json_refused(payload):
    with pytest.raises(ValueError):m.parse(payload)


def test_gzip_integrity_cap_and_real_schema(tmp_path):
    p=tmp_path/'input.gz';valid=gzip.compress(b'{"schema":"RAM"}',mtime=0)
    p.write_bytes(valid);assert m.parse(m.read_bound(p,m.sha(valid),compressed=True))==dict(schema='RAM')
    for raw in (valid[:-3],valid+b'extra',valid+valid,gzip.compress(b'a'*(m.LIMIT+1),mtime=0)):
        p.write_bytes(raw)
        with pytest.raises(ValueError):m.read_bound(p,m.sha(raw),compressed=True)
    p.write_bytes(valid)
    with pytest.raises(ValueError):m.read_bound(p,'f'*64)


def test_exclusive_own_metadata_writer_and_full_rebuild(tmp_path,monkeypatch):
    path=tmp_path/(m.NAME+'_RAM.json');value=dict(status='fixture_blocked',production_dispatch=False)
    seal=m.write_new(path,value);monkeypatch.setattr(m,'build',lambda:deepcopy(value))
    assert m.verify(path,seal['sha256'])==value
    with pytest.raises(ValueError):m.write_new(path,value)
    value['production_dispatch']=True
    with pytest.raises(ValueError):m.verify(path,seal['sha256'])
    with pytest.raises(ValueError):m.write_new(tmp_path/'not_own.json',value)


def test_saved_metadata_build_does_not_read_outcomes_NPZ_weights_or_reserved(monkeypatch):
    old=Path.read_bytes;opened=[]
    def read(path):
        s=str(path);opened.append(s)
        assert not s.endswith(('.npz','.pt','.ckpt','.npy','.png'))
        assert not s.endswith('report.json') and 'reserved' not in s.lower()
        return old(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    v=m.build()
    assert len(v['slots'])==2 and all(not v[k] for k in (
        'official_same_input_ready','production_dispatch','physical_release','P1_admission','P2_admission'))
    assert len(v['differences'])==5 and v['unknown']['official_process'] is None
    assert str(m.UPSTREAM) in opened
    assert v['declared_checkpoint']['model_weights_opened'] is False


def test_cold_import_no_neural_foreign_or_network():
    script='''import builtins,socket,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','tensorflow','gymnasium','sklearn','transformers','grasp_solver'):
  raise AssertionError('forbidden import '+name)
 return original(name,*args,**kwargs)
builtins.__import__=guard
def deny_network(event,args):
 if event in ('socket.__new__','socket.connect','socket.getaddrinfo'):
  raise AssertionError('network '+event)
sys.addaudithook(deny_network)
from experiments.spectral_reachability import agent_p2_grasp_raw15_prepare_20261029_v1 as m
assert m.budget()['official_solver_predict_per_root']==2700
assert m.official_facts()['sync_mode']=='gd'
assert 'torch' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=m.ROOT,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
