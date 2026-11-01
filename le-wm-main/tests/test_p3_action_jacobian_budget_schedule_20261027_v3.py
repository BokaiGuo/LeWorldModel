"""Array/metadata contracts only: no neural forward, simulator or CUDA."""
from copy import deepcopy
from pathlib import Path
import builtins
import hashlib
import importlib.util

import numpy as np
import pytest

from experiments.spectral_reachability import p3_action_jacobian_budget_schedule_20261027_v3 as v3

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    q=np.eye(192); j=np.zeros((192,10)); j[0,0]=2
    nominal=np.zeros(192); nominal[1]=7
    goal=nominal.copy(); goal[1]+=3
    return j,nominal,goal,q


def test_action_column_space_not_drift_or_singular_values_only():
    j,n,g,q=fixture()
    s=v3.jacobian_signal(j,n,g,q)
    assert s['chi']==pytest.approx(1) and s['response_rank']==1
    assert s['residual'][1]==3 and s['residual_squared_norm']==9
    g=n.copy();g[0]=3
    t=v3.jacobian_signal(j,n,g,q)
    assert t['chi']==pytest.approx(0) and s['singular_values']==t['singular_values']
    # Autonomous drift changes nominal, not the action image; r uses nominal.
    shift=np.ones(192)*100
    assert v3.jacobian_signal(j,n+shift,g+shift,q)['chi']==t['chi']


def test_projection_and_relative_cutoff_determine_action_image():
    j,n,g,q=fixture();n[:]=0;g[:]=0;g[1]=1
    j[1,1]=1e-8  # discarded relative to singular value 2, not absolute rank
    assert v3.jacobian_signal(j,n,g,q)['chi']==pytest.approx(1)
    j[1,1]=1
    assert v3.jacobian_signal(j,n,g,q)['chi']==pytest.approx(0)
    q[0,0]=0
    assert v3.jacobian_signal(j,n,g,q)['response_rank']==1


@pytest.mark.parametrize('bad',['shape','nan','zero_rank','zero_residual','bad_q','bool'])
def test_invalid_geometry_no_uniform_fallback(bad):
    j,n,g,q=fixture()
    if bad=='shape':j=j[:,:2]
    elif bad=='nan':j[0,0]=np.nan
    elif bad=='zero_rank':j[:]=0
    elif bad=='zero_residual':g=n
    elif bad=='bad_q':q[:]=1
    else:j=j.astype(bool)
    with pytest.raises(ValueError):v3.jacobian_signal(j,n,g,q)


def load_metadata():
    path=ROOT/'tests/test_p3_fixed_horizon_budget_schedule_cuda_20261027_v2.py'
    spec=importlib.util.spec_from_file_location('sealed_metadata_fixtures',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def arm(mode='uniform',fail_at=None,mismatch=False):
    f=load_metadata();model=f.Model(fail_at);p=f.packet()
    # Production geometry remains exactly 192x10 even in metadata tests.
    p['root_context']['latent_dim']=192
    p['root_context']['latent_history']=[[0.]*192 for _ in range(3)]
    j,n,g,q=fixture();p.update(goal=g.tolist(),projection=q.tolist(),
        inherited_jacobian=j.tolist(),inherited_nominal=n.tolist())
    def replay(c,actions):
        child=deepcopy(c)
        for a in np.asarray(actions).reshape(-1,5,2):
            model.predict(f.Metadata(192),model.action_encoder(f.Metadata(10)))
            x=np.asarray(child['latent_history'][-1]);x[0]+=float(a[0,0])+1
            child['latent_history']=child['latent_history'][1:]+[x.tolist()]
            child['action_history']=child['action_history'][1:]+[[float(a[0,0]),0.]*5]
        if mismatch and len(actions)==15:child['latent_history'][-1][0]+=.01
        return child
    row=v3.run_arm(model=model,replay=replay,p=p,
        binding=f.Binding('seal','task','a'*64,'model'),mode=mode)
    assert not model.action_encoder.hooks and not model.encoder.hooks
    assert 'predict' not in vars(model)
    return row,model


@pytest.mark.parametrize('mode',['uniform','dynamic'])
def test_144_plus_3_fullbank_h3_and_same_signal(mode):
    row,model=arm(mode)
    assert row['status']=='completed' and model.calls==147
    assert row['search']['allocation']==([4,4] if mode=='uniform' else [6,2])
    assert row['search']['signal']['chi']==pytest.approx(1)
    assert row['search']['best_plan']['depth']==3
    assert len(row['search']['events'])==144 and row['search']['svd_calls']==1
    assert row['action_encoder_samples']==147 and row['processed_action_context_tokens']==441
    assert [x['predict_samples'] for x in row['prediction_receipt']['phases']]==[144,3]
    assert row['inherited_cost']['predict_samples'] is None
    assert row['new_jacobian_materializations']==0


@pytest.mark.parametrize('fail_at',[1,16,17,144,145,147])
def test_failed_work_retained_no_refund_or_substitute_winner(fail_at):
    row,model=arm(fail_at=fail_at)
    assert row['status']=='incomplete'
    assert row['prediction_receipt']['attempted_predict_samples']==model.calls
    assert sum(x['failed_predict_calls'] for x in row['prediction_receipt']['phases'])==1


def test_whole_path_verification_mismatch_preserves_cost():
    row,model=arm(mismatch=True)
    assert row['status']=='incomplete' and model.calls==147 and not row['whole_h3_replay_exact']


def test_pairs_match_root_task_model_signal_and_no_total_cost_claim():
    u,_=arm();d,_=arm('dynamic')
    assert v3.compare([d,u],['a'*64])[0]['classification']=='matched_full_H3'
    d['search']['signal']['chi']=0
    assert v3.compare([d,u],['a'*64])[0]['classification']=='unmatched'
    with pytest.raises(ValueError):v3.compare([u,u],['a'*64])


def test_same_signal_both_arms_and_binding_mismatch_not_matched():
    u,_=arm();d,_=arm('dynamic')
    assert u['search']['signal']==d['search']['signal']
    d['binding']['task_id']='foreign task'
    assert v3.compare([u,d],['a'*64])[0]['classification']=='unmatched'


def test_incomplete_layer_keeps_duplicate_edges_and_no_shallow_winner():
    row,_=arm(fail_at=17)
    assert len(row['search']['events'])==80
    assert row['search']['best_plan'] is None
    # Every scheduled parent still attempts all sixteen, no dedup/filter/refund.
    for parent in row['search']['layers'][1]['parents']:
        assert [e['primitive_index'] for e in row['search']['events'] if e['depth']==2 and e['parent_path']==parent]==list(range(16))


@pytest.fixture(scope='module')
def inherited():
    body,_=v3.old.validate_prepared(ROOT/v3.OLD_PREPARED,v3.OLD_PREPARE_SHA)
    report=v3.io.decode((ROOT/v3.JACOBIAN_REPORT).read_bytes())
    compilation=v3.io.decode((ROOT/v3.old.COMPILATION).read_bytes())
    return body,report,compilation


def test_real_inheritance_is_exact_same_root_context_goal_q_and_frozen_model(inherited):
    packets=v3.inherit_packets(*inherited)
    assert len(packets)==4
    assert all(np.asarray(p['inherited_jacobian']).shape==(192,10) for p in packets)
    assert all(v3.jacobian_signal(p['inherited_jacobian'],p['inherited_nominal'],p['goal'],p['projection'])['chi']>=0 for p in packets)
    assert [p['primitive_bank'] for p in packets]==[p['primitive_bank'] for p in inherited[0]['packets']]


def test_old_selected_outcomes_are_not_signal_or_bank_inputs(inherited):
    body,report,compilation=deepcopy(inherited)
    expected=v3.inherit_packets(body,report,compilation)
    for plan in report['plans']:plan['arms']=['selected/outcomes deliberately ignored']
    for row in compilation['rows']:
        row['compiled']['selected_candidate_id']='not an input'
        for receipt in row['compiled']['receipts']:
            receipt['predicted_terminal_vector']=['not an input']
    assert v3.inherit_packets(body,report,compilation)==expected


@pytest.mark.parametrize('mutation',['root','model','compilation','goal','context','missing','duplicate','fd','training'])
def test_inheritance_binding_mutations_fail_closed(inherited,mutation):
    body,report,compilation=deepcopy(inherited)
    if mutation=='root':report['plans'][0]['root_sha256']='0'*64
    elif mutation=='model':report['protocol']['bindings'][next(iter(body['model_bindings']))]='0'*64
    elif mutation=='compilation':report['protocol']['compilation']='foreign.json'
    elif mutation=='goal':body['packets'][0]['goal'][0]+=.1
    elif mutation=='context':body['packets'][0]['root_context']['latent_history'][-1][0]+=.1
    elif mutation=='missing':report['plans'].pop()
    elif mutation=='duplicate':report['plans'][1]=deepcopy(report['plans'][0])
    elif mutation=='fd':report['plans'][0]['full_fd_relative_error']=1
    else:report['training']=True
    with pytest.raises(ValueError):v3.inherit_packets(body,report,compilation)


def test_prepare_no_neural_import_and_strict_hashes(tmp_path,monkeypatch):
    original=builtins.__import__
    def guard(name,*a,**kw):
        assert name.split('.')[0] not in {'torch','gymnasium','mujoco','pymunk'},name
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guard)
    out=tmp_path/'new_prepare';receipt=v3.prepare(ROOT,out)
    h=hashlib.sha256((out/'prepare.json').read_bytes()).hexdigest()
    body,identity=v3.validate_prepared(out,h)
    assert len(body['packets'])==4 and receipt['new_predict_samples']==0
    assert receipt['model_loaded'] is False and body['expected_predict_samples']==1176
    assert body['inherited_cost']['predict_samples'] is None
    assert body['total_pipeline_cost_equivalent'] is False
    assert receipt['prepare_numpy_svd_calls']==body['prepare_numpy_svd_calls']==4
    assert body['contract']['dynamic']=='K2=2+floor(4*chi+0.5); K3=8-K2'
    with pytest.raises(ValueError):v3.validate_prepared(out,'0'*64)
    with pytest.raises(ValueError):v3.prepare(ROOT,out)
