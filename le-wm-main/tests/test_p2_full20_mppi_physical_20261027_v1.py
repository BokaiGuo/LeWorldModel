"""Saved evidence and fake guards only; no simulator jobs or neural execution."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p2_full20_mppi_physical_20261027_v1 as m
from tests.test_p2_matched_sampling_physical_20261027_v1 import Env, state


@pytest.fixture(scope='module')
def saved():
    path = m.DEFAULT_PLANNING
    if not (path/'report.json').is_file(): pytest.skip('main-thread MPPI CUDA report not yet available')
    _,protocol,prepared,compilation = m.sampling.load_prepared(path)
    report = json.loads((path/'report.json').read_text())
    return report,protocol,prepared,compilation


def test_actual_four336_receipts_root_join_and_actions(saved):
    report,protocol,prepared,compilation = deepcopy(saved)
    report['rows'].reverse()
    compilation['rows'].reverse()
    plans = m.plans_from_saved(report,protocol,prepared,compilation)
    assert len(plans) == len({p['root_sha256'] for p in plans}) == 4
    assert [p['root_index'] for p in plans] == list(range(4))
    assert report['actual_predict_samples'] == 336
    assert all(p['method'] == m.sampling.ARM and np.asarray(p['commands']).shape == (10,2) for p in plans)
    for p in plans:
        assert p['selection_sha256'] == m.digest({k:v for k,v in p.items() if k != 'selection_sha256'})


@pytest.mark.parametrize('defect',['root','duplicate','missing','arm','fixture','samples','entry','budget_root',
    'encoder','receipt','provenance','metric','candidate','kernel','total','seed','source','gate'])
def test_actual_saved_mutations_reject(saved,defect):
    report,protocol,prepared,compilation = deepcopy(saved)
    row = report['rows'][0]
    arm = row['rows'][0]
    if defect == 'root': row['root']['context']['latent_history'][0][0] += 1
    if defect == 'duplicate': report['rows'][-1] = deepcopy(row)
    if defect == 'missing': report['rows'].pop()
    if defect == 'arm': arm['arm'] = 'cem'
    if defect == 'fixture': arm['fixture_only'] = True
    if defect == 'samples': arm['planning_predict_samples'] = 79
    if defect == 'entry': arm['actual_model_entries'].pop()
    if defect == 'budget_root': arm['actual_prediction_budget']['binding']['root_id'] = '0'*64
    if defect == 'encoder': arm['action_encoder_work']['processed_context_tokens'] -= 1
    if defect == 'receipt': arm['witness']['receipt']['actions'][0][0] += .01
    if defect == 'provenance': arm['witness']['receipt']['provenance']['metric_sha256'] = '0'*64
    if defect == 'metric': arm['selected_model_cost'] += 1
    if defect == 'candidate': arm['attempts'][1]['commands'][0][0][0] += .01
    if defect == 'kernel': arm['kernel_receipt']['cost_evaluations'] = 39
    if defect == 'total': report['actual_predict_samples'] = 335
    if defect == 'seed': arm['seed'] += 1
    if defect == 'source': report['source_seals'][0]['manifest_sha256'] = '0'*64
    if defect == 'gate': report['gates'] = {}
    with pytest.raises(ValueError): m.plans_from_saved(report,protocol,prepared,compilation)


def fake_plan():
    p = dict(root_index=0,root_sha256='a'*64,method=m.sampling.ARM,
        planned_horizon_tokens=2,planned_commands=10,corrected_search_horizon=2,
        commands=np.linspace(-.9,.9,20).reshape(10,2).tolist())
    p['selection_sha256'] = m.digest(p)
    return p


@pytest.mark.parametrize('options,count,status',[({},10,'completed'),({'missing_at':0},0,'observation_unavailable'),
    ({'missing_at':3},3,'observation_unavailable'),({'fail_at':4},3,'execution_failed'),
    ({'terminate_at':2},2,'environment_terminated')])
def test_durable_guard_persists_each_attempt_and_exact_action_prefix(tmp_path,options,count,status):
    plan = fake_plan()
    row = m.physical_branch(plan=plan,output=tmp_path,reset=lambda:{'fixture':True},env=Env(**options),
        state_of=state,evaluate=lambda s:dict(official={},pose={}))
    assert row['status'] == status and row['completed_commands'] == count
    assert [e['command'] for e in row['trace']] == plan['commands'][:count]
    events = [json.loads(line) for line in Path(row['journal_path']).read_text().splitlines()]
    assert sum(e['kind'] == 'attempt' for e in events) == row['physical_env_step_attempts']
    assert len(list(tmp_path.glob('*.physical.json'))) == 1
    assert json.loads(next(tmp_path.glob('*.physical.json')).read_text()) == row


def test_plan_hash_rejected_before_fake_environment_access(tmp_path):
    p = fake_plan()
    p['commands'][0][0] += .01
    with pytest.raises(ValueError,match='selection digest'):
        m.physical_branch(plan=p,output=tmp_path,reset=lambda:pytest.fail('reset'),env=None,
                          state_of=lambda e:pytest.fail('state'),evaluate=lambda s:{})


@pytest.mark.parametrize('missing,expected',[(None,1),(3,0)])
def test_only_fixed_H2_encoded_and_physics_precedes_setup(tmp_path,missing,expected):
    row = m.physical_branch(plan=fake_plan(),output=tmp_path,reset=lambda:{},env=Env(missing_at=missing),
        state_of=state,evaluate=lambda s:dict(official={},pose={}))
    calls = []
    def setup():
        assert (tmp_path/'physical_report.json').exists()
        return lambda p: calls.append(p) or np.zeros(192).tolist()
    with pytest.raises(ValueError,match='persist'): m.helpers.optional_encodings([row],tmp_path,setup)
    m.write(tmp_path/'physical_report.json',dict(rows=[row]))
    receipts = m.helpers.optional_encodings([row],tmp_path,setup)
    assert len(calls) == expected == receipts[0]['attempted_encode_samples']
    assert receipts[0]['status'] == ('completed' if expected else 'blocked_fixed_horizon')


def test_reset_failure_keeps_four_missing_roots_without_execution(tmp_path):
    plans = []
    for i in range(4):
        p = fake_plan()
        p.update(root_index=i,root_sha256=str(i)*64)
        plans.append(p)
    rows = []
    m.helpers.materialize_missing(plans,rows,tmp_path)
    assert len(rows) == len(list(tmp_path.glob('*.physical.json'))) == 4
    assert all(r['completed_commands'] == 0 and not r['full_horizon_completed'] for r in rows)


def test_real_prepare_has_no_environment_or_model_access(saved,tmp_path,monkeypatch):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    for name in ('require_cuda_allocation','load_compatible_pretrained','collect_nonterminal_boundary_contact_roots'):
        monkeypatch.setattr(base,name,lambda *a,**kw:pytest.fail('execution requested during prepare'))
    monkeypatch.setattr(base.gym,'make',lambda *a,**kw:pytest.fail('environment instantiated'))
    output = tmp_path/m.TAG
    m.prepare(m.DEFAULT_PLANNING,output)
    _,protocol,prepared,context = m.load_prepared(output)
    assert len(protocol['plans']) == 4 and protocol['total_planned_commands'] == 40
    assert protocol['actual_planning_predict_samples_verified'] == 336
    assert prepared['new_neural_calls'] == prepared['new_physical_steps'] == 0
    assert not (output/'physical_report.json').exists()
    assert protocol['environment_source_paths']
    assert str((m.DEFAULT_PLANNING/'report.json').resolve()) in protocol['bindings']
    names = set()
    for seal in prepared['source_seals']:
        manifest = json.loads((Path(seal['output_dir'])/'manifest.json').read_text())
        names.update(Path(e['source']).name for e in manifest['files'])
    assert {m.TAG+'.py','test_'+m.TAG+'.py',m.TAG+'.md',m.sampling.TAG+'.py',
        'p3_matched_physical_20261027_v1.py','pusht_rgb_observability_guard.py'} <= names
    with pytest.raises(FileExistsError): m.prepare(m.DEFAULT_PLANNING,output)


def test_run_persists_physics_before_optional_cuda_and_forbids_predict():
    source = inspect.getsource(m.run)
    assert source.index('load_prepared(prepared)') < source.index('base.gym.make')
    assert source.index("output/'physical_report.json'") < source.index('base.require_cuda_allocation')
    assert 'model.predict = forbid' in source
    assert 'helpers.optional_encodings' in source
