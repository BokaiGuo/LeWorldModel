"""Saved evidence and fake guards only; no simulator jobs or neural execution."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p2_no_composition_sra_physical_20261027_v2 as m
from tests.test_p2_matched_sampling_physical_20261027_v1 import Env, state


@pytest.fixture(scope='module')
def saved():
    path = m.DEFAULT_PLANNING
    if not (path/'report.json').is_file(): pytest.skip('main-thread SRA CUDA report not yet available')
    _,protocol,prepared = m.sampling.load_prepared(path)
    source = json.loads(Path(protocol['context']['compilation']).read_text())
    exported = m.sampling.selected_actions(path)
    return json.loads((path/'report.json').read_text()),protocol,prepared,source,exported['actions']


def test_real_four352_receipts_stable_identity_join(saved):
    report,protocol,prepared,source,actions = deepcopy(saved)
    report['rows'].reverse()
    source['rows'].reverse()
    # Only report/compilation arrays vary; the frozen protocol stays byte-bound.
    plans = m.plans_from_saved(report,protocol,prepared,source,actions)
    assert [p['root_index'] for p in plans] == [0,1,2,3]
    assert len({p['root_sha256'] for p in plans}) == 4
    for plan in plans:
        assert np.asarray(plan['commands']).shape == (10,2)
        assert plan['planning_accounting']['total_predict_samples'] == 88
        assert plan['selection_sha256'] == m.digest({k:v for k,v in plan.items() if k!='selection_sha256'})


@pytest.mark.parametrize('defect',['root','root_index','duplicate','missing','arm','fixture','samples',
    'entry','budget_root','encoder','receipt','metric','goal','projection','total','source','gate',
    'jacobian','svd','vjp','operations','claim','compile_arm','sample_limit','export_actions','export_missing'])
def test_actual_saved_mutations_reject(saved,defect):
    report,protocol,prepared,source,actions = deepcopy(saved)
    row = report['rows'][0]
    result = row['result']
    if defect == 'root': row['root']['context']['latent_history'][0][0] += 1
    if defect == 'root_index': row['root_index'] = 999
    if defect == 'duplicate': report['rows'][-1] = deepcopy(row)
    if defect == 'missing': report['rows'].pop()
    if defect == 'arm': result['arm'] = 'cem'
    if defect == 'fixture': result['fixture_only'] = True
    if defect == 'samples': result['planning_predict_samples'] = 79
    if defect == 'entry': row['actual_model_entries'].pop()
    if defect == 'budget_root': row['actual_prediction_budget']['binding']['root_id'] = '0'*64
    if defect == 'encoder': row['action_encoder_processed_context_token_samples'] -= 1
    if defect == 'receipt': result['witness']['receipt']['actions'][0][0] += .01
    if defect == 'metric': result['selected_model_cost'] += 1
    if defect == 'goal': next(r for r in source['rows'] if r['root_sha256']==row['root_sha256'])['compiled']['goal_vectors'][0][0] += 1
    if defect == 'projection': next(r for r in source['rows'] if r['root_sha256']==row['root_sha256'])['compiled']['scoring']['metric']['projection'][0][0] += 1
    if defect == 'total': report['actual_predict_calls'] = 336
    if defect == 'source': report['source_seals'][0]['manifest_sha256'] = '0'*64
    if defect == 'gate': report['gates'] = {}
    if defect == 'jacobian': row['compile_receipt']['completed_jacobian_calls'] = 0
    if defect == 'svd': row['compile_receipt']['completed_svd_calls'] = 1
    if defect == 'vjp': row['compile_receipt']['completed_scalar_output_vjp_directions'] = 1
    if defect == 'operations': row['compilation_operations'] = []
    if defect == 'claim': row['fair_total_budget_comparison'] = True
    if defect == 'compile_arm': row['actual_model_entries'][0]['scope']['arm'] = 'cem'
    if defect == 'sample_limit': row['actual_prediction_budget']['predict_sample_limit'] = 84
    if defect == 'export_actions': actions[row['root_sha256'],m.sampling.ARM][0,0] += .01
    if defect == 'export_missing': actions.pop((row['root_sha256'],m.sampling.ARM))
    with pytest.raises(ValueError): m.plans_from_saved(report,protocol,prepared,source,actions)


@pytest.mark.parametrize('defect',['candidate','bank','winner','provenance','fixture','missing_bank'])
def test_real_planner_exporter_rejects_saved_tampering(saved,tmp_path,monkeypatch,defect):
    report,protocol,prepared,source,actions = deepcopy(saved)
    row = report['rows'][0]
    if defect == 'candidate': row['result']['attempts'][1]['commands'][0][0][0] += .1
    if defect == 'winner': row['result']['selected_candidate_id'] = 'cem_00'
    if defect == 'provenance': row['result']['witness']['receipt']['provenance']['metric_sha256'] = '0'*64
    if defect == 'fixture': row['result']['fixture_only'] = True
    m.write(tmp_path/'report.json',report)
    for case in protocol['context']['cases']:
        name = f'root_{case["root_index"]}_bank.json'
        if defect == 'missing_bank' and case['root_index']==row['root_index']: continue
        bank = json.loads((m.DEFAULT_PLANNING/name).read_text())
        if defect == 'bank' and case['root_index']==row['root_index']: bank['commands'][0][0][0][0] += .1
        m.write(tmp_path/name,bank)
    monkeypatch.setattr(m.sampling,'load_prepared',lambda p:(tmp_path,protocol,prepared))
    with pytest.raises((ValueError,FileNotFoundError)):
        m.sampling.selected_actions(tmp_path)


def test_missing_real_report_refuses_before_load_or_output(tmp_path,monkeypatch):
    monkeypatch.setattr(m.sampling,'load_prepared',lambda *a:pytest.fail('load before report'))
    output = tmp_path/m.TAG
    with pytest.raises(ValueError,match='report missing'): m.prepare(tmp_path/'absent',output)
    assert not output.exists()


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
    assert protocol['actual_planning_predict_samples_verified'] == 352
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

def test_prepare_preserves_compile_overhead_and_binds_banks(saved,tmp_path):
    output = tmp_path/m.TAG
    m.prepare(m.DEFAULT_PLANNING,output)
    _,protocol,sealed,_ = m.load_prepared(output)
    assert protocol['planning_accounting'] == m.ACCOUNTING
    assert not protocol['prediction_budget_matched'] and not protocol['fair_total_budget_comparison']
    assert not protocol['P2_admission']
    assert protocol['planning_accounting']['four_root']['total_predict_samples'] == 352
    for i in range(4):
        assert str(m.DEFAULT_PLANNING/f'root_{i}_bank.json') in protocol['bindings']
    assert sealed['new_neural_calls'] == sealed['new_physical_steps'] == 0


def test_source_seal_protocol_binding_mutation_refused(saved,tmp_path):
    output = tmp_path/m.TAG
    m.prepare(m.DEFAULT_PLANNING,output)
    protocol = json.loads((output/'protocol.json').read_text())
    protocol['planning_accounting']['per_root']['total_predict_samples'] = 84
    # Deliberate corruption of test-only artifacts; production outputs untouched.
    (output/'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError): m.load_prepared(output)


def test_exporter_already_present_in_original_pre_execution_archive(saved):
    proof = m.verify_archived_exporter(saved[2])
    assert proof['sha256'] == m.EXPECTED_PLANNER_SHA256
    assert 'source_seal_1/sources/004.source' in proof['archive']
    assert m.sha(m.DEFAULT_PLANNING/'report.json') == m.EXPECTED_REPORT_SHA256
    assert '\ndef selected_actions(output):' in Path(proof['archive']).read_text()


def test_no_retroactive_exporter_reseal_allowed(saved):
    prepared = deepcopy(saved[2])
    prepared['source_seals'] = []
    with pytest.raises(ValueError,match='unique pre-execution'): m.verify_archived_exporter(prepared)

