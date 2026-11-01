"""Saved planner evidence, pure arrays and fake guards; no real physics/neural work."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p2_composition_paired_physical_20261027_v2 as m
from experiments.pusht_protocol_contracts import root_fingerprint
from tests.test_p2_matched_sampling_physical_20261027_v1 import Env, state


@pytest.fixture(scope='module')
def saved():
    if not (m.DEFAULT_PLANNING/'report.json').is_file():
        pytest.skip('main paired planner report not yet available')
    _,protocol,prepared=m.sampling.load_prepared(m.DEFAULT_PLANNING)
    report=json.loads((m.DEFAULT_PLANNING/'report.json').read_text())
    source=json.loads(Path(protocol['context']['compilation']).read_text())
    exported=m.sampling.selected_actions(m.DEFAULT_PLANNING)
    return report,protocol,prepared,source,exported['actions']


def test_actual_eight_witnesses_identity_join_and_inherited_costs(saved):
    report,protocol,prepared,source,actions=saved
    reordered=dict(report,rows=list(reversed(report['rows'])))
    source=dict(source,rows=list(reversed(source['rows'])))
    plans=m.plans_from_saved(reordered,protocol,prepared,source,actions)
    assert len(plans)==len({(p['root_sha256'],p['method']) for p in plans})==8
    for i in range(4):
        peers=[p for p in plans if p['root_index']==i]
        assert [p['method'] for p in peers]==list(m.sampling.ARMS if i%2==0 else m.sampling.ARMS[::-1])
    for p in plans:
        assert p['commands']==actions[p['root_sha256'],p['method']].tolist()
        assert p['commands']==p['witness']['receipt']['actions']
        assert len(p['commands'])==10
        assert len(p['witness']['edges'])==(1 if p['method']==m.sampling.ARMS[0] else 2)
        assert p['planning_accounting']['shared']['predict_calls']==4
        assert p['planning_accounting']['paired_unique']['predict_calls']==252
        assert p['selection_sha256']==m.digest({k:v for k,v in p.items() if k!='selection_sha256'})


@pytest.mark.parametrize('defect',['root','index','duplicate','missing','goal','q','metric','fixture','arm',
    'budget_root','limit','blocked','entries','vjp','svd_total','total','claim','receipt','export_action','export_missing'])
def test_saved_mutations_reject(saved,defect):
    report,protocol,prepared,source,actions=deepcopy(saved)
    row=report['rows'][0]; result=row['results'][0]
    key=row['root_sha256']
    c=next(r['compiled'] for r in source['rows'] if r['root_sha256']==key)
    if defect=='root': protocol['context']['cases'][0]['root']['context']['latent_history'][0][0]+=1
    if defect=='index': row['root_index']=999
    if defect=='duplicate': report['rows'][-1]=deepcopy(row)
    if defect=='missing': report['rows'].pop()
    if defect=='goal': c['goal_vectors'][0][0]+=1
    if defect=='q': c['scoring']['metric']['projection'][0][0]+=1
    if defect=='metric': c['scoring']['metric']['sha256']='0'*64
    if defect=='fixture': result['fixture_only']=True
    if defect=='arm': result['arm']='cem'
    if defect=='budget_root': row['actual_prediction_budget']['binding']['root_id']='0'*64
    if defect=='limit': row['actual_prediction_budget']['predict_sample_limit']=168
    if defect=='blocked': row['actual_prediction_budget']['blocked_calls']=1
    if defect=='entries': row['actual_model_entries'].pop()
    if defect=='vjp': row['completed_operations']['scalar_output_vjp_directions']-=1
    if defect=='svd_total': report['completed_svd_calls']-=1
    if defect=='total': report['actual_unique_predict_calls']=1024
    if defect=='claim': report['fair_total_budget_comparison']=True
    if defect=='receipt': result['winner']['witness']['receipt']['actions'][0][0]+=.1
    if defect=='export_action': actions[key,m.sampling.ARMS[0]][0,0]+=.01
    if defect=='export_missing': actions.pop((key,m.sampling.ARMS[0]))
    with pytest.raises(ValueError): m.plans_from_saved(report,protocol,prepared,source,actions)


def fake_plan(i=0,arm=None):
    arm=arm or m.sampling.ARMS[0]
    commands=np.linspace(-.8,.8,20).reshape(10,2).tolist()
    p=dict(root_index=i,root_sha256=str(i)*64,method=arm,
        commands=commands,planned_horizon_tokens=2,planned_commands=10,corrected_search_horizon=2,
        witness=dict(receipt=dict(actions=deepcopy(commands),receipt_sha256='f'*64)),receipt_sha256='f'*64)
    p['selection_sha256']=m.digest(p)
    return p


@pytest.mark.parametrize('options,attempts,returns,status',[
    ({},10,10,'completed'),({'missing_at':0},0,0,'observation_unavailable'),
    ({'missing_at':3},3,3,'observation_unavailable'),({'fail_at':4},4,3,'execution_failed'),
    ({'terminate_at':2},2,2,'environment_terminated')])
def test_durable_partial_guard_all_ten_exact_prefix(tmp_path,options,attempts,returns,status):
    plan=fake_plan(arm=m.sampling.ARMS[1])
    row=m.physical_branch(plan=plan,output=tmp_path,reset=lambda:{'fixture':True},env=Env(**options),
        state_of=state,evaluate=lambda s:dict(official={},pose={}))
    assert row['status']==status
    assert row['physical_env_step_attempts']==attempts and row['physical_env_step_returns']==returns
    assert [e['command'] for e in row['trace']]==plan['commands'][:returns]
    events=[json.loads(v) for v in Path(row['journal_path']).read_text().splitlines()]
    assert sum(e['kind']=='attempt' for e in events)==attempts
    assert sum(e['kind']=='step_returned' for e in events)==returns
    assert json.loads(next(tmp_path.glob('*.physical.json')).read_text())==row


def test_return_then_render_failure_retains_nonrefunded_partial(tmp_path):
    class RenderFail(Env):
        def render(self):
            if self.n==4: raise RuntimeError('fixture render failure after returned step')
            return super().render()
    row=m.physical_branch(plan=fake_plan(),output=tmp_path,reset=lambda:{},env=RenderFail(),
        state_of=state,evaluate=lambda s:dict(official={},pose={}))
    assert row['physical_env_step_attempts']==row['physical_env_step_returns']==4
    assert row['completed_commands']==4 and len(row['trace'])==3
    assert not row['full_horizon_completed']
    assert next(tmp_path.glob('*.physical.json')).is_file()


@pytest.mark.parametrize('defect',['method','digest','witness','nan','short','bool'])
def test_illegal_or_unbound_plan_before_environment(tmp_path,defect):
    p=fake_plan()
    if defect=='method': p['method']='cem'
    if defect=='digest': p['commands'][0][0]+=.01
    if defect=='witness': p['witness']['receipt']['actions'][0][0]+=.01
    if defect=='nan': p['commands'][0][0]=float('nan')
    if defect=='short': p['commands'].pop()
    if defect=='bool': p['commands']=[[False,False]]*10
    if defect not in ('digest','nan'):
        p['selection_sha256']=m.digest({k:v for k,v in p.items() if k!='selection_sha256'})
    with pytest.raises(ValueError):
        m.physical_branch(plan=p,output=tmp_path,reset=lambda:pytest.fail('reset'),env=None,
            state_of=lambda e:pytest.fail('state'),evaluate=lambda s:pytest.fail('evaluate'))


def test_reset_failure_materializes_branch_without_dispatch(tmp_path):
    def bad_reset(): raise ValueError('fixture strict RGB/goal mismatch')
    row=m.physical_branch(plan=fake_plan(),output=tmp_path,reset=bad_reset,env=None,
        state_of=lambda e:pytest.fail('state'),evaluate=lambda s:{})
    assert row['status']=='branch_failed' and row['physical_env_step_attempts']==0
    assert row['terminal_rgb_path'] is None
    assert next(tmp_path.glob('*.physical.json')).is_file()


@pytest.mark.parametrize('defect',['nan_goal','goal_shape','goal_dtype','nan_state','changed_goal','duplicate'])
def test_all_roots_and_strict_goals_upfront(defect):
    roots=[dict(state=np.arange(7,dtype=float)+i,goal=np.arange(7,dtype=float)+10+i) for i in range(4)]
    refs={root_fingerprint(r):{} for r in roots}
    assert len(m.validate_roots(roots,refs,root_fingerprint))==4
    if defect=='nan_goal': roots[-1]['goal'][0]=np.nan
    if defect=='goal_shape': roots[-1]['goal']=np.zeros((1,7))
    if defect=='goal_dtype': roots[-1]['goal']=[True]*7
    if defect=='nan_state': roots[-1]['state'][0]=np.inf
    if defect=='changed_goal': roots[-1]['goal'][0]+=1
    if defect=='duplicate': roots[-1]=deepcopy(roots[0])
    with pytest.raises(ValueError): m.validate_roots(roots,refs,root_fingerprint)


def test_real_warmup_raw3x2_to_normalized_between2x10(saved):
    normalization=saved[1]['context']['normalization']
    expected=saved[1]['context']['cases'][0]['root']['context']['action_history']
    result=m.validate_reset_history(np.zeros((3,2),np.float32),normalization,expected)
    assert result['normalized_between_history']==expected
    with pytest.raises(ValueError): m.validate_reset_history(np.zeros((3,10)),normalization,expected)
    with pytest.raises(ValueError): m.validate_reset_history(np.ones((3,2))*.1,normalization,expected)


@pytest.mark.parametrize('missing,expected',[(None,1),(3,0)])
def test_encoding_only_complete_h2_and_after_durable_physics(tmp_path,missing,expected):
    row=m.physical_branch(plan=fake_plan(),output=tmp_path,reset=lambda:{},env=Env(missing_at=missing),
        state_of=state,evaluate=lambda s:dict(official={},pose={}))
    calls=[]
    def setup():
        assert (tmp_path/'physical_report.json').exists()
        calls.append('setup')
        return lambda p: calls.append('encode') or np.zeros(192).tolist()
    with pytest.raises(ValueError,match='persist'): m.optional_encodings([row],tmp_path,setup)
    m.write(tmp_path/'physical_report.json',dict(rows=[row]))
    receipts=m.optional_encodings([row],tmp_path,setup)
    assert calls==(['setup','encode'] if expected else [])
    assert receipts[0]['attempted_encode_samples']==expected
    assert receipts[0]['status']==('completed' if expected else 'blocked_fixed_horizon')


def test_encoding_setup_failure_keeps_all_eight_and_physical_bytes(tmp_path):
    plans=[fake_plan(i,arm) for i in range(4) for arm in m.sampling.ARMS]
    rows=[]
    m.helpers.materialize_missing(plans,rows,tmp_path)
    assert len(rows)==len(list(tmp_path.glob('*.physical.json')))==8
    # One fake complete endpoint is sufficient to exercise setup-failure receipts.
    rows[0].update(full_horizon_completed=True,completed_commands=10)
    m.write(tmp_path/'physical_report.json',dict(rows=rows))
    before={p:p.read_bytes() for p in tmp_path.glob('*.physical.json')}
    def fail(): raise RuntimeError('fixture CUDA setup failure')
    receipts=m.optional_encodings(rows,tmp_path,fail)
    assert len(receipts)==8 and all(r['status']=='encoding_setup_failed' for r in receipts)
    assert all(p.read_bytes()==value for p,value in before.items())


def test_original_exporter_archive_and_strict_report_prerequisite(saved,tmp_path,monkeypatch):
    proof=m.verify_archived_exporter(saved[2])
    assert proof['sha256']==m.EXPECTED_PLANNER_SHA256
    monkeypatch.setattr(m.sampling,'load_prepared',lambda *a:pytest.fail('load before report exists'))
    output=tmp_path/m.TAG
    with pytest.raises(ValueError,match='CUDA report missing'): m.prepare(tmp_path/'missing',output)
    assert not output.exists()


def test_model_free_prepare_and_source_tamper(saved,tmp_path,monkeypatch):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    for name in ('require_cuda_allocation','load_compatible_pretrained','collect_nonterminal_boundary_contact_roots'):
        monkeypatch.setattr(base,name,lambda *a,**k:pytest.fail('execution during prepare'))
    monkeypatch.setattr(base.gym,'make',lambda *a,**k:pytest.fail('simulator instantiated'))
    output=tmp_path/m.TAG
    p=m.prepare(m.DEFAULT_PLANNING,output)
    assert p['branches']==8 and p['total_planned_commands']==80
    assert p['planning_totals']['actual_unique_predict_calls']==1008
    assert p['exporter_verification_work']['neural_predict_calls']==0
    assert p['claim_boundary']['P2_admission'] is False
    s=json.loads((output/'prepared.json').read_text())
    assert s['new_neural_calls']==s['new_physical_steps']==0
    assert not (output/'physical_report.json').exists()
    assert p['environment_source_paths']
    with pytest.raises(FileExistsError): m.prepare(m.DEFAULT_PLANNING,output)
    archive=Path(s['source_seals'][0]['output_dir'])/'sources/000.source'
    with archive.open('a') as stream: stream.write('\n# pytest-only corruption\n')
    with pytest.raises(ValueError): m.load_prepared(output)


def test_static_execution_order_no_old_runner_or_neural_planning():
    source=inspect.getsource(m.run)
    assert source.index('load_prepared(prepared)')<source.index('base.gym.make')
    assert source.index('validate_roots(')<source.index('for plan in plans:',source.index('validate_roots('))
    assert source.index("output/'physical_report.json'")<source.index('base.require_cuda_allocation')
    assert 'model.predict = forbid' in source
    assert '.cuda(' not in source and 'sampling.cuda' not in source
    assert 'helpers.run(' not in source
    assert 'encode_terminals=False' in source
