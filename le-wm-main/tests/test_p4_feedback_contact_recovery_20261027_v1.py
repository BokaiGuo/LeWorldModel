"""Pure saved-data fixtures/arithmetic and actual saved-report regression."""
from copy import deepcopy
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p4_feedback_contact_recovery_20261027_v1 as m


class Memory:
    def __init__(self):
        self.files = {}
    def raw(self,path):
        return self.files[str(path)]
    def hash(self,path):
        return hashlib.sha256(self.raw(path)).hexdigest()
    def json(self,path):
        return m.saved.strict_json(self.raw(path))


@pytest.fixture
def case():
    t = Memory()
    image = np.zeros((224,224,3),dtype=np.uint8)
    buf = io.BytesIO(); Image.fromarray(image).save(buf,format='PNG')
    t.files['expected.png'] = buf.getvalue()
    commands = [[0.,0.]]*10
    initial = [200.,170.,200.,200.,0.,0.,0.]
    goal = [200.,185.,200.,200.,0.,0.,0.]
    original = dict(context_identity=dict(state=initial,goal=goal,history=[],history_rgb_sha256='history',goal_rgb_sha256='goal'),
        trace=[dict(state=initial,command=commands[j]) for j in range(5)],final_state=initial)
    plan = dict(root_sha256=m.physical.ROOT,root_index=1,role='selected',arm='actual',method='selected_actual',
        selection_sha256='selection',commands=commands,selected_candidate_id='suffix_6',expected_prefix_rgb='expected.png',
        expected_prefix_rgb_sha256=t.hash('expected.png'))
    events = []
    trace = []
    def obs(step,state):
        path = f'image{step}.png'; t.files[path] = buf.getvalue()
        event = dict(kind='initial' if step == 0 else 'observation',state=state,
            rgb_path=path,rgb_file_sha256=t.hash(path))
        if step:
            event.update(primitive_step=step,command=commands[step-1],reward=0.,terminated=False,truncated=False)
            trace.append({k:v for k,v in event.items() if k not in ('rgb_path','rgb_file_sha256')})
        events.append(event)
    obs(0,initial)
    gate = dict(status='matched',primitive_step=5,expected_pixel_sha256=m.saved.rgb_sha(image),
        observed_pixel_sha256=m.saved.rgb_sha(image),pixels_exact=True)
    for step in range(1,11):
        events.extend([dict(kind='attempt',primitive_step=step,command=commands[step-1]),dict(kind='step_returned',primitive_step=step)])
        s = initial.copy()
        if step in (6,7,9,10): s[1] = 185.
        obs(step,s)
        if step == 5: events.append(dict(kind='prefix_identity',**gate))
    row = {k:plan[k] for k in ('root_sha256','root_index','role','arm','method','selection_sha256')}
    row.update(completed_commands=10,attempted_commands=10,physical_env_step_attempts=10,physical_env_step_returns=10,
        status='completed',full_horizon_completed=True,journal_path='journal.jsonl',trace=trace,
        final_state=trace[-1]['state'],context_identity={k:v for k,v in original['context_identity'].items() if k != 'state'},
        prefix_identity=gate,pose=m.pose(trace[-1]['state'],goal),terminal_rgb_path='image10.png',terminal_rgb_sha256=t.hash('image10.png'))
    def update():
        t.files['journal.jsonl'] = b'\n'.join(json.dumps(e,sort_keys=True).encode() for e in events)
        row['journal_sha256'] = t.hash('journal.jsonl')
    update()
    return t,row,plan,original,events,update


@pytest.fixture
def geometry():
    return m.geometry.contract(m.saved.Inputs())


def test_fixture_reentry_not_force_causality(case,geometry):
    t,row,plan,original,_,_ = case
    r = m.analyze_branch(t,row,plan,original,geometry)
    s = r['summary']
    assert s['prefix_near_contact'] is False
    assert s['suffix_near_contact_entry_steps'] == [6,9]
    assert s['suffix_geometric_reentry_observed'] is True
    assert s['suffix_near_contact_steps'] == 4
    assert s['exact_terminal_minus_prefix_pose'] == 0
    assert s['force_contact'] is s['causal_push'] is None
    assert len(r['steps']) == 11


@pytest.mark.parametrize('key',['root_sha256','root_index','role','arm','method','selection_sha256'])
def test_identity_corruption_rejected(case,geometry,key):
    t,row,plan,original,_,_ = case
    row[key] = 'changed'
    with pytest.raises(ValueError,match='identity'):
        m.analyze_branch(t,row,plan,original,geometry)


@pytest.mark.parametrize('mutation',['command','initial','prefix_state','goal','history','prefix_hash','gate_order',
    'dispatch_order','pose','final_state','journal_hash','rgb_hash','count','trace','missing_step','full'])
def test_receipt_corruption_rejected(case,geometry,mutation):
    t,row,plan,original,events,update = case
    if mutation == 'command': next(e for e in events if e['kind']=='attempt')['command'] = [.2,0]
    elif mutation == 'initial': events[0]['state'] = [0.]*7
    elif mutation == 'prefix_state': next(e for e in events if e['kind']=='observation')['state'] = [0.]*7
    elif mutation == 'goal': row['context_identity']['goal'] = [0.]*7
    elif mutation == 'history': row['context_identity']['history'] = [[1,2]]
    elif mutation == 'prefix_hash': plan['expected_prefix_rgb_sha256'] = 'bad'
    elif mutation == 'gate_order':
        gate = next(e for e in events if e['kind']=='prefix_identity'); events.remove(gate); events.append(gate)
    elif mutation == 'dispatch_order': events[1],events[2] = events[2],events[1]
    elif mutation == 'pose': row['pose']['pose_cost'] += 1
    elif mutation == 'final_state': row['final_state'] = [0.]*7
    elif mutation == 'rgb_hash': events[-1]['rgb_file_sha256'] = 'bad'
    elif mutation == 'count': row['completed_commands'] = 9
    elif mutation == 'trace': row['trace'][0] = dict(row['trace'][0],reward=10)
    elif mutation == 'missing_step': events.remove(next(e for e in events if e['kind']=='observation'))
    elif mutation == 'full': row['full_horizon_completed'] = False
    update()
    if mutation == 'journal_hash': row['journal_sha256'] = 'bad'
    with pytest.raises(ValueError):
        m.analyze_branch(t,row,plan,original,geometry)


def test_partial_endpoint_not_substituted(case,geometry):
    t,row,plan,original,events,update = case
    events[:] = events[:13]  # initial + four complete attempts/returns/observations
    row.update(completed_commands=4,attempted_commands=4,physical_env_step_attempts=4,physical_env_step_returns=4,
        status='environment_terminated',full_horizon_completed=False,trace=row['trace'][:4],
        final_state=row['trace'][3]['state'],terminal_rgb_path='image4.png',terminal_rgb_sha256=t.hash('image4.png'))
    row['pose'] = m.pose(row['final_state'],original['context_identity']['goal'])
    update()
    r = m.analyze_branch(t,row,plan,original,geometry)
    assert r['summary']['pose_at_exact_terminal'] is None
    assert r['summary']['suffix_geometric_reentry_observed'] is None
    assert r['summary']['suffix_observed_steps'] == 0


def test_continued_near_contact_is_not_reentry(geometry):
    steps = []
    previous = None
    for step in range(11):
        state = [200.,185.,200.,200.,0.,0.,0.]
        item = m.geometry.label(state,previous,geometry)
        item.update(primitive_step=step,pose=m.pose(state,state))
        steps.append(item); previous = state
    summary = m.summarize_steps(steps)
    assert summary['prefix_near_contact'] is True
    assert summary['suffix_near_contact_steps'] == 5
    assert summary['suffix_geometric_reentry_observed'] is False


def test_original_pose_formula_and_angle_wrap():
    goal = [0,0,3,4,0,0,0]
    state = [9,8,6,8,2*np.pi-.2,0,0]
    assert m.pose(state,goal)['pose_cost'] == pytest.approx(9)


@pytest.mark.parametrize('mutation',[None,'root','early_endpoint','image','mse','projected','cost','samples','duplicate','missing'])
def test_saved_encoding_arithmetic_and_corruption(case,mutation):
    t,row,plan,_,_,_ = case
    plan.update(q=np.eye(192).tolist(),goal=[0.]*192,
        saved_candidate=dict(predicted_prefix=[0.]*192,predicted_terminal=[0.]*192))
    encodings = []
    folder = Path('/saved_fixture')
    for kind,step in (('prefix',5),('terminal',10)):
        path = folder/f'root_1_{plan["method"]}_replay0_rgb_{step}.png'
        t.files[str(path)] = t.raw(f'image{step}.png')
        encodings.append(dict(method=plan['method'],kind=kind,root_sha256=m.physical.ROOT,status='completed',
            attempted_encode_samples=1,comparable=True,vector=[1.]*192,prediction_mse=1.,
            projected_prediction_mse=1.,observed_Q_goal_cost=1.,rgb_path=str(path),rgb_sha256=t.hash(path)))
    e = encodings[1]
    if mutation == 'root': e['root_sha256'] = 'wrong'
    elif mutation == 'early_endpoint': row['completed_commands'] = 9
    elif mutation == 'image': e['rgb_sha256'] = 'wrong'
    elif mutation == 'mse': e['prediction_mse'] = .5
    elif mutation == 'projected': e['projected_prediction_mse'] = .5
    elif mutation == 'cost': e['observed_Q_goal_cost'] = .5
    elif mutation == 'samples': e['attempted_encode_samples'] = 2
    for e in encodings:
        t.files[str(folder/f'root_1_{plan["method"]}_{e["kind"]}.encoding.json')] = json.dumps(e).encode()
    if mutation == 'duplicate': encodings.append(deepcopy(encodings[-1]))
    elif mutation == 'missing': encodings.pop()
    if mutation is None:
        result = m.encoding_costs(t,folder,encodings,row,plan)
        assert result['terminal']['predicted_Q_goal_cost'] == 0
        assert result['terminal']['observed_Q_goal_cost'] == 1
    else:
        with pytest.raises(ValueError): m.encoding_costs(t,folder,encodings,row,plan)


def test_missing_report_creates_no_output(tmp_path):
    output = tmp_path/'anything'
    result = m.run(tmp_path/'missing.json',output)
    assert result['status'] == 'awaiting_physical_report'
    assert not output.exists()


def test_exclusive_output(tmp_path):
    report = tmp_path/'report.json'; report.write_text('{}')
    output = tmp_path/(m.TAG+'_existing'); output.mkdir()
    with pytest.raises(FileExistsError): m.run(report,output)


def test_failclosed_error_retained(tmp_path,monkeypatch):
    report = tmp_path/'report.json'; report.write_text('{}')
    def broken(_): raise ValueError('bad binding fixture')
    monkeypatch.setattr(m,'audit',broken)
    output = tmp_path/(m.TAG+'_error')
    r = m.run(report,output)
    assert r['status'] == 'integrity_fail'
    assert r['new_neural_calls'] == r['new_physics_steps'] == 0
    assert r['P1_admission'] is r['P4_admission'] is False
    assert (output/'report.json').is_file()


def test_import_does_not_load_neural_or_simulator():
    code = ('import sys; from experiments.spectral_reachability import p4_feedback_contact_recovery_20261027_v1; '
            'assert not any(x in sys.modules for x in ("torch","gymnasium","stable_worldmodel"))')
    subprocess.run([sys.executable,'-B','-c',code],check=True,cwd=m.REPO)
    tree = ast.parse(Path(m.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in ('step','reset','predict','make','load_compatible_pretrained','backward')


def test_real_saved_report_all_slots():
    r = m.audit(m.DEFAULT_REPORT)
    assert r['status'] == 'saved_contact_recovery_recorded'
    assert len(r['rows']) == 6
    assert {(v['role'],v['arm']) for v in r['rows']} == {(role,arm) for role in m.physical.ROLES for arm in m.physical.ARMS}
    assert all(len(v['steps']) == 11 and v['physical_complete'] for v in r['rows'])
    assert r['all_inputs_unchanged']
    assert all(v['summary']['force_contact'] is v['summary']['causal_push'] is None for v in r['rows'])


def test_same_algorithm_independent_of_task_goal(geometry):
    state = [200.,185.,200.,200.,0.,0.,0.]
    label = m.geometry.label(state,None,geometry)
    p1 = m.pose(state,[0,0,200,200,0,0,0])
    p2 = m.pose(state,[0,0,400,400,3,0,0])
    assert p1 != p2
    assert label == m.geometry.label(state,None,geometry)
