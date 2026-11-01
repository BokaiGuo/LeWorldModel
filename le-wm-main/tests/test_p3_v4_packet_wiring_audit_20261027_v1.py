"""Non-neural contract/tamper fixtures; no simulator or outcome fixtures."""
from copy import deepcopy
import ast
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p3_v4_packet_wiring_audit_20261027_v1 as a


def fixture():
    initial = dict(layout='states_with_between_state_actions_v1', context_length=3,
                   latent_dim=192, action_dim=10, latent_history=np.arange(576).reshape(3, 192).tolist(),
                   action_history=np.tile(-a.MEAN/a.STD, (2, 5)).tolist())
    commands = (np.arange(20).reshape(10, 2)/30-.3).tolist()
    context = dict(initial, latent_history=[initial['latent_history'][-1], [0.]*192, [1.]*192],
                   action_history=a.tokens(commands)[-2:].tolist())
    plan = dict(root_index=0, root_sha256='root', method='uniform', selection_sha256='selection',
                planned_horizon_tokens=2, planned_commands=10, commands=commands,
                goal_vector_sha256='goal', projection_sha256='Q')
    identity = dict(raw_history=np.zeros((3, 2)).tolist(), normalized_between_history=initial['action_history'], state=[0.]*7)
    events = [dict(kind='initial', state=identity['state'], rgb_path='initial.png', rgb_file_sha256='initial')]
    trace = []
    for i, command in enumerate(commands, 1):
        observed = dict(kind='observation', primitive_step=i, command=command, state=[float(i)]*7,
                        terminated=False, truncated=False, reward=0.)
        trace.append(observed)
        events.extend([dict(kind='attempt', primitive_step=i, command=command),
                       dict(kind='step_returned', primitive_step=i),
                       dict(observed, rgb_path=f'{i}.png', rgb_file_sha256=str(i))])
    row = dict(plan, status='completed', full_horizon_completed=True, endpoint_fresh=True,
               endpoint_scope='full_H2', physical_env_step_attempts=10, physical_env_step_returns=10,
               completed_commands=10, attempted_commands=10, actual_endpoint_primitive_steps=10,
               context_identity=identity, trace=trace, final_state=trace[-1]['state'],
               terminal_rgb_path='10.png', terminal_rgb_sha256='10',
               guard=dict(root_id='root', abstained=False, primitive_commands_authorized=10,
                          observations=[dict(primitive_step=i) for i in range(11)]))
    enc = dict(plan, status='completed', attempted_encode_samples=1, endpoint_scope='full_H2',
               terminal_rgb_sha256='10', terminal_vector=[0.]*192)
    return initial, commands, context, plan, row, events, enc


def test_good_contract_not_efficacy():
    initial, commands, context, plan, row, events, enc = fixture()
    a.context_witness(initial, commands, context)
    a.warmup_witness(row['context_identity'], initial)
    a.branch_witness(plan, row, events, enc)


@pytest.mark.parametrize('mutation', ['swap_xy', 'reverse_time', 'reverse_packets', 'old_latent', 'old_token', 'scaled_twice'])
def test_context_tamper_rejected(mutation):
    initial, commands, context, *_ = fixture()
    if mutation == 'swap_xy':
        commands = np.asarray(commands)[:, ::-1].tolist()
    elif mutation == 'reverse_time':
        commands = commands[::-1]
    elif mutation == 'reverse_packets':
        commands = commands[5:]+commands[:5]
    elif mutation == 'old_latent':
        context['latent_history'][0] = initial['latent_history'][0]
    elif mutation == 'old_token':
        context['action_history'][0] = initial['action_history'][0]
    else:
        commands = (np.asarray(commands)/100).tolist()
    with pytest.raises(ValueError):
        a.context_witness(initial, commands, context)


@pytest.mark.parametrize('mutation', ['root', 'selection', 'goal', 'Q', 'endpoint', 'horizon', 'event_time', 'event_order', 'command', 'state', 'guard', 'termination'])
def test_branch_tamper_rejected(mutation):
    _, _, _, plan, row, events, enc = deepcopy(fixture())
    if mutation in ('root', 'selection', 'goal', 'Q'):
        field = dict(root='root_sha256', selection='selection_sha256', goal='goal_vector_sha256', Q='projection_sha256')[mutation]
        enc[field] = 'wrong'
    elif mutation == 'endpoint':
        enc['terminal_rgb_sha256'] = 'initial'
    elif mutation == 'horizon':
        row['planned_horizon_tokens'] = 3
    elif mutation == 'event_time':
        events[1]['primitive_step'] = 2
    elif mutation == 'event_order':
        events[1], events[2] = events[2], events[1]
    elif mutation == 'command':
        events[1]['command'] = [1., 1.]
    elif mutation == 'state':
        row['final_state'] = [999.]*7
    elif mutation == 'guard':
        row['guard']['root_id'] = 'other'
    else:
        events[3]['terminated'] = True
    with pytest.raises(ValueError):
        a.branch_witness(plan, row, events, enc)


def test_warmup_drop_first_not_last():
    initial, _, _, _, row, *_ = fixture()
    row['context_identity']['raw_history'][1][0] = .1
    with pytest.raises(ValueError):
        a.warmup_witness(row['context_identity'], initial)


def test_duplicate_identity_rejected():
    with pytest.raises(ValueError):
        a.index([dict(root=1), dict(root=1)], ('root',))


@pytest.mark.parametrize('bad', [np.zeros((5, 2)), np.zeros((10, 3)), np.full((10, 2), np.nan), np.ones((10, 2))*1.1])
def test_bad_packets_rejected(bad):
    with pytest.raises(ValueError):
        a.tokens(bad)


def test_file_change_rejected(tmp_path):
    path = tmp_path/'input.json'
    path.write_text('{}')
    inputs = a.Inputs()
    inputs.bind(path)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError):
        inputs.unchanged()


def test_duplicate_json_rejected(tmp_path):
    path = tmp_path/'input.json'
    path.write_text('{"root":1,"root":2}')
    with pytest.raises(ValueError):
        a.read(path)


def test_source_fragment_is_fail_closed(tmp_path):
    path = tmp_path/'source.py'
    path.write_text('def step(action):\n    return action\n')
    a.source_witness(path, 'step', ['return action'])
    with pytest.raises(ValueError):
        a.source_witness(path, 'step', ['return action*100'])


def test_audit_has_no_neural_simulator_import_or_metric_recompute():
    tree = ast.parse(a.Path(a.__file__).read_text())
    allowed = {'__future__', 'argparse', 'ast', 'hashlib', 'json', 'pathlib', 'numpy'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name in allowed for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module in allowed
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {'predict', 'step', 'reset', 'render', 'svd', 'mean', 'argmin'}
            if node.func.attr == 'encode':
                # UTF-8 strings for SHA only; neural encode requires an input.
                assert not node.args and not node.keywords


def test_frozen_main01_integration_readonly():
    """Actual saved arrays only; no physical benefit assertion or new inference."""
    if not (a.PROJECT/a.PHYSICAL/'report.json').exists():
        pytest.skip('actual sealed main01 artifacts not installed')
    report = a.read(a.PROJECT/a.PHYSICAL/'report.json')
    execution = a.read(a.PROJECT/a.PHYSICAL/'execution_protocol.json')
    plans = a.read(a.Path(execution['prepared'])/'protocol.json')['plans']
    assert len(plans) == len(report['rows']) == 8
    enc = a.index(report['encodings'], ('root_sha256', 'method'))
    for plan, row in zip(plans, report['rows'], strict=True):
        events = [json.loads(line) for line in a.Path(row['journal_path']).read_text().splitlines()]
        a.context_witness(plan['predictor_context'], plan['commands'], plan['selected_predicted_context'])
        a.warmup_witness(row['context_identity'], plan['predictor_context'])
        a.branch_witness(plan, row, events, enc[plan['root_sha256'], plan['method']])
