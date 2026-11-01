"""Synthetic saved-byte graphs in RAM only; no real CUDA/model/physics evidence."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import random
import struct

import pytest

from experiments.spectral_reachability import moving_p2_saved_program_graph_20261027_v1 as q


def vec(value=0, **kwargs):
    return dict(dict.fromkeys(q.AXES, value), **kwargs)


def fixture(specs=None, updates=None, base=None):
    """Specs: (edge_id, parent, child, nonnegative_cost, raw_commands, resources)."""
    memory = {}

    def put(name, data):
        raw = data if type(data) is bytes else q.encode(data)
        memory[name] = raw
        return dict(file=name, sha256=q.sha(raw))

    c = dict(assets={k: put(k, ('SYNTHETIC ' + k).encode()) for k in q.ASSETS},
             action_dtype='float64-le-raw', action_dim=1, lower=[-1.], upper=[1.],
             max_steps=20, max_l1=10., max_l2_squared=10., min_programs=1, max_programs=8,
             resource_caps=vec(100), prohibited=dict.fromkeys(q.POLICY, True),
             objective='sum_saved_nonnegative_cost')
    c.update(updates or {})
    if specs is None:
        specs = [('sa', 'S', 'A', 1, [.25], vec(1)),
                 ('at', 'A', 'T', 2, [-.5], vec(2)),
                 ('st', 'S', 'T', 7, [.75], vec(3))]
    ids = sorted({'S', 'T'} | {s[i] for s in specs for i in (1, 2)})
    nodes = [dict(node_id=n, state_kind='saved_predicted',
                  latent=put(n + '_latent', struct.pack('<d', float(i))),
                  state=put(n + '_state', ('SYNTHETIC state ' + n).encode()),
                  context=put(n + '_context', ('SYNTHETIC predictor context ' + n).encode()))
             for i, n in enumerate(ids)]
    by_id = {n['node_id']: n for n in nodes}
    edges = []
    contract_sha = q.sha(q.encode(c))
    for name, parent, child, cost, actions, resources in specs:
        fmt = 'f' if c['action_dtype'] == 'float32-le-raw' else 'd'
        commands = put(name + '_commands', struct.pack('<' + fmt * len(actions), *actions))
        e = dict(edge_id=name, program_type='reconfiguration' if parent == 'S' else 'progress',
                 parent=q.endpoint(by_id[parent], contract_sha), child=q.endpoint(by_id[child], contract_sha),
                 contract_sha256=contract_sha, steps=len(actions) // c['action_dim'],
                 commands=commands, resource_delta=resources, cost=cost)
        receipt = dict(e, schema=q.STEM + '.saved_transition', backend='HOST_CUDA',
                       status='completed', producer_source_sha256=q.sha(b'SYNTHETIC SOURCE ONLY'),
                       invocation=put(name + '_invocation', b'SYNTHETIC receipt, NOT REAL EVIDENCE'))
        receipt.pop('commands'); receipt['commands_sha256'] = commands['sha256']
        e['receipt'] = put(name + '_receipt', receipt)
        edges.append(e)
    graph = dict(schema=q.STEM + '.graph', contract=c, base_resources=vec(1) if base is None else base,
                 nodes=nodes, edges=edges, start_node='S', target_nodes=['T'])
    return graph, memory


def run(data, algorithm='shortest', **kwargs):
    return q.search(data[0], data[1].__getitem__, algorithm=algorithm,
                    max_expansions=kwargs.pop('max_expansions', 10000), **kwargs)


def replace_ref(data, ref, change):
    obj = q.parse(data[1][ref['file']]); change(obj)
    data[1][ref['file']] = q.encode(obj)
    ref['sha256'] = q.sha(data[1][ref['file']])


def test_shortest_returns_complete_original_witness_and_never_execution():
    data = fixture(); before = deepcopy(data)
    r = run(data)
    assert data == before
    assert r['status'] == 'saved_path_found' and r['optimality_proven']
    p = r['path']
    assert p['edge_ids'] == ['sa', 'at'] and p['node_ids'] == ['S', 'A', 'T']
    assert p['commands'] == [[.25], [-.5]] and p['cost'] == dict(numerator=3, denominator=1)
    assert p['resources'] == vec(4)  # Entire saved graph setup charged before path deltas.
    assert len(p['receipt_refs']) == len(p['witness_refs']) == 2
    assert p['saved_edges'][0]['child'] == p['saved_edges'][1]['parent'] == p['endpoints'][1]
    assert not r['execution_authorized'] and not r['P2_admission']
    assert r['physical_outcome'] is None and not r['evidence_authenticity_independently_proven']
    assert any('P1' in x for x in r['execution_blockers'])
    p['saved_edges'][0]['cost'] = 999
    assert data == before


def test_beam_returns_saved_edges_without_optimality_claim():
    r = run(fixture(), 'beam', beam_width=1)
    assert r['status'] == 'saved_path_found' and r['path']['edge_ids'] == ['sa', 'at']
    assert not r['optimality_proven'] and r['beam_pruned_labels'] == 1


def test_resource_constrained_shortest_keeps_more_than_cheapest_label_per_node():
    specs = [('sa', 'S', 'A', 0, [.25], vec(predict_samples=5)),
             ('sx', 'S', 'X', 1, [.25], vec(predict_samples=1)),
             ('xa', 'X', 'A', 0, [.25], vec(predict_samples=1)),
             ('at', 'A', 'T', 0, [.25], vec(predict_samples=3))]
    data = fixture(specs, updates={'resource_caps': vec(100, predict_samples=5)}, base=vec())
    r = run(data)
    assert r['path']['edge_ids'] == ['sx', 'xa', 'at']
    assert r['path']['resources']['predict_samples'] == 5
    assert any('predict_samples' in reason for reason in r['rejection_counts'])


def test_beam_failure_after_pruning_is_not_unreachability():
    specs = [('sa', 'S', 'A', 0, [.25], vec(predict_samples=5)),
             ('sx', 'S', 'X', 1, [.25], vec(predict_samples=1)),
             ('xt', 'X', 'T', 0, [.25], vec(predict_samples=1))]
    data = fixture(specs)
    assert run(data)['status'] == 'saved_path_found'
    r = run(data, 'beam', beam_width=1)
    assert r['status'] == 'beam_exhausted_no_path' and not r['optimality_proven']
    assert r['path'] is None and r['beam_pruned_labels'] > 0


@pytest.mark.parametrize('field', ['latent_sha256', 'state_sha256', 'context_sha256', 'contract_sha256'])
def test_each_endpoint_identity_must_close_exactly(field):
    data = fixture(); data[0]['edges'][1]['parent'][field] = q.sha(b'near but NOT SAME')
    r = run(data)
    assert r['status'] == 'invalid_saved_graph' and r['path'] is None


def test_no_latent_nearest_neighbor_bridge_or_reversed_edge():
    specs = [('sa', 'S', 'A', 1, [.25], vec()), ('bt', 'B', 'T', 1, [.25], vec())]
    data = fixture(specs)
    # Identical latent bytes with different state/context remain distinct endpoints.
    a = next(n for n in data[0]['nodes'] if n['node_id'] == 'A')
    b = next(n for n in data[0]['nodes'] if n['node_id'] == 'B')
    b['latent'] = deepcopy(a['latent'])
    edge = data[0]['edges'][1]
    edge['parent'] = q.endpoint(b, edge['contract_sha256'])
    replace_ref(data, edge['receipt'], lambda r: r.update(parent=edge['parent']))
    r = run(data)
    assert r['status'] == 'no_feasible_saved_path' and r['path'] is None
    assert 'no_saved_outgoing_edge' in r['rejection_counts']
    assert run(fixture([('ts', 'T', 'S', 0, [.25], vec())]))['status'] == 'no_feasible_saved_path'


@pytest.mark.parametrize('updates,reason', [
    ({'max_steps': 1}, 'cumulative_steps_exceeded'),
    ({'max_l1': .3}, 'cumulative_action_cap_exceeded'),
    ({'max_l2_squared': .1}, 'cumulative_action_cap_exceeded'),
    ({'max_programs': 1}, 'max_programs_reached'),
])
def test_cumulative_constraints_not_edge_local(updates, reason):
    specs = [('sa', 'S', 'A', 1, [.25], vec()), ('at', 'A', 'T', 1, [.25], vec())]
    r = run(fixture(specs, updates=updates))
    assert r['status'] == 'no_feasible_saved_path' and reason in r['rejection_counts']


def test_exact_binary_action_cap_has_no_epsilon_or_clipping():
    specs = [('sa', 'S', 'A', 0, [.1], vec()), ('at', 'A', 'T', 0, [.2], vec())]
    assert Fraction(.1) + Fraction(.2) > Fraction(.3)
    r = run(fixture(specs, updates={'max_l1': .3}))
    assert r['status'] == 'no_feasible_saved_path'


@pytest.mark.parametrize('dtype,fmt', [('float32-le-raw', 'f'), ('float64-le-raw', 'd')])
def test_raw_action_bits_order_and_signed_zero_preserved(dtype, fmt):
    specs = [('st', 'S', 'T', 1, [.25, -.5, -0., 0.], vec())]
    r = run(fixture(specs, updates={'action_dtype': dtype, 'action_dim': 2,
                                  'lower': [-1., -1.], 'upper': [1., 1.]}))
    assert r['status'] == 'saved_path_found'
    values = [v for row in r['path']['commands'] for v in row]
    assert struct.pack('<' + fmt * len(values), *values) == struct.pack('<' + fmt * 4, .25, -.5, -0., 0.)
    assert r['path']['steps'] == 2


def test_saved_goal_requires_explicit_target_and_minimum_programs():
    data = fixture(updates={'min_programs': 2})
    assert run(data)['path']['edge_ids'] == ['sa', 'at']
    data[0]['target_nodes'] = ['S']
    assert run(data)['status'] == 'no_feasible_saved_path'  # No zero-edge fake composition.


def test_cycles_zero_cost_and_depth_are_bounded():
    specs = [('sa', 'S', 'A', 0, [.25], vec()), ('as', 'A', 'S', 0, [.25], vec()),
             ('aa', 'A', 'A', 0, [.25], vec()), ('at', 'A', 'T', 0, [.25], vec())]
    r = run(fixture(specs))
    assert r['path']['edge_ids'] == ['sa', 'at'] and r['optimality_proven']
    assert r['rejection_counts']['cycle_not_in_simple_path_scope'] == 2


@pytest.mark.parametrize('algorithm,width', [('shortest', None), ('beam', 1)])
def test_search_cap_is_incomplete_not_no_path(algorithm, width):
    r = run(fixture(), algorithm, max_expansions=1, beam_width=width)
    assert r['status'] == 'search_limit' and not r['optimality_proven']
    assert r['path'] is None


def test_beam_limit_preserves_already_found_saved_incumbent_without_certification():
    r = run(fixture(), 'beam', beam_width=2, max_expansions=3)
    assert r['status'] == 'search_limit' and r['path']['edge_ids'] == ['st']
    assert not r['optimality_proven'] and not r['execution_authorized']
    assert r['consumer_host_wall_ns'] is None and not r['scientific_evidence_generated']
    assert r['search_policy'] == dict(algorithm='beam', beam_width=2, max_expansions=3)


@pytest.mark.parametrize('algorithm,cap,width', [('shortest', True, None), ('shortest', 0, None),
    ('shortest', 10, 1), ('beam', 10, None), ('beam', 10, True), ('geometry', 10, None)])
def test_search_controls_cannot_change_the_contract_silently(algorithm, cap, width):
    r = run(fixture(), algorithm, beam_width=width, max_expansions=cap)
    assert r['status'] == 'invalid_saved_graph'


def test_base_graph_cost_cannot_be_refunded_by_selecting_a_short_path():
    data = fixture(updates={'resource_caps': vec(3)}, base=vec(4))
    assert run(data)['status'] == 'invalid_saved_graph'


def test_missing_actual_action_bytes_is_not_a_zero_action_program():
    data = fixture(); del data[1][data[0]['edges'][0]['commands']['file']]
    assert run(data)['status'] == 'invalid_saved_graph'


def test_conflicting_alias_refs_to_same_file_fail_snapshot_binding():
    data = fixture()
    data[0]['nodes'][1]['latent'] = dict(file=data[0]['nodes'][0]['latent']['file'], sha256=q.sha(b'wrong'))
    assert run(data)['status'] == 'invalid_saved_graph'


def test_typed_receipt_binding_does_not_accept_boolean_cost_for_one():
    data = fixture()
    replace_ref(data, data[0]['edges'][0]['receipt'], lambda r: r.update(cost=True))
    assert run(data)['status'] == 'invalid_saved_graph'


def test_fixed_order_ties_do_not_depend_on_saved_list_order():
    specs = [('sa', 'S', 'A', 0, [.25], vec()), ('at', 'A', 'T', 1, [.25], vec()),
             ('st', 'S', 'T', 1, [.25], vec())]
    data = fixture(specs); expected = run(data)['path']['edge_ids']
    data[0]['edges'].reverse(); data[0]['nodes'].reverse()
    assert run(data)['path']['edge_ids'] == expected


@pytest.mark.parametrize('axis', q.AXES)
@pytest.mark.parametrize('where', ['base', 'delta', 'cap'])
def test_every_resource_axis_is_required_not_filled_zero(axis, where):
    data = fixture()
    v = {'base': data[0]['base_resources'], 'delta': data[0]['edges'][0]['resource_delta'],
         'cap': data[0]['contract']['resource_caps']}[where]
    v[axis] = None
    assert run(data)['status'] == 'invalid_saved_graph'


@pytest.mark.parametrize('value', [True, -1, 1.5, '0'])
def test_invalid_resource_count(value):
    data = fixture(); data[0]['base_resources']['flops'] = value
    assert run(data)['status'] == 'invalid_saved_graph'


@pytest.mark.parametrize('field', ['edge_id', 'parent', 'child', 'contract_sha256',
                                  'commands_sha256', 'steps', 'resource_delta',
                                  'cost', 'program_type', 'backend', 'status'])
def test_saved_transition_receipt_cannot_be_rebound(field):
    data = fixture(); ref = data[0]['edges'][0]['receipt']
    replace_ref(data, ref, lambda r: r.update({field: 'WRONG_SYNTHETIC_BINDING'}))
    assert run(data)['status'] == 'invalid_saved_graph'


@pytest.mark.parametrize('mutation', [
    lambda g: g.update(p1_passed=True),
    lambda g: g['contract']['prohibited'].update(training=False),
    lambda g: g['contract']['prohibited'].update(q_fitting=False),
    lambda g: g['edges'][0].update(cost=-1),
    lambda g: g['edges'][0].update(physical_success=True),
    lambda g: g['edges'][0].update(steps=True),
    lambda g: g['edges'][0].update(contract_sha256=q.sha(b'new model')),
    lambda g: g['edges'][0].update(commands=None),
    lambda g: g['edges'][0]['commands'].update(file='../outside'),
    lambda g: g['edges'][0]['commands'].update(file='/outside'),
    lambda g: g['edges'].append(deepcopy(g['edges'][0])),
    lambda g: g['nodes'].append(deepcopy(g['nodes'][0])),
    lambda g: g.update(target_nodes=['invented_target']),
    lambda g: g['contract'].update(max_l1=None),
])
def test_illegal_contract_or_scientific_flags_cannot_open_gate(mutation):
    data = fixture(); mutation(data[0])
    r = run(data)
    assert r['status'] == 'invalid_saved_graph' and not r['execution_authorized']


def test_entire_saved_graph_is_verified_not_just_winning_path():
    data = fixture()
    ref = data[0]['edges'][2]['commands']
    data[1][ref['file']] = b'corrupt unused direct edge'
    assert run(data)['status'] == 'invalid_saved_graph'


@pytest.mark.parametrize('raw', [b'\0', struct.pack('<d', float('nan')), struct.pack('<d', 2.)])
def test_shape_finiteness_and_raw_action_domain(raw):
    data = fixture(); ref = data[0]['edges'][0]['commands']
    data[1][ref['file']] = raw; ref['sha256'] = q.sha(raw)
    assert run(data)['status'] == 'invalid_saved_graph'


def test_absent_graph_and_empty_saved_edges_do_not_synthesize_research_assets():
    r = q.search(None, {}.__getitem__, algorithm='shortest', max_expansions=10)
    assert r['status'] == 'invalid_saved_graph' and r['path'] is None
    data = fixture([])
    assert run(data)['status'] == 'no_feasible_saved_path'


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'])
def test_json_is_strict(raw):
    with pytest.raises(ValueError):
        q.parse(raw)


def test_snapshot_loads_named_saved_bytes_once():
    data = fixture(); counts = {}
    def load(name):
        counts[name] = counts.get(name, 0) + 1
        return data[1][name]
    r = q.search(data[0], load, algorithm='shortest', max_expansions=100)
    assert r['status'] == 'saved_path_found' and set(counts.values()) == {1}


@pytest.mark.parametrize('seed', range(20))
def test_uniform_cost_matches_exhaustive_RAM_DAG_oracle(seed):
    rng = random.Random(seed)
    node_ids = ['S', 'A', 'B', 'C', 'T']
    specs = []
    for i, parent in enumerate(node_ids):
        for child in node_ids[i+1:]:
            if rng.random() < .65:
                specs.append((parent + child, parent, child, rng.randrange(5), [.25],
                              vec(predict_samples=rng.randrange(1, 4))))
    data = fixture(specs, updates={'resource_caps': vec(100, predict_samples=5)}, base=vec())
    feasible = []
    def visit(node, cost, resource):
        if node == 'T':
            feasible.append(cost)
        for _, parent, child, weight, _, res in specs:
            if parent == node and resource + res['predict_samples'] <= 5:
                visit(child, cost + weight, resource + res['predict_samples'])
    visit('S', 0, 0)
    r = run(data)
    if feasible:
        assert r['status'] == 'saved_path_found' and r['optimality_proven']
        assert r['path']['cost'] == dict(numerator=min(feasible), denominator=1)
    else:
        assert r['status'] == 'no_feasible_saved_path'


def test_cli_only_RAM_reads_and_stdout(monkeypatch, capsys):
    data = fixture()
    files = {'/SYNTHETIC/graph': q.encode(data[0])}
    monkeypatch.setattr(Path, 'read_bytes', lambda p: files[str(p)])
    monkeypatch.setattr(q, 'file_loader', lambda root: data[1].__getitem__)
    args = ['--graph', '/SYNTHETIC/graph', '--input-root', '/SYNTHETIC',
            '--algorithm', 'shortest', '--max-expansions', '100']
    assert q.main(args) == 0
    assert not q.parse(capsys.readouterr().out)['execution_authorized']
    files['/SYNTHETIC/graph'] = b'{"a":NaN}'
    assert q.main(args) == 2
    assert q.parse(capsys.readouterr().out)['status'] == 'invalid_saved_graph'


def test_symlink_escape_blocked_in_RAM(monkeypatch):
    original = Path.resolve
    monkeypatch.setattr(Path, 'resolve', lambda p: Path('/OUTSIDE/file') if str(p) == '/SYNTHETIC/escape' else original(p))
    monkeypatch.setattr(Path, 'read_bytes', lambda p: pytest.fail('must not read escaped bytes'))
    with pytest.raises(ValueError, match='symlink escapes'):
        q.file_loader('/SYNTHETIC')('escape')


def test_new_source_has_only_standard_library_imports():
    import ast
    tree = ast.parse(Path(q.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert set(imports) <= {'argparse', 'collections', 'copy', 'fractions', 'hashlib',
                            'heapq', 'json', 'math', 'pathlib', 'struct'}
