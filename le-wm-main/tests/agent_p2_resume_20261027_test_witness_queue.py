"""RAM mocks + real saved-layout slices only; no model/simulator instantiation."""
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_p2_resume_20261027_witness_queue as m


def h(v): return m.witness.digest(v)


def context():
    return dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
                action_dim=10, latent_history=[[0.]*192 for _ in range(3)], action_history=[[0.]*10 for _ in range(2)])


def advance(c, actions):
    c = deepcopy(c)
    for offset in range(0, len(actions), 5):
        packet = [x for row in actions[offset:offset+5] for x in row]
        latent = c['latent_history'][-1].copy()
        latent[:10] = [a+b for a, b in zip(latent[:10], packet)]
        c['latent_history'] = c['latent_history'][1:]+[latent]
        c['action_history'] = c['action_history'][1:]+[packet]
    return c


def attempt(case, arm, j):
    actions = [[((j % 4)-1)*.125, .125]]*5 + [[0., -.125]]*5
    chunks = [actions[:5], actions[5:]] if arm == 'conditional_suffix_composition' else [actions]
    prov = {k: h(k) for k in m.witness.PROVENANCE_KEYS}; prov['metric_sha256'] = case['metric_sha256']
    edges = []; c = deepcopy(case['root']['context'])
    for ei, chunk in enumerate(chunks):
        end = advance(c, chunk)
        edges.append(dict(program_id=f'{arm}_{j:02d}_edge_{ei}', kind='progress', root_id=case['root_sha256'],
                          provenance=prov, action_domain=case['action_domain'], input_context=c,
                          output_context=end, actions=chunk))
        c = end
    receipt = m.witness.compose(root=case['root'], provenance=prov, action_domain=case['action_domain'],
                                edges=edges, replay=advance, max_edges=len(edges), max_actions=10)
    return dict(index=j, candidate_id=f'{arm}_{j:02d}', completed=True, actions=actions,
                witness=dict(edges=edges, receipt=receipt),
                suffix_compilation=dict(input_context_sha256=h(edges[0]['output_context'])))


@pytest.fixture(scope='module')
def native():
    cases = []; paired_rows = []; sra_rows = []
    for i in range(4):
        root = h(['root', i]); domain = dict(name='raw', units='official PushT live command',
            low=[-1., -1.], high=[1., 1.], action_block=5, normalization_sha256=h('normalizer'))
        case = dict(root_index=i, root_sha256=root, goal_vector_sha256=h(['goal', i]),
                    metric_sha256=h('metric'), projection_sha256=h('projection'), action_domain=domain,
                    root=dict(root_id=root, root_rgb_sha256=h(['rgb', i]), goal_rgb_sha256=h(['goal_rgb', i]), context=context()))
        cases.append(case); results = []
        for arm in ('joint_h2_no_composition', 'conditional_suffix_composition'):
            attempts = [attempt(case, arm, j) for j in range(40)]
            ledger = [dict(completed=True, predict_calls=1, predict_samples=1, action_encoder_calls=1) for _ in range(84)]
            results.append(dict(arm=arm, attempts=attempts, ledger=ledger, complete=True))
        entries = [dict(kind=k, samples=1, context_tokens=3, completed=True) for k in ('predict', 'action_encoder')]
        paired_rows.append(dict(root_sha256=root, results=results, actual_model_entries=entries, resource_receipt={'fixture_only': True}))
        sra_rows.append(dict(root_sha256=root, actual_model_entries=entries,
            result=dict(attempts=[dict(candidate_id=f'joint_h2_no_composition_sra_{j:02d}', completed=True,
                                      actions=deepcopy(a['actions'])) for j, a in enumerate(results[0]['attempts'])])))
    return {k: dict(protocol=dict(context=dict(cases=deepcopy(cases))),
                    report=dict(rows=deepcopy(paired_rows if k == 'composition' else sra_rows)),
                    report_sha256=h(k), bindings={}, verification_work=dict(neural_predict_calls=0)) for k in m.KINDS}


def test_all320_fixed_candidates_480_original_edges_queue_stays_blocked(native):
    result = m.project(native)
    assert result['candidate_denominator'] == len(result['candidates']) == 320
    assert result['edge_denominator'] == 480 and result['root_denominator'] == 4
    assert result['queue_audit']['status'] == 'blocked' and len(result['queue_audit']['rows']) == 4
    assert result['execution_authorized'] is result['P2_admission'] is False
    assert result['strict_resource_graph_ready'] is False
    assert result['strict_resource_unknown_axes'] == list(m.graph.AXES)
    for row in result['queue_manifest']['slots']:
        assert row['identity']['blind_sha256'] is row['identity']['task_id'] is None
        assert all(v is None for v in row['matched_total_cap'].values())
    for r in result['candidates']:
        if len(r['edges']) == 2: assert r['edges'][0]['child'] == r['edges'][1]['parent']
        assert all(e['steps'] == (5 if len(r['edges']) == 2 else 10) for e in r['edges'])


@pytest.mark.parametrize('damage', ['history', 'goal', 'normalizer', 'bank', 'denominator', 'duplicate_root', 'order'])
def test_full_census_identity_mutations_never_filter_to_favorable_remainder(native, damage):
    n = deepcopy(native)
    if damage == 'history': n['no_composition_sra']['protocol']['context']['cases'][0]['root']['context']['action_history'][0][0] = .125
    elif damage == 'goal': n['no_composition_sra']['protocol']['context']['cases'][0]['goal_vector_sha256'] = h('other')
    elif damage == 'normalizer': n['no_composition_sra']['protocol']['context']['cases'][0]['action_domain']['normalization_sha256'] = h('other')
    elif damage == 'bank': n['no_composition_sra']['report']['rows'][0]['result']['attempts'][0]['actions'][0][0] = .5
    elif damage == 'denominator': n['composition']['report']['rows'][0]['results'][0]['attempts'].pop()
    elif damage == 'duplicate_root': n['composition']['report']['rows'].append(deepcopy(n['composition']['report']['rows'][0]))
    else: n['composition']['report']['rows'][0]['results'].reverse()
    with pytest.raises(ValueError): m.project(n)


def test_explicit_root_join_not_saved_row_position(native):
    n = deepcopy(native); n['no_composition_sra']['report']['rows'].reverse()
    assert m.graph.encode(m.project(n)) == m.graph.encode(m.project(native))


@pytest.mark.parametrize('damage', ['boundary', 'model', 'raw_action', 'suffix', 'goal'])
def test_native_action_witness_cannot_be_relabelled(native, damage):
    case = deepcopy(native['composition']['protocol']['context']['cases'][0])
    a = deepcopy(native['composition']['report']['rows'][0]['results'][1]['attempts'][0])
    if damage == 'boundary': a['witness']['edges'][1]['input_context']['action_history'][0][0] += .125
    elif damage == 'model': a['witness']['edges'][0]['provenance']['checkpoint_sha256'] = h('other')
    elif damage == 'raw_action': a['actions'][0][0] += .125
    elif damage == 'suffix': a['suffix_compilation']['input_context_sha256'] = h('other')
    else: case['root']['goal_rgb_sha256'] = h('other')
    with pytest.raises(ValueError): m.candidate_record(case, a, 'conditional_suffix_composition', 0, {}, {})


def test_prepare_verify_consume_no_dispatch_even_forged_authority(native, monkeypatch):
    inputs = {k: dict(directory='/RAM/'+k, report_sha256=h(k)) for k in m.KINDS}
    monkeypatch.setattr(m, 'load_native', lambda kind, *args: deepcopy(native[kind]))
    monkeypatch.setattr(m, 'own_files', lambda: [])
    p = m.prepare(inputs); expected = m.graph.sha(m.graph.encode(p))
    out = m.consume(p, expected)
    assert out['candidate_denominator'] == len(out['candidate_actions']) == 320
    assert out['new_neural_calls'] == out['new_physics_steps'] == 0
    assert not out['scientific_pass'] and not out['execution_authorized']
    with pytest.raises(ValueError, match='confirmatory dispatch blocked'): m.consume(p, expected, confirmatory=True)
    p['engineering_ready'] = False
    with pytest.raises(ValueError, match='SHA/schema'): m.verify(p, expected)
    with pytest.raises(ValueError, match='witness/queue/census drift'): m.verify(p, m.graph.sha(m.graph.encode(p)))


def test_confirmatory_rejected_before_any_read_or_export(monkeypatch):
    monkeypatch.setattr(m, 'read', lambda *a: pytest.fail('must not read'))
    with pytest.raises(ValueError, match='no confirmatory run API'): m.main(['run-saved', '--confirmatory'])


@pytest.fixture(scope='module')
def real_slice():
    directory = m.ROOT/'results/p2_composition_paired_20261027_v2_prepared01'
    assert m.file_sha(directory/'report.json') == '0c2a92842fa993cdc3dc566bab86bf124db15a6a0d5b054cbfef5e2d75f86148'
    protocol = m.read(directory/'protocol.json'); report = m.read(directory/'report.json')
    return protocol['context']['cases'][0], report['rows'][0]['results'][1]['attempts'][0]


def test_real_saved_layout_seam_192_history_rawfloat32_native_receipt(real_slice):
    case, a = real_slice; nodes = {}
    r = m.candidate_record(case, a, 'conditional_suffix_composition', 0, {'saved_slice_only': True}, nodes)
    assert len(r['edges']) == 2 and r['edges'][0]['child'] == r['edges'][1]['parent']
    assert all(n['context']['latent_dim'] == 192 and n['context']['context_length'] == 3 for n in nodes.values())
    assert all(len(n['context']['action_history']) == 2 for n in nodes.values())
    assert len(bytes.fromhex(r['edges'][0]['commands_float32_le_hex'])) == 5*2*4
    assert r['native_receipt_sha256'] == a['witness']['receipt']['receipt_sha256']


@pytest.mark.parametrize('damage', ['boundary', 'goal', 'action', 'model', 'suffix'])
def test_real_saved_layout_mutations_failclosed(real_slice, damage):
    case, a = deepcopy(real_slice)
    if damage == 'boundary': a['witness']['edges'][1]['input_context']['action_history'][0][0] += .125
    elif damage == 'goal': case['root']['goal_rgb_sha256'] = h('mutated')
    elif damage == 'action': a['witness']['edges'][0]['actions'][0][0] += .125
    elif damage == 'model': a['witness']['edges'][0]['provenance']['checkpoint_sha256'] = h('mutated')
    else: a['suffix_compilation']['input_context_sha256'] = h('mutated')
    with pytest.raises(ValueError): m.candidate_record(case, a, 'conditional_suffix_composition', 0, {}, {})


def test_real_native_report_sha_rejected_before_any_native_export():
    with pytest.raises(ValueError, match='external native report SHA'):
        m.load_native('composition', m.ROOT/'results/p2_composition_paired_20261027_v2_prepared01', h('not the report'))
