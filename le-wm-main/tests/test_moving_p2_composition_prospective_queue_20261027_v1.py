"""Synthetic RAM contracts only. MOCK_MAIN is not a scientific certificate."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import moving_p2_composition_prospective_queue_20261027_v1 as q


def h(value):
    return q.sha(str(value).encode())


def fixture():
    memory = {}

    def put(name, obj):
        data = obj if type(obj) is bytes else q.encode(obj)
        memory[name] = data
        return dict(file=name, sha256=q.sha(data))

    m = q.build_draft(['slot0', 'slot1'])
    m['sealed_at_ns'] = 100
    c = m['contract']
    c['assets'] = {k: put(k, ('MOCK_' + k).encode()) for k in q.ASSETS}
    c['horizon_steps'] = 10  # Synthetic shape, not a proposed scientific threshold.
    c['p1_required_metrics'] = ['MOCK_preregistered_real_gain']
    c['preregistration'] = put('prereg', dict(schema=q.STEM + '.preregistration',
        p1_required_metrics=c['p1_required_metrics'], assets_sha256=q.sha(q.encode(c['assets'])),
        horizon_steps=10))
    for i, row in enumerate(m['slots']):
        row['identity'] = dict(root_sha256=h(('root', i)), goal_sha256=h(('goal', i)),
                               history_sha256=h(('history', i)), blind_sha256=h(('blind', i)),
                               task_id=f'MOCK_task{i}')
        row['shared_cost_upper'] = dict.fromkeys(q.AXES, 1)
        row['matched_total_cap'] = dict.fromkeys(q.AXES, 6)
        for name, arm in row['arms'].items():
            arm['shared_sha256'] = q.shared_sha(c, row['identity'])
            arm['cost_upper'] = {p: dict.fromkeys(q.AXES, 1) for p in q.PHASES}
            arm['plan_rule'] = put(f'rule_{i}_{name}', dict(schema=q.STEM + '.plan_rule', arm=name,
                program_count=3 if name == 'composition' else 1, action_witness_required=True,
                intermediate_states_from_actions_only=True,
                search_rule=put(f'search_{i}_{name}', b'MOCK search rule, not executable')))
    contract_sha = q.sha(q.encode(c))
    p1 = dict(schema=q.STEM + '.p1', contract_sha256=contract_sha, certified_at_ns=90,
              gates=dict.fromkeys(q.P1_GATES, True), roots=[h('MOCK_prior_p1')],
              ci={'MOCK_preregistered_real_gain': [-2, -1]},
              preregistration_sha256=c['preregistration']['sha256'],
              evidence=[put('p1_evidence', b'MOCK ONLY not real evidence')])
    excluded = [h('MOCK_prior_p1'), h('MOCK_old_failed_root')]
    fresh = dict(schema=q.STEM + '.freshroot', issued_at_ns=101, contract_sha256=contract_sha,
                 identities=[deepcopy(r['identity']) for r in m['slots']], excluded_roots=excluded,
                 coverage_complete=True, independent=True, not_used_for_design=True,
                 no_outcome_selection=True,
                 exclusion_manifest=put('exclusions', dict(schema=q.STEM + '.exclusion_projection',
                     excluded_roots=excluded, unresolved_sources=[],
                     evidence=[put('historical_index', b'MOCK history census')])) )
    m['p1'] = put('p1', p1)
    m['freshroot'] = put('freshroot', fresh)
    trusted = dict(schema=q.STEM + '.main_authority', reviewer='MAIN', contract_sha256=contract_sha,
                   spec_sha256=q.spec_sha(m),
                   p1_sha256=m['p1']['sha256'], freshroot_sha256=m['freshroot']['sha256'])
    return m, memory, trusted


def run(data, trusted=True):
    m, memory, authority = data
    return q.build_queue(m, memory.__getitem__, authority if trusted else None)


def rewrite(data, name, fn, trust=False):
    m, memory, authority = data
    ref = m[name]
    obj = q.parse(memory[ref['file']]); fn(obj)
    memory[ref['file']] = q.encode(obj)
    ref['sha256'] = q.sha(memory[ref['file']])
    if trust:
        authority[name + '_sha256'] = ref['sha256']


def test_complete_mock_contract_is_never_execution_or_scientific_admission():
    data = fixture(); before = deepcopy(data)
    r = run(data)
    assert r['status'] == 'contract_valid_pending_MAIN'
    assert len(r['rows']) == 2 and data == before
    assert not r['execution_authorized'] and not r['P2_admission']
    assert not r['scientific_evidence_generated'] and r['actual_experiment_cost'] is None
    assert r['external_attestations_not_independently_proven']
    assert r['rows'][0]['planned_total_upper']['composition'] == dict.fromkeys(q.AXES, 6)


def test_draft_preserves_unknown_and_fixed_failed_denominator():
    draft = q.build_draft(['a', 'b', 'c'])
    r = q.build_queue(draft, {}.__getitem__)
    assert r['status'] == 'blocked' and len(r['rows']) == 3
    assert all(v is None for v in draft['slots'][0]['shared_cost_upper'].values())
    assert any('p1:' in s for s in r['blockers'])
    assert any('freshroot:' in s for s in r['blockers'])


@pytest.mark.parametrize('missing', ['p1', 'freshroot'])
def test_missing_attestation_is_not_implicit_pass(missing):
    data = fixture(); data[0][missing] = None
    assert run(data)['status'] == 'blocked'


def test_self_reported_passes_are_not_external_authority():
    assert run(fixture(), trusted=False)['status'] == 'blocked'


@pytest.mark.parametrize('gate', q.P1_GATES)
def test_all_real_p1_gates_required(gate):
    data = fixture()
    rewrite(data, 'p1', lambda p: p['gates'].update({gate: False}), trust=True)
    assert run(data)['status'] == 'blocked'


@pytest.mark.parametrize('bounds', [[-1, 0], [0, 1], [-1, None], [-1, True], [-1, -2], []])
def test_ci_touching_zero_missing_or_malformed_is_blocked(bounds):
    data = fixture()
    rewrite(data, 'p1', lambda p: p['ci'].update(MOCK_preregistered_real_gain=bounds), trust=True)
    assert run(data)['status'] == 'blocked'


@pytest.mark.parametrize('field', ['coverage_complete', 'independent', 'not_used_for_design', 'no_outcome_selection'])
def test_freshroot_unknown_or_false_cannot_release(field):
    data = fixture()
    rewrite(data, 'freshroot', lambda p: p.update({field: None}), trust=True)
    assert run(data)['status'] == 'blocked'


@pytest.mark.parametrize('kind', ['p1', 'freshroot'])
def test_rehashed_but_not_MAIN_approved_material_blocked(kind):
    data = fixture()
    rewrite(data, kind, lambda p: p.update(schema='fake'))
    assert run(data)['status'] == 'blocked'


@pytest.mark.parametrize('axis', q.AXES)
@pytest.mark.parametrize('phase', q.PHASES)
def test_every_phase_axis_missing_is_unknown_not_zero(phase, axis):
    data = fixture()
    data[0]['slots'][0]['arms']['composition']['cost_upper'][phase][axis] = None
    r = run(data)
    assert r['status'] == 'blocked' and len(r['rows']) == 2
    assert r['rows'][0]['planned_total_upper'] is None
    assert all(row['status'] == 'blocked' for row in r['rows'])


def test_cost_includes_prefix_derivatives_replay_execution_and_shared_setup():
    data = fixture()
    data[0]['slots'][0]['arms']['composition']['cost_upper']['prefix_search']['predict_samples'] = 2
    assert run(data)['status'] == 'blocked'  # True total 7, not suffix-only 1.
    data[0]['slots'][0]['matched_total_cap']['predict_samples'] = 7
    data[2]['spec_sha256'] = q.spec_sha(data[0])  # Mock a NEW approval, never reuse old approval.
    r = run(data)
    assert r['status'] == 'contract_valid_pending_MAIN'
    assert r['rows'][0]['planned_total_upper']['composition']['predict_samples'] == 7
    assert r['rows'][0]['planned_total_upper']['no_composition_sra']['predict_samples'] == 6
    # Common upper cap does NOT assert matched realized compute.


@pytest.mark.parametrize('value', [True, -1, 1.5, '0'])
def test_bad_cost_counts(value):
    data = fixture(); data[0]['slots'][0]['shared_cost_upper']['flops'] = value
    assert run(data)['status'] == 'blocked'


@pytest.mark.parametrize('mutation', [
    lambda m: m['slots'][0]['arms']['composition'].update(shared_sha256=h('wrong goal')),
    lambda m: m['slots'][0]['arms']['no_composition_sra'].update(composition_enabled=True),
    lambda m: m['contract']['prohibited'].update(training=False),
    lambda m: m['contract'].update(execution_backend='CPU'),
    lambda m: m['slots'][1].update(identity=deepcopy(m['slots'][0]['identity'])),
    lambda m: m.update(denominator=3),
    lambda m: m.update(sealed_at_ns=True),
    lambda m: m.update(passed=True),
])
def test_contract_ablation_identity_denominator_and_policy(mutation):
    data = fixture(); mutation(data[0])
    assert run(data)['status'] == 'blocked'


def test_extra_arm_cannot_override_root_goal_or_action_domain():
    data = fixture(); data[0]['slots'][0]['arms']['composition']['action_domain'] = 'wider'
    assert run(data)['status'] == 'blocked'


def test_postapproval_budget_edit_is_not_new_prospective_authority():
    data = fixture()
    data[0]['slots'][0]['matched_total_cap']['predict_samples'] = 7
    r = run(data)
    assert r['status'] == 'blocked'
    assert any('prospective spec binding mismatch' in s for s in r['blockers'])


def test_new_goal_cannot_reuse_a_historical_physical_root():
    data = fixture()
    root = data[0]['slots'][0]['identity']['root_sha256']
    # Even a newly trusted (mock) freshness declaration cannot contradict its census.
    def collide(p):
        p['excluded_roots'].append(root)
        ref = p['exclusion_manifest']
        projection = q.parse(data[1][ref['file']])
        projection['excluded_roots'].append(root)
        data[1][ref['file']] = q.encode(projection)
        ref['sha256'] = q.sha(data[1][ref['file']])
    rewrite(data, 'freshroot', collide, trust=True)
    assert run(data)['status'] == 'blocked'


def test_unknown_historical_coverage_is_not_self_filled():
    data = fixture()
    def unresolved(p):
        ref = p['exclusion_manifest']
        projection = q.parse(data[1][ref['file']])
        projection['unresolved_sources'] = ['MOCK unindexed historical source']
        data[1][ref['file']] = q.encode(projection)
        ref['sha256'] = q.sha(data[1][ref['file']])
    rewrite(data, 'freshroot', unresolved, trust=True)
    assert run(data)['status'] == 'blocked'


def test_file_loader_rejects_escaping_symlink_without_reading(monkeypatch):
    original = Path.resolve
    def resolve(path):
        return Path('/OUTSIDE/secret') if str(path) == '/MOCK/escape' else original(path)
    monkeypatch.setattr(Path, 'resolve', resolve)
    monkeypatch.setattr(Path, 'read_bytes', lambda path: pytest.fail('must not read escaped file'))
    with pytest.raises(ValueError, match='symlink escapes'):
        q.file_loader('/MOCK')('escape')


def test_p1_root_reuse_even_with_new_goal_is_blocked():
    data = fixture()
    root = data[0]['slots'][0]['identity']['root_sha256']
    rewrite(data, 'p1', lambda p: p.update(roots=[root]), trust=True)
    assert run(data)['status'] == 'blocked'


def test_fresh_exclusion_collision_and_short_census():
    data = fixture()
    rewrite(data, 'freshroot', lambda p: p.update(identities=p['identities'][:1]), trust=True)
    r = run(data)
    assert r['status'] == 'blocked' and len(r['rows']) == 2


def test_authority_chronology():
    data = fixture()
    rewrite(data, 'p1', lambda p: p.update(certified_at_ns=101), trust=True)
    assert run(data)['status'] == 'blocked'
    data = fixture()
    rewrite(data, 'freshroot', lambda p: p.update(issued_at_ns=99), trust=True)
    assert run(data)['status'] == 'blocked'


def test_witness_and_multi_program_rule_required():
    data = fixture()
    ref = data[0]['slots'][0]['arms']['composition']['plan_rule']
    rule = q.parse(data[1][ref['file']]); rule['program_count'] = 1
    data[1][ref['file']] = q.encode(rule); ref['sha256'] = q.sha(data[1][ref['file']])
    assert run(data)['status'] == 'blocked'


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'])
def test_strict_json(raw):
    with pytest.raises(ValueError):
        q.parse(raw)


def test_bound_evidence_missing_bytes_bad_sha_path_escape():
    for name in ('not_found', '../escape', '/absolute'):
        data = fixture(); data[0]['p1']['file'] = name
        assert run(data)['status'] == 'blocked'
    data = fixture(); data[1]['p1'] += b' '
    assert run(data)['status'] == 'blocked'


def test_unknown_addition_stays_unknown():
    assert q.add_vectors([dict.fromkeys(q.AXES, 1), dict.fromkeys(q.AXES, None)]) == dict.fromkeys(q.AXES, None)


def test_cli_RAM_only(monkeypatch, capsys):
    data = fixture(); m, memory, authority = data
    files = {'/MOCK/manifest': q.encode(m), '/MOCK/authority': q.encode(authority)}
    monkeypatch.setattr(Path, 'read_bytes', lambda self: files[str(self)])
    monkeypatch.setattr(q, 'file_loader', lambda root: memory.__getitem__)
    assert q.main(['draft', '--slots', 's0', 's1']) == 0
    assert q.parse(capsys.readouterr().out)['slots'][0]['identity'] is None
    base = ['validate', '--manifest', '/MOCK/manifest', '--input-root', '/MOCK']
    assert q.main(base) == 2
    assert q.parse(capsys.readouterr().out)['status'] == 'blocked'
    approved = base + ['--main-authority', '/MOCK/authority', '--main-authority-sha256', q.sha(files['/MOCK/authority'])]
    assert q.main(approved) == 0
    assert not q.parse(capsys.readouterr().out)['execution_authorized']
    files['/MOCK/manifest'] = b'{"schema":NaN}'
    assert q.main(base) == 2
    assert q.parse(capsys.readouterr().out)['status'] == 'blocked'


def test_import_has_no_neural_or_physics_dependencies():
    # Check this new source's imports, without making claims about pytest plugins.
    import ast
    tree = ast.parse(Path(q.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert set(imports) <= {'argparse', 'copy', 'hashlib', 'json', 'math', 'pathlib'}
