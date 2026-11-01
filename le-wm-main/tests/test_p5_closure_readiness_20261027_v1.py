"""Pure small fixtures: protocol/tool tests, never P5 scientific evidence."""
import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p5_closure_readiness_20261027_v1 as m


def fixture():
    roots = [str(i) * 64 for i in range(4)]
    n = dict(all_checks=True, check_count=45, checks={str(i): True for i in range(45)},
             P5_admission=False, C2_established=False, input_files_checked=101,
             root_identity_sha256='e' * 64, model_state_identities={}, stage_gates={}, observations=[])
    methods = [dict(method=name, total_compute_matched=False, direct_paper_comparison=False,
                    paper_protocol_reproduced=False, paper_reported_cells=0,
                    local_roots_scope='calibration', missing_or_unmatched=['budget'],
                    local_evidence=[{'root_sha256': r} for r in roots] if name in m.baseline.METHODS else [])
               for name in ['CEM', 'MPPI', 'gradient MPC', 'GN', 'GRASP', 'Hi-LeWM', 'LeWM',
                            'PlanWAM', 'random shooting', 'no-composition SRA']]
    inv = dict(scientific_admission=False, missing_assets=[], methods=methods,
               checkpoint_bindings_recorded_not_rehashed={'config.json': 'a' * 64})
    p1 = dict(status='cuda_completed_h2_prerequisite', model_unchanged=True,
              P1_admission=False, P3_admission=False, P4_admission=False,
              contract=dict(new_independent_roots=0), protocol_id='a' * 64,
              rows=[dict(root_sha256=r, arm=a, status='completed', compilation=dict(candidates=[{}] * 10))
                    for r in roots for a in ('old_Q', 'goal_patch_Q')])
    strict = dict(status='native_dynamics_strict_load_only', P5_admission=False)
    equiv = dict(status='paired_equivalence_failed', P5_admission=False)
    return n, inv, p1, strict, equiv


def test_summary_mock_is_not_efficacy():
    s = m.evidence_summary(*fixture())
    assert s['P1']['candidates'] == 80
    assert s['P1']['independent_roots'] == 0
    assert s['P1']['plateau'] == s['P1']['two_stage_task_reach'] == 'unverified'
    assert s['native']['root_namespace'] != s['P1']['root_namespace']
    assert s['architecture']['official_weight_authentication'] == 'unknown'
    assert s['architecture']['encoder_equivalence'] == 'paired_equivalence_failed'


@pytest.mark.parametrize('field,value', [('all_checks', 1), ('check_count', 44), ('P5_admission', True), ('C2_established', True)])
def test_native_flags_do_not_unlock(field, value):
    args = fixture()
    args[0][field] = value
    with pytest.raises(ValueError):
        m.evidence_summary(*args)


def test_native_empty_or_false_checks():
    for checks in ({}, {'0': False}, {str(i): 1 for i in range(45)}):
        args = fixture()
        args[0]['checks'] = checks
        with pytest.raises(ValueError):
            m.evidence_summary(*args)


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'wrong_arm', 'partial_bank', 'uncompleted', 'new_root'])
def test_p1_census(kind):
    args = fixture()
    p = args[2]
    if kind == 'duplicate':
        p['rows'][-1] = copy.deepcopy(p['rows'][0])
    elif kind == 'missing':
        p['rows'].pop()
    elif kind == 'wrong_arm':
        p['rows'][0]['arm'] = 'unbound_Q'
    elif kind == 'partial_bank':
        p['rows'][0]['compilation']['candidates'].pop()
    elif kind == 'uncompleted':
        p['rows'][0]['status'] = 'failed'
    else:
        p['contract']['new_independent_roots'] = 1
    with pytest.raises(ValueError):
        m.evidence_summary(*args)


@pytest.mark.parametrize('field', ['P1_admission', 'P3_admission', 'P4_admission'])
def test_upstream_true_cannot_self_admit(field):
    args = fixture()
    args[2][field] = True
    with pytest.raises(ValueError):
        m.evidence_summary(*args)


@pytest.mark.parametrize('field', ['total_compute_matched', 'direct_paper_comparison', 'paper_protocol_reproduced'])
def test_reported_or_budget_flags_not_local_controls(field):
    args = fixture()
    args[1]['methods'][0][field] = True
    with pytest.raises(ValueError):
        m.evidence_summary(*args)


@pytest.mark.parametrize('kind', ['wrong_root', 'missing', 'duplicate_method', 'missing_asset', 'admission'])
def test_baseline_wiring(kind):
    args = fixture()
    inv = args[1]
    if kind == 'wrong_root':
        inv['methods'][0]['local_evidence'][0]['root_sha256'] = 'f' * 64
    elif kind == 'missing':
        inv['methods'][0]['local_evidence'].pop()
    elif kind == 'duplicate_method':
        inv['methods'][0]['method'] = inv['methods'][1]['method']
    elif kind == 'missing_asset':
        inv['missing_assets'] = ['missing.py']
    else:
        inv['scientific_admission'] = True
    with pytest.raises(ValueError):
        m.evidence_summary(*args)


@pytest.mark.parametrize('idx,status', [(3, 'D2P_ready'), (4, 'paired_equivalence_passed')])
def test_strictload_and_failed_equivalence_scope(idx, status):
    args = fixture()
    args[idx]['status'] = status
    with pytest.raises(ValueError):
        m.evidence_summary(*args)


def test_requirement_graph_separates_E3_and_architecture():
    slots = m.task_slots()
    d = {r['id']: r for r in slots}
    order = m.graph(slots)
    assert len(order) == 9
    assert 'formal_P1' in d['P5_independent_replication']['depends_on']
    assert 'native_P0_task' in d['P5_independent_replication']['depends_on']
    assert 'native_P0_task' not in d['same_model_multitask']['depends_on']
    assert all(r['bound_completion_artifact'] is None for r in slots)
    assert all(order.index(dep) < order.index(r['id']) for r in slots for dep in r['depends_on'])
    assert d['conditional_roots']['depends_on'] == []


@pytest.mark.parametrize('kind', ['cycle', 'unknown', 'duplicate'])
def test_invalid_dependency_graph(kind):
    rows = m.task_slots()
    if kind == 'cycle':
        rows[0]['depends_on'] = ['used_components']
    elif kind == 'unknown':
        rows[0]['depends_on'] = ['not_declared']
    else:
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError):
        m.graph(rows)


@pytest.mark.parametrize('raw', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{broken'])
def test_bad_json(raw):
    with pytest.raises(ValueError):
        m.decode(raw)


def test_reader_hash_drift_and_reserve_no_writes(tmp_path):
    # Fixture writes only in pytest's temporary directory, never saved artifacts.
    p = tmp_path / 'input.json'
    p.write_text('{"x":1}')
    r = m.Reader(tmp_path)
    assert r.read('input.json') == {'x': 1}
    with pytest.raises(ValueError, match='SHA'):
        r.raw('input.json', '0' * 64)
    p.write_text('{"x":2}')
    with pytest.raises(ValueError, match='SHA'):
        r.stable()


@pytest.mark.parametrize('kind', ['traversal', 'absolute', 'symlink', 'directory', 'cap', 'total_cap'])
def test_reader_rejects_nonlocal_or_unbounded(tmp_path, monkeypatch, kind):
    p = tmp_path / 'small.json'
    p.write_text('{}')
    rel = 'small.json'
    if kind == 'traversal':
        rel = '../small.json'
    elif kind == 'absolute':
        rel = str(p)
    elif kind == 'symlink':
        (tmp_path / 'link.json').symlink_to(p)
        rel = 'link.json'
    elif kind == 'directory':
        rel = '.'
    elif kind == 'cap':
        monkeypatch.setattr(m, 'CAP', 1)
    else:
        monkeypatch.setattr(m, 'TOTAL_CAP', 1)
    with pytest.raises(ValueError):
        m.Reader(tmp_path).raw(rel)


@pytest.mark.parametrize('change', ['true', 'integer_flag', 'model', 'source', 'requirement', 'missing'])
def test_manifest_rejects_tamper_or_stale(change):
    current = dict(scientific_admission=False, execution_authorized=False,
                   execution_manifest={'execution_authorized': False},
                   model='a', bindings={'source': 'a'}, requirement=None)
    saved = copy.deepcopy(current)
    assert m.verify_manifest(saved, current)
    if change == 'true':
        saved['execution_authorized'] = True
    elif change == 'integer_flag':
        saved['execution_manifest']['execution_authorized'] = 0
    elif change == 'model':
        saved['model'] = 'b'
    elif change == 'source':
        saved['bindings']['source'] = 'b'
    elif change == 'requirement':
        saved['requirement'] = 'mock_pass'
    else:
        del saved['requirement']
    with pytest.raises(ValueError):
        m.verify_manifest(saved, current)


def test_cli_reserve_stops_before_reads(monkeypatch, capsys):
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda p: type('Disk', (), {'free': m.RESERVE - 1})())
    monkeypatch.setattr(m, 'produce', lambda: pytest.fail('must not read below reserve'))
    assert m.main([]) == 2
    assert '200 MiB' in capsys.readouterr().out


def test_cli_stdout_only_and_blocked_exit(monkeypatch, capsys):
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda p: type('Disk', (), {'free': m.RESERVE})())
    monkeypatch.setattr(m, 'produce', lambda: {'scientific_admission': False, 'execution_authorized': False})
    assert m.main([]) == 0
    assert json.loads(capsys.readouterr().out)['scientific_admission'] is False
    assert m.main(['--require-admission']) == 3


def test_no_neural_physics_subprocess_or_network_imports():
    import ast
    tree = ast.parse(Path(m.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split('.')[0])
    assert imported <= {'argparse', 'hashlib', 'json', 'math', 'pathlib', 'shutil', 'stat'}
