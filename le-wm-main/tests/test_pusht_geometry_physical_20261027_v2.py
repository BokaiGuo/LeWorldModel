"""Nonneural durability/census contracts; fixtures are not calibration results."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import pusht_geometry_physical_20261027_v2 as p


@pytest.fixture(scope='module')
def saved():
    """Finite algebraic records accepted by the real planning verifier; no model."""
    g = p.planning
    goal, q = np.ones(192), np.eye(192)
    jac, origin = np.eye(192, 10), np.zeros(192)
    def measured(params, value=0.):
        return dict(prefix_params=np.asarray(params, dtype=np.float32).astype(float).tolist(),
            predicted_prefix=np.full(192, value).tolist(),
            nominal_terminal=origin.tolist(), jacobian=jac.tolist())
    records = [measured(np.zeros(10))]
    for epsilon in g.EPSILONS:
        for axis in range(10):
            for sign in (-1, 1):
                params = np.zeros(10); params[axis] = sign*epsilon
                records.append(measured(params))
    grids = g.gradients_from_records(records, goal, q)
    common = g.packet.spectral_bank(jac, origin, goal, q)
    arms = []
    for arm in p.ARMS:
        runs = []
        for center in g.starts(common):
            trace = []
            for iteration in range(g.ITERATIONS):
                probes = [measured(center)]
                for axis in range(10):
                    for sign in (-1, 1):
                        params = np.asarray(center, dtype=np.float32).copy()
                        params[axis] += sign*g.EPSILONS[0]
                        probes.append(measured(params))
                gradients = g.gradients_from_records(probes, goal, q, center=center, epsilons=g.EPSILONS[:1])
                next_params = g.update(center, gradients[0]['gradients'][arm])
                trace.append(dict(iteration=iteration, center_params=center, probe_records=probes,
                    gradients=gradients, next_params=next_params))
                center = next_params
            runs.append(trace)
        bank = g.augmented_bank(common, runs, goal, q, arm)
        candidates = []
        for i, params in enumerate(bank['params']):
            m = measured(params, i/100)
            score = g.objective(m, goal, q)
            suffix = np.asarray(score['suffix_solve']['params'], dtype=np.float32)
            commands = np.concatenate((np.tanh(np.asarray(params, dtype=np.float32)).reshape(5, 2),
                np.tanh(suffix).reshape(5, 2)))
            candidates.append(dict(**m, **score, candidate_id=f'prefix_{i}',
                kind=bank['kinds'][i], suffix_params=suffix.astype(float).tolist(),
                commands=commands.tolist(), predicted_terminal=origin.tolist(),
                independent_model_replay_exact=True))
        arms.append(dict(arm=arm, bank=bank, optimization=runs, candidates=candidates,
            selected_candidate_id='prefix_11', greedy_candidate_id='prefix_11'))
    roots = [f'root{i}' for i in range(4)]
    source = dict(protocol=dict(roots=roots), rows=[dict(root_sha256=r,
        compiled=dict(goal_vectors=[goal.tolist()], scoring=dict(metric=dict(projection=q.tolist()))))
        for r in roots])
    report = dict(status='plans_sealed', gates=dict(all96=True),
        protocol=dict(roots=roots, epsilons=list(g.EPSILONS), arms=list(p.ARMS),
            damping=g.packet.DAMPING, cap=g.packet.CAP, radii=list(g.packet.RADII),
            iterations=g.ITERATIONS, step=g.STEP, candidates_per_arm=12),
        plans=[dict(root_index=i, root_sha256=r, arms=deepcopy(arms),
            root_jacobian=jac.tolist(), root_nominal=origin.tolist(),
            probe_records=deepcopy(records), gradient_grids=deepcopy(grids)) for i, r in enumerate(roots)])
    return report, source


@pytest.fixture(scope='module')
def plans(saved):
    return p.plans_from_saved(*saved)


def test_real_verifier_called_and_inputs_untouched(saved, monkeypatch):
    snapshot = deepcopy(saved)
    real = p.verify_plans
    calls = []
    def verify(report, source):
        calls.append((report, source))
        return real(report, source)
    monkeypatch.setattr(p, 'verify_plans', verify)
    plans = p.plans_from_saved(*saved)
    assert len(calls) == 1 and saved == snapshot
    assert len(plans) == 96
    for plan in plans:
        assert plan['selection_sha256'] == p.content_digest(
            {k:v for k,v in plan.items() if k != 'selection_sha256'})
        assert plan['commands'] == plan['saved_candidate']['commands']
        assert plan['corrected_search_horizon'] == 2
    assert p.packet_physical.ARMS == ('constant', 'full')


def test_prefix_v3_api_physical_v2_name_and_failed_report_rejected(saved):
    assert p.planning.TAG == 'pusht_geometry_prefix_20261027_v3'
    assert p.verify_plans is p.planning.verify_plans
    assert p.TAG == 'pusht_geometry_physical_20261027_v2'
    report, source = deepcopy(saved)
    report['status'] = 'integrity_fail'
    with pytest.raises(ValueError, match='sealed geometry planning'):
        p.plans_from_saved(report, source)


@pytest.mark.parametrize('mutation', ['root_index', 'duplicate', 'count', 'unsafe',
    'commands', 'bool_commands', 'nan', 'terminal', 'selected', 'greedy', 'gates', 'roots'])
def test_bad_plans_fail_closed(saved, mutation):
    report, source = deepcopy(saved)
    row = report['plans'][0]; arm = row['arms'][0]; c = arm['candidates'][0]
    if mutation == 'root_index': row['root_index'] = True
    elif mutation == 'duplicate': arm['candidates'][1]['candidate_id'] = 'prefix_0'
    elif mutation == 'count': arm['candidates'].pop()
    elif mutation == 'unsafe': c['candidate_id'] = '../../x'
    elif mutation == 'commands': c['commands'][0][0] = 1.1
    elif mutation == 'bool_commands': c['commands'] = [[False, False]]*10
    elif mutation == 'nan': c['predicted_terminal'][0] = float('nan')
    elif mutation == 'terminal': c['predicted_terminal'] = [0.]
    elif mutation == 'selected': arm['selected_candidate_id'] = 'prefix_0'
    elif mutation == 'greedy': arm['greedy_candidate_id'] = 'prefix_0'
    elif mutation == 'gates': report['gates'] = {}
    else: report['protocol']['roots'].reverse()
    with pytest.raises((ValueError, KeyError)):
        p.plans_from_saved(report, source)


class Env:
    def __init__(self, missing_at=None, fail_at=None, terminate_at=None):
        self.n = 0
        self.missing_at, self.fail_at, self.terminate_at = missing_at, fail_at, terminate_at
    def render(self):
        rgb = np.zeros((224, 224, 3), np.uint8)
        if self.n != self.missing_at:
            rgb[10:12, 10:12] = [65, 105, 225]
        return rgb
    def step(self, command):
        if self.n+1 == self.fail_at:
            raise RuntimeError('step failed')
        self.n += 1
        return None, 0., self.n == self.terminate_at, False, {}


def physical(tmp_path, plans, env):
    return p.durable.physical_branch(plan=plans[0], repetition=0, output=tmp_path,
        reset=lambda:{}, env=env, state_of=lambda e:np.array([e.n, 0, 0, 0, 0.]),
        evaluate=lambda state:dict(official={}, pose={}))


def persist_census(tmp_path, plans, first=None):
    rows = [] if first is None else [first]
    p.materialize_missing(plans, rows, tmp_path)
    p.write(tmp_path/'physical_report.json', dict(rows=rows, census=p.branch_census(plans, rows)))
    return rows


def test_all96_blocked_census_and_exclusive_files(tmp_path, plans):
    rows = persist_census(tmp_path, plans)
    assert len(rows) == len(list(tmp_path.glob('*.physical.json'))) == 96
    assert all(r['status'] == 'blocked' and r['attempted_commands'] == 0 for r in rows)
    census = p.branch_census(plans, rows)
    assert census['all96branchcensus'] and len(census['branch_ids']) == 96
    assert all(x['arms'] == dict(geometry=12, stopK=12) for x in census['roots'])
    displays = p.comparisons(plans, rows)
    assert len(displays) == 8
    assert all(x['roles']['selected']['candidate_id'] == 'prefix_11' for x in displays)
    assert all(x['roles']['zero']['candidate_id'] == 'prefix_0' for x in displays)
    with pytest.raises(FileExistsError):
        p.materialize_missing(plans, [], tmp_path)


@pytest.mark.parametrize('mutation', ['drop', 'duplicate', 'hash', 'method', 'horizon'])
def test_census_rejects_equal_count_substitution(tmp_path, plans, mutation):
    rows = persist_census(tmp_path, plans)
    if mutation == 'drop': rows.pop()
    elif mutation == 'duplicate': rows[-1] = rows[0]
    elif mutation == 'hash': rows[0]['selection_sha256'] = 'changed'
    elif mutation == 'method': rows[0]['method'] = 'geometry_prefix_12'
    else: rows[0]['planned_commands'] = 5
    with pytest.raises(ValueError): p.branch_census(plans, rows)


@pytest.mark.parametrize('stop,eligible', [(0, (False, False)), (3, (False, False)),
    (5, (True, False)), (7, (True, False)), (10, (True, True)), (None, (True, True))])
def test_guard_and_exact_endpoints_no_substitution(tmp_path, plans, stop, eligible):
    row = physical(tmp_path, plans, Env(missing_at=stop))
    assert row['completed_commands'] == (10 if stop is None else stop)
    assert row['status'] == ('completed' if stop is None else 'observation_unavailable')
    rows = persist_census(tmp_path, plans, row)
    paths = []
    def setup():
        assert len(list(tmp_path.glob('*.physical.json'))) == 96
        def encode(path):
            paths.append(path.name)
            return [0.]*192
        return encode
    receipts = p.encode_observations(rows, plans, tmp_path, setup)
    assert len(receipts) == 192
    assert tuple(r['comparable'] for r in receipts[:2]) == eligible
    assert all(r['status'] == 'blocked' for r in receipts[2:])
    assert paths == [f'root_0_geometry_prefix_0_replay0_rgb_{step}.png'
        for step, ok in zip((5, 10), eligible) if ok]


def test_attempt_is_fsynced_before_env_step(tmp_path, plans, monkeypatch):
    fsyncs = []
    original = p.durable.os.fsync
    def fsync(fd):
        original(fd)
        fsyncs.append(fd)
    monkeypatch.setattr(p.durable.os, 'fsync', fsync)
    class CheckedEnv(Env):
        def step(self, command):
            events = [json.loads(s) for s in next(tmp_path.glob('*.events.jsonl')).read_text().splitlines()]
            assert events[-1]['kind'] == 'attempt'
            assert events[-1]['primitive_step'] == self.n+1
            assert fsyncs
            fsyncs.clear()
            return super().step(command)
    row = physical(tmp_path, plans, CheckedEnv(fail_at=4))
    assert row['physical_env_step_attempts'] == 4 and row['physical_env_step_returns'] == 3
    assert row['journal_sha256'] == p.sha(row['journal_path'])
    assert json.loads(next(tmp_path.glob('*.physical.json')).read_text()) == row
    with pytest.raises(ValueError): p.image_receipt(row, 'prefix')


def test_termination_not_shortened_terminal(tmp_path, plans):
    row = physical(tmp_path, plans, Env(terminate_at=7))
    assert row['status'] == 'environment_terminated' and row['completed_commands'] == 7
    assert p.image_receipt(row, 'prefix').name.endswith('_rgb_5.png')
    with pytest.raises(ValueError): p.image_receipt(row, 'terminal')


@pytest.mark.parametrize('failure', ['setup', 'prefix', 'journal', 'png'])
def test_encoding_failures_preserve_physical_receipts(tmp_path, plans, failure):
    row = physical(tmp_path, plans, Env())
    if failure == 'journal': row['journal_sha256'] = 'changed'
    if failure == 'png':
        # Test-fixture corruption only, never an existing experiment artifact.
        with p.image_receipt(row, 'prefix').open('ab') as stream: stream.write(b'corrupt')
    # Re-persist the journal-corruption fixture in an exclusive NEW file path.
    if failure == 'journal':
        path = next(tmp_path.glob('*.physical.json'))
        original = json.loads(path.read_text())
        row['journal_sha256'] = original['journal_sha256']
        with Path(row['journal_path']).open('a') as stream: stream.write('\n')
    rows = persist_census(tmp_path, plans, row)
    before = {path:path.read_bytes() for path in tmp_path.glob('*.physical.json')}
    def setup():
        if failure == 'setup': raise RuntimeError('no host CUDA allocation')
        def encode(path):
            if failure == 'prefix' and path.name.endswith('_rgb_5.png'):
                raise RuntimeError('encoder error')
            return [0.]*192
        return encode
    receipts = p.encode_observations(rows, plans, tmp_path, setup)
    assert len(receipts) == 192
    assert all(path.read_bytes() == data for path, data in before.items())
    expected = dict(setup=('encoding_setup_failed', 'encoding_setup_failed'),
        prefix=('encoding_failed', 'completed'), journal=('blocked', 'blocked'), png=('blocked', 'completed'))
    assert tuple(r['status'] for r in receipts[:2]) == expected[failure]


def test_incomplete_or_changed_physics_blocks_encoder_setup(tmp_path, plans):
    def forbidden(): pytest.fail('encoder setup must not run')
    with pytest.raises(ValueError): p.encode_observations([], plans, tmp_path, forbidden)
    rows = persist_census(tmp_path, plans)
    rows[0]['status'] = 'changed'
    with pytest.raises(ValueError): p.encode_observations(rows, plans, tmp_path, forbidden)


def test_preflight_only_never_makes_environment_or_output(tmp_path, plans, monkeypatch, capsys):
    def forbidden(*args, **kwargs): pytest.fail('no environment/model permitted')
    base = SimpleNamespace(gym=SimpleNamespace(make=forbidden), require_cuda_allocation=forbidden)
    monkeypatch.setattr(p, 'prepare', lambda path:(base, None, None, None, {}, plans, None, None, {}))
    output = tmp_path/p.TAG
    p.run(planning_report=tmp_path/'report.json', output=output, preflight_only=True, encode_terminals=True)
    value = json.loads(capsys.readouterr().out)
    assert value['branches'] == 96 and value['census']['all96branchcensus']
    assert value['planned_env_steps'] == 960 and not output.exists()


def test_setup_failure_keeps_all96_before_encoding(tmp_path, plans, saved, monkeypatch):
    def fail(*args, **kwargs): raise RuntimeError('environment unavailable')
    base = SimpleNamespace(gym=SimpleNamespace(make=fail))
    monkeypatch.setattr(p, 'prepare', lambda path:(base, None, saved[1], None, {}, plans, None, None, {}))
    output = tmp_path/(p.TAG+'_setup_failure')
    with pytest.raises(RuntimeError, match='all 96 branches retained'):
        p.run(planning_report=tmp_path/'report.json', output=output, encode_terminals=True)
    report = json.loads((output/'report.json').read_text())
    assert len(report['rows']) == 96 and len(report['encodings']) == 192
    assert report['all96branchcensus']['all96branchcensus']
    assert not report['P1_admission'] and report['new_predict_calls'] == 0
    assert all(e['status'] == 'blocked' for e in report['encodings'])
    with pytest.raises(FileExistsError):
        p.run(planning_report=tmp_path/'report.json', output=output)


def test_output_basename_exclusive(tmp_path, plans, monkeypatch):
    monkeypatch.setattr(p, 'prepare', lambda path:(None, None, None, None, {}, plans, None, None, {}))
    with pytest.raises(ValueError, match='basename'):
        p.run(planning_report=tmp_path/'report.json', output=tmp_path/'another_consumer')


@pytest.fixture
def provenance(tmp_path, plans, monkeypatch):
    """Small sealed provenance graph; no checkpoint or simulator is constructed."""
    import sys
    from experiments import spectral_reachability as package
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal
    roots = list(dict.fromkeys(x['root_sha256'] for x in plans))
    rgb = np.zeros((224, 224, 3), np.uint8)
    rgb[10:12, 10:12] = [65, 105, 225]
    from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import sha_rgb
    upstream = dict(protocol=dict(root_seed=42), rows=[dict(root_sha256=r,
        history_rgb_sha256=sha_rgb(np.stack([rgb]*3)), goal_rgb_sha256=sha_rgb(rgb)) for r in roots])
    upstream_path, bank_path = tmp_path/'upstream.json', tmp_path/'bank.json'
    p.write(upstream_path, upstream)
    p.write(bank_path, dict(upstream_path=str(upstream_path)))
    intervention_dir = tmp_path/'intervention'; intervention_dir.mkdir()
    intervention_path = intervention_dir/'report.json'
    p.write(intervention_path, dict(protocol=dict(bank_path=str(bank_path),
        bank_sha256=p.sha(bank_path), upstream_sha256=p.sha(upstream_path))))
    compilation = tmp_path/'compilation.json'; p.write(compilation, {})
    policy = tmp_path/'policy'; policy.mkdir()
    bindings = {str(path):p.sha(path) for path in (compilation, intervention_path)}
    dep = tmp_path/'planning_dependency.py'
    p.write(dep, dict(source='fixture'))
    seals = [create_seal(files=[dep], args_json='{}', output_dir=tmp_path/f'planning_seal_{i}') for i in range(2)]
    protocol = dict(compilation=str(compilation), policy=str(policy), bindings=bindings, roots=roots)
    report_path = tmp_path/'report.json'
    p.write(report_path, dict(protocol=protocol, source_seals=seals))
    calls = []
    def validate(compilation_arg, policy_arg):
        calls.append((compilation_arg, policy_arg))
        assert all(p.sha(path) == value for path, value in bindings.items())
        return dict(protocol=dict(roots=roots)), deepcopy(bindings), {}
    monkeypatch.setattr(checked, 'validate_inputs', validate)
    monkeypatch.setattr(p, 'plans_from_saved', lambda report, source:deepcopy(plans))
    base = SimpleNamespace(physical=SimpleNamespace(_load_preflight=lambda path:json.loads(Path(path).read_text())))
    name = 'experiments.spectral_reachability.audit_pusht_d2p_goal_agent_intervention'
    monkeypatch.setitem(sys.modules, name, base)
    monkeypatch.setattr(package, 'audit_pusht_d2p_goal_agent_intervention', base, raising=False)
    return SimpleNamespace(report=report_path, base=base, checked=checked, upstream=upstream,
        seals=seals, dep=dep, calls=calls, bindings=bindings)


def test_prepare_official_validation_seals_report_and_rgb_refs(provenance):
    ctx = p.prepare(provenance.report)
    assert len(provenance.calls) == 1 and len(ctx[5]) == 96
    bindings = ctx[4]
    assert bindings[str(provenance.report)] == p.sha(provenance.report)
    for seal in provenance.seals:
        for name in ('manifest.json', 'args.json'):
            path = Path(seal['output_dir'])/name
            assert bindings[str(path)] == p.sha(path)
    assert len(ctx[7]) == 4
    assert all(len(r['history_rgb_sha256']) == len(r['goal_rgb_sha256']) == 64 for r in ctx[7].values())


@pytest.mark.parametrize('mutation', ['bindings', 'second_seal', 'dependency', 'bank', 'upstream', 'root_refs', 'no_seals'])
def test_prepare_provenance_mutations_rejected(provenance, monkeypatch, mutation):
    report = json.loads(provenance.report.read_text())
    if mutation == 'bindings': report['protocol']['bindings'] = {}
    elif mutation == 'second_seal': report['source_seals'][1]['manifest_sha256'] = 'changed'
    elif mutation == 'dependency':
        with provenance.dep.open('a') as stream: stream.write('changed')
    elif mutation == 'bank':
        path = provenance.report.parent/'bank.json'
        with path.open('a') as stream: stream.write(' ')
    elif mutation == 'upstream':
        path = provenance.report.parent/'upstream.json'
        with path.open('a') as stream: stream.write(' ')
    elif mutation == 'root_refs':
        bad = deepcopy(provenance.upstream); bad['rows'].pop()
        monkeypatch.setattr(provenance.base.physical, '_load_preflight', lambda path:bad)
    else: report['source_seals'] = []
    if mutation in ('bindings', 'second_seal', 'no_seals'):
        # Fixture rewrite only; the consumer itself always uses exclusive output.
        with provenance.report.open('w') as stream: json.dump(report, stream)
    with pytest.raises(ValueError): p.prepare(provenance.report)


def test_run_seals_consumer_helpers_test_doc_before_any_physics(tmp_path, plans, provenance, monkeypatch):
    env = Env()
    env.unwrapped = env
    env.variation_space = dict(agent=dict(color=SimpleNamespace(value=np.array([65,105,225]))))
    env.eval_state = lambda goal, state:(False, 0.)
    env.close = lambda:None
    base = provenance.base
    base.gym = SimpleNamespace(make=lambda *args, **kwargs:env)
    monkeypatch.setattr(provenance.checked, 'sources_before_allocate', lambda source, config:[provenance.dep])
    output = tmp_path/(p.TAG+'_seal_barrier')
    collected = []
    def collect(*args, **kwargs):
        seals = json.loads((output/'source_seals.json').read_text())
        from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
        paths = set()
        for seal in seals:
            verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
            manifest = json.loads((Path(seal['output_dir'])/'manifest.json').read_text())
            paths.update(entry['source'] for entry in manifest['files'])
        root = Path(p.__file__).resolve().parents[2]
        assert {str(Path(p.__file__)), str(Path(p.packet_physical.__file__)), str(Path(p.planning.__file__)),
            str(root/f'tests/test_{p.TAG}.py'), str(root/f'docs/{p.TAG}.md'), str(provenance.dep)} <= paths
        assert (output/'all96branchcensus.planned.json').exists()
        collected.append(True)
        raise RuntimeError('fixture stops before any real physical root')
    base.collect_nonterminal_boundary_contact_roots = collect
    with pytest.raises(RuntimeError, match='all 96 branches retained'):
        p.run(planning_report=provenance.report, output=output)
    assert collected == [True] and env.n == 0
    report = json.loads((output/'report.json').read_text())
    assert report['gates']['sources_unchanged'] and report['gates']['inputs_unchanged']
    assert len(report['rows']) == 96


def test_actual_sealed_v3_prepare_and_preflight_model_free(tmp_path, monkeypatch, capsys):
    root = Path(p.__file__).resolve().parents[2]
    report_path = root/'results/pusht_geometry_prefix_20261027_v3_cuda_v1/report.json'
    if not report_path.is_file():
        pytest.skip('fresh sealed v3 artifact is not installed in this checkout')
    import torch
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    def forbidden(*args, **kwargs):
        pytest.fail('real prepare/preflight must not construct an environment or model or allocate CUDA')
    monkeypatch.setattr(base.gym, 'make', forbidden)
    monkeypatch.setattr(base, 'load_compatible_pretrained', forbidden)
    monkeypatch.setattr(base, 'require_cuda_allocation', forbidden)
    cuda_before = torch.cuda.is_initialized()
    ctx = p.prepare(report_path)
    assert len(ctx[5]) == 96 and len(ctx[7]) == 4
    assert ctx[4][str(report_path)] == p.sha(report_path)
    assert Path(p.planning.__file__).name == 'pusht_geometry_prefix_20261027_v3.py'
    assert (root/f'docs/{p.TAG}.md').is_file()
    output = tmp_path/(p.TAG+'_real_preflight')
    p.run(planning_report=report_path, output=output, preflight_only=True)
    # Imported dependencies may log before the CLI's final one-line JSON.
    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt['status'] == 'preflight_only' and receipt['census']['all96branchcensus']
    assert len(receipt['census']['branch_ids']) == 96
    assert receipt['new_predict_calls'] == 0 and not output.exists()
    assert torch.cuda.is_initialized() == cuda_before
