"""Algebra/durability fixtures only: no neural allocation or real simulator."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import pusht_geometry_capability_20261027_v1 as p


@pytest.fixture(scope='module')
def saved():
    g = p.planning
    goal, q, jac, origin = np.ones(192), np.eye(192), np.eye(192, 10), np.zeros(192)
    def measured(params):
        return dict(prefix_params=np.asarray(params, dtype=np.float32).astype(float).tolist(),
            predicted_prefix=goal.tolist(), nominal_terminal=origin.tolist(), jacobian=jac.tolist())
    def probes(center, epsilons):
        values = [measured(center)]
        for eps in epsilons:
            for axis in range(10):
                for sign in (-1, 1):
                    delta = np.asarray(center, dtype=np.float32).copy(); delta[axis] += sign*eps
                    values.append(measured(delta))
        return values
    records = probes(np.zeros(10), g.EPSILONS)
    common = g.packet.spectral_bank(jac, origin, goal, q)
    roots = [f'root{i}' for i in range(4)]
    rows = []
    for root_index, root in enumerate(roots):
        arms = []
        for arm in g.ARMS:
            runs = []
            for center in g.starts(common):
                traces = []
                for iteration in range(g.ITERATIONS):
                    values = probes(center, g.EPSILONS[:1])
                    gradients = g.gradients_from_records(values, goal, q, center=center, epsilons=g.EPSILONS[:1])
                    nxt = g.update(center, gradients[0]['gradients'][arm])
                    traces.append(dict(iteration=iteration, center_params=center, probe_records=values,
                        gradients=gradients, next_params=nxt))
                    center = nxt
                runs.append(traces)
            bank = g.augmented_bank(common, runs, goal, q, arm)
            candidates = []
            for i, params in enumerate(bank['params']):
                m = measured(params)
                if root_index == 0 and i == 11:
                    m['nominal_terminal'] = goal.tolist()
                score = g.objective(m, goal, q)
                suffix = np.asarray(score['suffix_solve']['params'], dtype=np.float32)
                candidates.append(dict(**m, **score, candidate_id=f'prefix_{i}', kind=bank['kinds'][i],
                    suffix_params=suffix.astype(float).tolist(),
                    commands=g.reference_commands(m['prefix_params'], suffix.astype(float)).tolist(),
                    predicted_terminal=m['nominal_terminal'], independent_model_replay_exact=True))
            arms.append(dict(arm=arm, bank=bank, optimization=runs, candidates=candidates,
                selected_candidate_id='prefix_11' if root_index == 0 else 'prefix_0', greedy_candidate_id='prefix_0'))
        rows.append(dict(root_index=root_index, root_sha256=root, arms=arms, root_jacobian=jac.tolist(),
            root_nominal=origin.tolist(), probe_records=records,
            gradient_grids=g.gradients_from_records(records, goal, q)))
    report = dict(schema=g.TAG, status='plans_sealed', gates=dict(all96=True), plans=rows,
        protocol=dict(roots=roots, arms=list(g.ARMS), epsilons=list(g.EPSILONS), damping=.002,
            cap=g.packet.CAP, radii=list(g.packet.RADII), iterations=g.ITERATIONS, step=g.STEP, candidates_per_arm=12))
    source = dict(protocol=dict(roots=roots), rows=[dict(root_sha256=r,
        compiled=dict(goal_vectors=[goal.tolist()], scoring=dict(metric=dict(projection=q.tolist())))) for r in roots])
    return report, source


@pytest.fixture(scope='module')
def plans(saved):
    return p.plans_from_saved(*saved)


def test_real_v3_verifier_model_only_selection_and_aliases(saved, plans, monkeypatch):
    snapshot = deepcopy(saved); calls = []
    real = p.planning.verify_plans
    def verify(*args):
        calls.append(True)
        return real(*args)
    monkeypatch.setattr(p.planning, 'verify_plans', verify)
    assert p.plans_from_saved(*saved) == plans
    assert calls == [True] and saved == snapshot
    assert len(plans) == 123 and {v['root_sha256'] for v in plans} == {'root0'}
    assert sum(v['role'] == 'greedy' for v in plans) == 41
    greedy = next(v for v in plans if v['method'] == 'greedy_zero')
    zero = next(v for v in plans if v['method'] == 'zero_zero')
    assert greedy['aliases'] == ['greedy','zero'] and greedy['commands'] == zero['commands']
    assert greedy['selection_sha256'] != zero['selection_sha256']


@pytest.mark.parametrize('mutation', ['failed', 'v2', 'gates', 'index', 'commands', 'selection', 'nan', 'root_order'])
def test_fail_closed_before_qualification(saved, mutation):
    report, source = deepcopy(saved)
    if mutation == 'failed': report['status'] = 'integrity_fail'
    elif mutation == 'v2': report['schema'] = 'pusht_geometry_prefix_20261027_v2'
    elif mutation == 'gates': report['gates'] = {}
    elif mutation == 'index': report['plans'][0]['root_index'] = True
    elif mutation == 'commands': report['plans'][0]['arms'][0]['candidates'][0]['commands'][0][0] = 2.
    elif mutation == 'selection': report['plans'][0]['arms'][0]['selected_candidate_id'] = 'prefix_0'
    elif mutation == 'nan': report['plans'][0]['arms'][0]['candidates'][0]['nominal_terminal'][0] = float('nan')
    else: report['protocol']['roots'].reverse()
    with pytest.raises((ValueError, KeyError)):
        p.plans_from_saved(report, source)


def test_fullpacket_41_suffixes_tanh_and_zero_tail(plans):
    bank = p.suffix_bank()
    assert len(bank) == 41 and not np.any(bank[0]['logits'])
    assert {v['epsilon'] for v in bank[1:]} == {.05, .1}
    for role in p.ROLES:
        roleplans = [v for v in plans if v['role'] == role]
        prefixes = [v['commands'][:5] for v in roleplans]
        assert all(x == prefixes[0] for x in prefixes)
        for plan, b in zip(roleplans, bank, strict=True):
            assert np.array_equal(np.asarray(plan['commands'])[5:], np.tanh(b['logits']).reshape(5,2))
            assert len(plan['commands']) == 10
        for grid in range(2):
            signed = np.asarray([v['probe']['logits'] for v in roleplans[1+20*grid:21+20*grid]])
            assert np.linalg.matrix_rank(signed) == 10
            assert np.array_equal(signed[::2], -signed[1::2])


def test_disk_gate_limits_reserve_and_no_writes(tmp_path):
    out = tmp_path/p.TAG
    assert p.disk_guard(out, 123, free_bytes=2*1024**3)['planned_bytes'] == 433*1024**2
    with pytest.raises(ValueError, match='disk gate'):
        p.disk_guard(out, 492, free_bytes=20*1024**3)
    with pytest.raises(ValueError, match='reserve'):
        p.disk_guard(out, 123, free_bytes=900*1024**2)
    assert not out.exists()


def test_preflight_and_disk_failure_never_construct_env(tmp_path, plans, saved, monkeypatch, capsys):
    def forbidden(*args, **kwargs): pytest.fail('no real environment or CUDA permitted')
    base = SimpleNamespace(gym=SimpleNamespace(make=forbidden), require_cuda_allocation=forbidden)
    monkeypatch.setattr(p, 'prepare', lambda path:(base, None, saved[1], None, {}, plans, None, None, {}))
    monkeypatch.setattr(p.shutil, 'disk_usage', lambda path:SimpleNamespace(free=2*1024**3))
    out = tmp_path/p.TAG
    p.run(planning_report=tmp_path/'report.json', output=out, preflight_only=True, encode_terminals=True)
    result = json.loads(capsys.readouterr().out)
    assert result['census']['planned_steps'] == 1230 and not out.exists()
    monkeypatch.setattr(p.shutil, 'disk_usage', lambda path:SimpleNamespace(free=800*1024**2))
    with pytest.raises(ValueError, match='disk gate'):
        p.run(planning_report=tmp_path/'report.json', output=out)
    assert not out.exists()


class Env:
    """Contract fixture, never a simulator."""
    def __init__(self, stop=None, terminate=None, fail=None):
        self.n, self.stop, self.terminate, self.fail = 0, stop, terminate, fail
    def render(self):
        rgb = np.zeros((224,224,3), np.uint8)
        if self.n != self.stop:
            rgb[10:12,10:12] = [65,105,225]
        return rgb
    def step(self, command):
        if self.n+1 == self.fail:
            raise RuntimeError('fixture step failure')
        self.n += 1
        return None, 0., self.n == self.terminate, False, {}


def physical_fixture(tmp_path, plans, env):
    from experiments.spectral_reachability import pusht_full_packet_physical_20261027 as helper
    row = p.durable.physical_branch(plan=plans[0], repetition=0, output=tmp_path, reset=lambda:{}, env=env,
        state_of=lambda e:np.array([0.,0.,e.n,2.,.1]), evaluate=lambda state:dict(official={}, pose={}))
    rows = [row]
    helper.materialize_missing(plans, rows, tmp_path)
    p.write(tmp_path/'physical_report.json', dict(rows=rows))
    slots = p.endpoint_slots(plans, rows, tmp_path, helper)
    return helper, rows, slots


@pytest.mark.parametrize('stop,expected', [(0,(False,False)),(3,(False,False)),(5,(True,False)),
    (7,(True,False)),(10,(True,True)),(None,(True,True))])
def test_exact_step_slots_and_all_receipts_before_encoder(tmp_path, plans, stop, expected):
    helper, rows, slots = physical_fixture(tmp_path, plans, Env(stop=stop))
    assert len(slots) == 246 and len(list(tmp_path.glob('*.physical.json'))) == 123
    assert tuple(s['status'] == 'observed' for s in slots[:2]) == expected
    paths = []
    def setup():
        assert json.loads((tmp_path/'endpoint_receipts.json').read_text()) == slots
        return lambda path: paths.append(path.name) or [0.]*192
    enc = helper.encode_observations(rows, plans, tmp_path, setup)
    assert len(enc) == 246
    assert tuple(e['comparable'] for e in enc[:2]) == expected
    assert paths == [f'root_0_selected_zero_replay0_rgb_{step}.png' for step, ok in zip((5,10), expected) if ok]


def test_earlytermination_failed_attempt_and_durable_journal(tmp_path, plans):
    helper, rows, slots = physical_fixture(tmp_path, plans, Env(terminate=7))
    assert rows[0]['completed_commands'] == 7 and not rows[0]['full_horizon_completed']
    assert slots[0]['status'] == 'observed' and slots[1]['status'] == 'missing'
    journal = Path(rows[0]['journal_path'])
    assert p.sha(journal) == rows[0]['journal_sha256']
    events = [json.loads(s) for s in journal.read_text().splitlines()]
    assert sum(e['kind'] == 'attempt' for e in events) == 7
    other = tmp_path/'failed'; other.mkdir()
    _, rows, slots = physical_fixture(other, plans, Env(fail=4))
    assert rows[0]['physical_env_step_attempts'] == 4 and rows[0]['physical_env_step_returns'] == 3
    assert all(s['status'] == 'missing' for s in slots)


@pytest.mark.parametrize('mutation', ['drop','duplicate','hash','horizon'])
def test_census_rejects_incomplete_and_substitution(plans, mutation):
    values = deepcopy(plans)
    if mutation == 'drop': values.pop()
    elif mutation == 'duplicate': values[-1] = values[0]
    elif mutation == 'hash': values[0]['selection_sha256'] = 'changed'
    else: values[0]['planned_commands'] = 5
    with pytest.raises(ValueError): p.census(values)


def test_phi_uses_block_pose_and_not_agent():
    assert np.allclose(p.phi([99.,88.,512.,256.,np.pi/2]), [1.,.5,0.,20/512])
    assert np.array_equal(p.phi([1.,2.,3.,4.,0.]), p.phi([9.,8.,3.,4.,0.]))
    with pytest.raises(ValueError): p.phi([0.,1.])


def analysis_fixture(plans):
    slots, enc = [], []
    # Different own baselines and different linear response per role.
    factors = dict(selected=2., greedy=1., zero=.5)
    offsets = dict(selected=.2, greedy=.4, zero=.6)
    for plan in plans:
        u = np.asarray(plan['probe']['logits'])
        state = [9.,8.,256.+factors[plan['role']]*u[0],256.+factors[plan['role']]*u[1],0.]
        vector = np.full(192, offsets[plan['role']]); vector[:10] += factors[plan['role']]*u
        slots.append(dict(root_sha256='root0', method=plan['method'], kind='terminal', status='observed', phi=p.phi(state).tolist()))
        enc.append(dict(root_sha256='root0', method=plan['method'], kind='terminal', status='completed', comparable=True, vector=vector.tolist()))
    metrics = {'root0':(np.ones(192), 2*np.eye(192))}
    goals = {'root0':[0.,0.,300.,300.,0.]}
    return slots, enc, metrics, goals


def test_real_fd_Q_common_target_ridge_and_sensitivity(plans):
    result = p.response_analysis(plans, *analysis_fixture(plans))
    assert len(result['responses']) == 6 and len(result['selected_vs_baselines']) == 4
    for space in ('real_Q', 'physical_phi'):
        rows = [r for r in result['responses'] if r['space'] == space]
        assert all(r['status'] == 'completed' for r in rows)
        assert rows[0]['common_residual'] == rows[1]['common_residual'] == rows[2]['common_residual']
        if space == 'real_Q':
            assert rows[0]['own_residual'] != rows[1]['own_residual']
        for r in rows:
            assert r['epsilon_sensitivity']['response_frobenius_delta'] < 1e-10
            for grid in r['grids']:
                assert len(grid['paired_logs']) == 10
                assert grid['own_fit']['radius'] == np.sqrt(5)*.75
                assert grid['own_fit']['norm'] <= np.sqrt(5)*.75+1e-12
                k = np.asarray(grid['response'])
                for fit, residual in (('own_fit','own_residual'),('common_zero_role_fit','common_residual')):
                    assert grid[fit] == p.planning.packet.bounded_ridge(k, r[residual])
    selected = result['responses'][0]
    assert np.allclose(np.asarray(selected['grids'][0]['raw_jacobian']), 2*np.eye(192,10))
    assert np.allclose(np.asarray(selected['grids'][0]['response']), 4*np.eye(192,10))


def test_missing_terminal_or_encode_never_pads(plans):
    slots, enc, metrics, goals = analysis_fixture(plans)
    slots[1]['status'] = 'missing'
    enc[2]['status'] = 'encoding_failed'
    result = p.response_analysis(plans, slots, enc, metrics, goals)
    selected = next(r for r in result['responses'] if r['role'] == 'selected' and r['space'] == 'real_Q')
    assert selected['status'] == 'blocked'
    assert selected['grids'][0]['status'] == 'blocked'
    assert selected['grids'][1]['status'] == 'completed'
    assert all(r['status'] == 'completed' for r in result['responses'] if r['role'] != 'selected')


def test_failed_physical_setup_retains_slots_no_cuda(tmp_path, saved, plans, monkeypatch):
    def fail(*args, **kwargs): raise RuntimeError('fixture setup fails')
    base = SimpleNamespace(gym=SimpleNamespace(make=fail))
    monkeypatch.setattr(p, 'prepare', lambda path:(base, None, saved[1], None, {}, plans, None, None, {}))
    monkeypatch.setattr(p.shutil, 'disk_usage', lambda path:SimpleNamespace(free=2*1024**3))
    out = tmp_path/p.TAG
    with pytest.raises(RuntimeError, match='all planned slots retained'):
        p.run(planning_report=tmp_path/'report.json', output=out)
    report = json.loads((out/'report.json').read_text())
    assert len(report['rows']) == 123 and len(report['endpoint_receipts']) == 246
    assert report['new_predict_calls'] == 0 and not report['P1_admission']
    assert all(s['status'] == 'missing' for s in report['endpoint_receipts'])
    assert all(r['status'] == 'blocked' for r in report['analysis']['responses'])


def test_existing_v3_model_artifact_only_no_physical_outcomes():
    """Optional real MODEL receipt regression; never call prepare or read physics."""
    root = Path(p.__file__).resolve().parents[2]
    path = root/'results/pusht_geometry_prefix_20261027_v3_cuda_v1/report.json'
    if not path.is_file():
        pytest.skip('sealed model-only v3 artifact not available')
    report = json.loads(path.read_text())
    if report.get('status') != 'plans_sealed':
        pytest.skip('upstream model planning not yet sealed')
    compilation_path = Path(report['protocol']['compilation'])
    if not compilation_path.is_absolute():
        compilation_path = root/compilation_path
    source = json.loads(compilation_path.read_text())
    plans = p.plans_from_saved(report, source)
    expected = []
    for row in report['plans']:
        arm = next(a for a in row['arms'] if a['arm'] == 'geometry')
        if arm['selected_candidate_id'] != arm['greedy_candidate_id']:
            expected.append(row['root_sha256'])
    assert set(p.census(plans)['qualified_roots']) == set(expected)
    assert len(plans) == 123*len(expected)
    for plan in plans:
        row = next(r for r in report['plans'] if r['root_sha256'] == plan['root_sha256'])
        arm = next(a for a in row['arms'] if a['arm'] == 'geometry')
        assert plan['role_candidate_ids'] == dict(selected=arm['selected_candidate_id'],
            greedy=arm['greedy_candidate_id'], zero='prefix_0')
