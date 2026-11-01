"""RAM-only: artificial transitions and saved-interface fixtures, no simulator."""
from copy import deepcopy
import io
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_baseline_physical_20261027_v1 as m
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv, fixture as old_fixture, REQUEST
from test_p1_constructive_plan_physical_20261027_v1 import MemoryStore


ALL = [a for _, _, arms, _ in m.AUTHORITIES.values() for a in arms]
GRAD = 'moving_gradient_open_loop'


def fixture(arms=None):
    arms = [GRAD] if arms is None else arms
    _, data = old_fixture()
    plans = []
    for s in (0, 1):
        for a in arms:
            row = dict(root_slot=s, arm=a, commands=np.full((10, 2), .25, np.float32).tolist())
            row['selection_digest'] = m.source.digest(row); plans.append(row)
    p = dict(plans=plans, contract=m.contract(arms, 600), bindings={}, sourcepack_sha256=m.SOURCE_SHA)
    return p, data


def execute(factory=ArrayEnv, *, arms=None, p=None, data=None, store=None):
    fp, fd = fixture(arms); p = fp if p is None else p; data = fd if data is None else data
    store = MemoryStore() if store is None else store; envs = []
    def make():
        e = factory(); envs.append(e); return e
    def fake_sha(path):
        assert Path(path).name == 'journal.jsonl'
        return m.source.digest(store.events)
    forbidden = AssertionError('old execute/source episode forbidden')
    with patch.object(m.kernel.moving, 'validate_env', lambda _: None), \
         patch.object(m.source, 'sample_state', lambda *a: REQUEST.copy()), \
         patch.object(m, 'sha', fake_sha), patch.object(m.kernel, 'execute', side_effect=forbidden), \
         patch.object(m.kernel.moving, 'episode', side_effect=forbidden), \
         patch.object(m.kernel.moving, 'witness', side_effect=forbidden):
        r = m.execute(p, data, store, make, authorization=m.AUTH)
    return r, store, envs


@pytest.mark.parametrize('arms', [[GRAD], ALL])
def test_fixed_denominator_actual_dispatch_and_counts(arms):
    r, store, envs = execute(arms=arms); n = 2*len(arms)
    assert r['status'] == 'moving_baseline_physical_recorded', r
    counts = dict(reset=2*n, public_step=20*n, explicit_render=22*n, goal_render=2*n,
                  set_state=4*n, physics_step=204*n, goal_bind=n)
    assert r['counts'] == {k: dict(attempted=v, completed=v) for k, v in counts.items()}
    assert r['fixed_branch_denominator'] == len(r['rows']) == n
    assert r['full_horizons'] == n and r['new_neural_calls'] == 0 and r['P1_admission'] is False
    for row, env in zip(r['rows'], envs, strict=True):
        assert env.closed and row['initial_exact'] and row['full_horizon']
        assert m.kernel.moving.exact(np.asarray(env.commands[:10]), m.kernel.moving.warmup_commands())
        assert m.kernel.moving.exact(np.asarray(env.commands[10:]), np.full((10, 2), .25, np.float32))
        assert [x['step'] for x in row['checkpoints']] == [0, 5, 10]
        assert np.array_equal(env.goal_pose, [300, 300, 0])
    previous = None
    for event in store.events:
        assert event['previous_sha256'] == previous
        assert event['event_sha256'] == m.source.digest({k: v for k, v in event.items() if k != 'event_sha256'})
        previous = event['event_sha256']
    assert r['journal_chain_tail'] == previous


@pytest.mark.parametrize('fault', ['truncate', 'terminal'])
def test_native_stop_retains_partial_and_actual_counts(fault):
    r, _, envs = execute(lambda: ArrayEnv(fault=fault))
    assert r['status'] == 'moving_baseline_physical_recorded', r
    assert all(len(e.commands) == 11 for e in envs)
    assert r['full_horizons'] == 0 and r['counts']['physics_step']['completed'] == 228
    assert all(row['own_returned'] == 1 and not row['full_horizon'] for row in r['rows'])


@pytest.mark.parametrize('fault', ['guard', 'render', 'substitute'])
def test_guard_or_substitution_keeps_fresh_frame_before_reject(fault):
    r, s, envs = execute(lambda: ArrayEnv(fault=fault))
    assert r['status'] == 'partial_or_failed' and len(r['rows']) == 2
    assert all(len(e.commands) == 11 for e in envs)
    assert all(row['last_observation']['png'] in s.files for row in r['rows'])


def test_primary_step_survives_close_and_no_retry():
    r, _, envs = execute(lambda: ArrayEnv(fault='throw', close_error=True))
    assert len(envs) == 2 and r['status'] == 'partial_or_failed'
    assert all('primary step' in row['first_error']['message'] and 'close' in str(row['secondary_errors']) for row in r['rows'])
    assert r['counts']['public_step'] == dict(attempted=22, completed=20)


@pytest.mark.parametrize('field', ['history', 'states', 'raw', 'snapshot'])
def test_root_reconstruction_precedes_any_plan(field):
    p, d = fixture(); ref = d['references'][0]
    if field == 'history': ref['history'][1, 0, 0, 0] ^= 1
    elif field == 'states': ref['states'][1, 0] += 1e-8
    elif field == 'raw': ref['raw_history'] = np.zeros((10, 2), np.float32)
    else: ref['dynamics'][1]['state_float64_bits'] = 'wrong'
    r, _, _ = execute(p=p, data=d)
    assert not r['rows'][0]['initial_exact'] and r['rows'][0]['own_attempted'] == 0
    assert r['rows'][1]['full_horizon']


@pytest.mark.parametrize('stage', ['branch_closed', 'run_finished', 'report'])
def test_durable_failures_never_complete(stage):
    s = MemoryStore(); emit, put = s.emit, s.put
    def newemit(kind, **fields):
        if kind == stage: raise OSError('mock journal fsync failure')
        return emit(kind, **fields)
    def newput(name, value, final=False):
        if stage == 'report' and final: raise OSError('mock final fsync failure')
        return put(name, value, final)
    s.emit, s.put = newemit, newput
    r, _, envs = execute(store=s)
    assert r['status'] == 'partial_or_failed'
    if stage == 'branch_closed':
        assert len(envs) == 1 and r['rows'][1]['status'] == 'not_started'


def manifest_fixture(arms=None):
    arms = [GRAD] if arms is None else arms
    p, data = fixture(arms)
    authorities = [dict(authority=k, output='/MOCK/'+k, report_sha256='c'*64)
                   for k, (_, _, names, _) in m.AUTHORITIES.items() if any(a in arms for a in names)]
    manifest = dict(schema=m.NAME+'.selection', sourcepack='/MOCK/sourcepack.json', sourcepack_sha256=m.SOURCE_SHA,
                    source_prepared='/MOCK/source_prepare.json', source_prepare_sha256='b'*64,
                    roots=[dict(root_slot=i, root_sha256=r['root_sha256']) for i, r in enumerate(data['references'])],
                    arms=arms, authorities=authorities, selection_fixed_before_physics=True,
                    physical_outcome_selection=False, wall_seconds=600)
    rows = {}
    for e in authorities:
        rows[e['authority']] = [dict(source_slot=s, arm=a, commands=np.full((10, 2), .25, np.float32).tolist(),
                                    task_id='t'+str(s), blind_sha256=data['references'][s]['blind_sha256'],
                                    task_export='/MOCK/task_'+str(s)+'.json.gz', task_sha256='f'*64)
                                for s in (0, 1) for a in m.AUTHORITIES[e['authority']][2] if a in arms]
    return manifest, data, rows


def select(manifest, data, rows):
    with patch.object(m, 'ownconsume', side_effect=lambda e: (deepcopy(rows[e['authority']]), {})), \
         patch.object(m, 'task_bridge'):
        return m.selection(manifest, data)


@pytest.mark.parametrize('arms', [[GRAD], ALL, ['CEM'], ['kernel_faithful_mppi_nominal']])
def test_actual_api_shape_adapter_fixed_order(arms):
    manifest, data, rows = manifest_fixture(arms); plans, _ = select(manifest, data, rows)
    assert [(x['root_slot'], x['arm']) for x in plans] == [(s, a) for s in (0, 1) for a in arms]
    assert all(x['selection_digest'] == m.source.digest({k: v for k, v in x.items() if k != 'selection_digest'}) for x in plans)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'slot', 'arm', 'outofbounds', 'rounded', 'roots', 'outcome', 'extraauthority'])
def test_selection_rejects_no_dispatch(fault):
    manifest, data, rows = manifest_fixture(); values = rows['gradient']
    if fault == 'missing': values.pop()
    elif fault == 'duplicate': values.append(deepcopy(values[0]))
    elif fault == 'slot': values[0]['source_slot'] = True
    elif fault == 'arm': values[0]['arm'] = 'GN'
    elif fault == 'outofbounds': values[0]['commands'][0][0] = 1.1
    elif fault == 'rounded': values[0]['commands'][0][0] = .1
    elif fault == 'roots': manifest['roots'].reverse()
    elif fault == 'outcome': manifest['physical_outcome_selection'] = True
    else: manifest['authorities'].append(dict(authority='unapproved', output='/MOCK', report_sha256='f'*64))
    with pytest.raises(ValueError): select(manifest, data, rows)


def test_unknown_authority_never_imported():
    with patch.object(m.importlib, 'import_module', side_effect=AssertionError('must not import')):
        with pytest.raises(ValueError):
            m.ownconsume(dict(authority='arbitrary.module', output='/MOCK', report_sha256='f'*64))
        with patch.object(m, 'sha', return_value='wrong'), pytest.raises(ValueError):
            m.ownconsume(dict(authority='gradient', output='/MOCK', report_sha256='f'*64))


@pytest.mark.parametrize('authority', list(m.AUTHORITIES))
def test_ownconsume_uses_actual_signature_and_return_key(authority):
    name, expected, arms, key = m.AUTHORITIES[authority]
    path = m.ROOT/'experiments/spectral_reachability'/(name+'.py')
    called = []; payload = {key: [], 'schema': name+'.consume', 'P1_admission': False}
    mod = NS(__file__=str(path), consume=lambda directory, h: (called.append((directory, h)) or payload))
    report = dict(prepared='/MOCK/prepare.json', prepare_sha256='p'*64, artifact_sha256={})
    def hashfile(p): return expected if Path(p) == path else 'p'*64 if str(p).endswith('prepare.json') else 'r'*64
    with patch.object(m, 'sha', side_effect=hashfile), patch.object(m, 'read', side_effect=[report, dict(bindings_sha256={})]), \
         patch.object(m.importlib, 'import_module', return_value=mod):
        rows, bindings = m.ownconsume(dict(authority=authority, output='/MOCK/run', report_sha256='r'*64))
    assert rows == [] and called == [(Path('/MOCK/run'), 'r'*64)] and str(path.resolve()) in bindings


def test_task_bridge_exact_pixels_raw_and_goal_without_model():
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as enc
    _, d = fixture(); ref = d['references'][0]
    plan = dict(source_slot=0, blind_sha256=ref['blind_sha256'], task_id='task',
                task_export='/MOCK/task_00.json.gz', task_sha256='f'*64)
    tc = dict(pixel_sha256=[m.kernel.moving.image_sha(a) for a in [*ref['history'], ref['goal_rgb']]],
              raw_action_sha256=m.kernel.moving.image_sha(ref['raw_history']))
    envelope = dict(source_slot=0, blind_sha256=ref['blind_sha256'], task_contract=tc)
    interface = dict(task_id='task', gn_task=dict(root_raw_action_history=ref['raw_history'].astype(float).tolist()))
    with patch.object(enc, 'load_task', return_value=interface), patch.object(enc, 'read', return_value=envelope):
        bindings = {}; m.task_bridge(plan, ref, bindings)
        assert bindings[plan['task_export']] == 'f'*64
        tc['pixel_sha256'][-1] = 'badgoal'
        with pytest.raises(ValueError): m.task_bridge(plan, ref, {})


def test_authorization_and_budget_predispatch():
    p, d = fixture(); store = MemoryStore(); ops = m.kernel.Ops(p, store)
    p['contract']['caps']['public_step'] = 0
    with pytest.raises(ValueError): ops.call('public_step', lambda: pytest.fail('dispatched'))
    with pytest.raises(ValueError): m.execute(p, d, store, lambda: pytest.fail('factory'), authorization='wrong')
    with pytest.raises(ValueError): m.run('none', 'none', 'none', lambda: None, authorization='wrong')
    x = object.__new__(m.Store); x.path = NS(iterdir=lambda: [])
    with patch.object(m.shutil, 'disk_usage', return_value=NS(free=m.FLOOR-1)), pytest.raises(ValueError): x.check(1)


def test_store_write_fsync_failure_in_ram_only():
    class FakePath:
        def __init__(self): self.buffer = io.BytesIO()
        def iterdir(self): return []
        def __truediv__(self, name): return self
        def is_symlink(self): return False
        def open(self, mode):
            class File(io.BytesIO):
                def fileno(self): return 123
            return File()
    store = object.__new__(m.Store); store.path = FakePath()
    with patch.object(m.shutil, 'disk_usage', return_value=NS(free=m.FLOOR+m.CAP)), \
         patch.object(m.os, 'fsync', side_effect=OSError('mock fsync')):
        with pytest.raises(OSError, match='fsync'): store.write('receipt.json', b'{}')


def test_eight_branch_complete_ram_png_journal_capacity():
    from PIL import Image
    r, store, _ = execute(arms=ALL)
    sizes = {}
    for name, value in store.files.items():
        if isinstance(value, np.ndarray):
            buffer = io.BytesIO(); Image.fromarray(value).save(buffer, format='PNG')
            sizes[name] = len(buffer.getvalue())
        else:
            sizes[name] = len(value)
    sizes['journal.jsonl'] = sum(len(m.raw(e)) for e in store.events)
    assert len([n for n in sizes if n.endswith('.png')]) == 176
    assert sum(sizes.values()) < m.CAP
    assert len(m.raw(r)) < m.FINAL


def test_local_static_closure_includes_actual_kernel_and_loader():
    paths = m.physical_files()
    assert Path(m.kernel.__file__).resolve() in paths
    assert Path(m.kernel.low.__file__).resolve() in paths
    assert Path(m.kernel.moving.__file__).resolve() in paths
    assert any(p.name == 'p1_constructive_moving_history_encode_cuda_20261027_v1.py' for p in paths)
    assert all(p.is_relative_to(m.ROOT/'experiments') and p.suffix == '.py' for p in paths)


def test_verify_rebuild_and_tamper_ram_only():
    p = dict(manifest='/MOCK/selection.json', manifest_sha256='a'*64)
    with patch.object(m.Path, 'stat', return_value=NS(st_size=100)), patch.object(m, 'sha', return_value='b'*64), \
         patch.object(m, 'read', return_value=deepcopy(p)), patch.object(m, 'material', return_value=deepcopy(p)):
        assert m.verify('/MOCK/prepared.json', 'b'*64) == p
        with patch.object(m, 'material', return_value=dict(p, tampered=True)), pytest.raises(ValueError):
            m.verify('/MOCK/prepared.json', 'b'*64)


def test_z_import_no_model_or_simulator():
    assert not any(n in sys.modules for n in ('torch', 'gymnasium', 'pymunk', 'stable_worldmodel'))
