"""RAM arrays/callbacks only; not physical or scientific evidence. No real prepare."""
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import moving_prediction_matched_sampling_physical_20261027_v1 as m
from test_p1_constructive_moving_history_physical_20261027_v1 import ArrayEnv, fixture as moving_fixture, REQUEST
from test_p1_constructive_plan_physical_20261027_v1 import MemoryStore


def fixture():
    _, data = moving_fixture(); plans = []
    for s in (0, 1):
        for a in m.ARMS:
            p = dict(root_slot=s, root_sha256=data['references'][s]['root_sha256'], arm=a,
                     commands=np.full((10, 2), .25, np.float32).tolist())
            p['selection_digest'] = m.source.digest(p); plans.append(p)
    return dict(plans=plans, contract=m.contract(), bindings={}, sourcepack_sha256=m.SOURCE_SHA,
                source_prepare_id='b'*64), data


def execute(factory=ArrayEnv, *, p=None, data=None, store=None):
    fp, fd = fixture(); p = fp if p is None else p; data = fd if data is None else data
    store = MemoryStore() if store is None else store; envs = []
    def make(): e = factory(); envs.append(e); return e
    def digest(path):
        assert Path(path).name == 'journal.jsonl'
        return m.source.digest(store.events)
    with patch.object(m.kernel.moving, 'validate_env', lambda _: None), \
         patch.object(m.source, 'sample_state', lambda *a: REQUEST.copy()), patch.object(m, 'sha', digest), \
         patch.object(m.helper, 'execute', side_effect=AssertionError('old execute forbidden')), \
         patch.object(m.kernel.moving, 'episode', side_effect=AssertionError('source episode forbidden')), \
         patch.object(m.kernel.moving, 'witness', side_effect=AssertionError('witness forbidden')):
        result = m.execute(p, data, store, make, authorization=m.AUTH,
                           execution_binding=dict(prepared='/RAM/prepare.json', prepare_sha256='a'*64))
    return result, store, envs


def test_complete_four_NEW_branches_even_identical_commands():
    r, store, envs = execute()
    assert r['status'] == 'matched_sampling_physical_recorded_not_admitted', r['errors']
    c = dict(reset=8, set_state=16, goal_render=8, goal_bind=4, public_step=80, physics_step=816,
             explicit_render=88, environment_construct=4, environment_close=4)
    assert r['counts'] == {k: dict(attempted=v, completed=v) for k, v in c.items()}
    assert len(envs) == r['fixed_branch_denominator'] == r['full_horizons'] == 4
    assert r['engineering_failures'] == r['early_stops'] == r['not_started'] == 0
    for row, env in zip(r['rows'], envs, strict=True):
        assert env.closed and len(env.commands) == 20 and row['initial_exact']
        assert m.kernel.moving.exact(np.asarray(env.commands[:10]), m.kernel.moving.warmup_commands())
        assert m.kernel.moving.exact(np.asarray(env.commands[10:]), np.asarray(row['selected_plan']['commands'], np.float32))
        assert [c['step'] for c in row['checkpoints']] == [0, 5, 10]
        assert all(len(c['state']) == 7 and c['png'] in store.files for c in row['checkpoints'])
        assert row['own_attempted'] == row['own_returned'] == row['warmup_returned'] == 10
        assert len(row['final_state']) == 7 and row['final_metrics']['object_success'] is None
        assert np.array_equal(env.goal_pose, [300., 300., 0.])
        assert 'reset' not in env.__dict__ and '_set_state' not in env.__dict__
    for k in ('P1_admission', 'P2_admission', 'P3_admission', 'scientific_admission', 'independent_confirmation',
              'equal_FLOPs', 'equal_total_compute', 'compute_advantage', 'walltime_advantage'):
        assert r[k] is False
    assert r['new_neural_calls'] == r['new_independent_samples'] == 0
    previous = None
    for e in store.events:
        assert e['previous_sha256'] == previous
        assert e['event_sha256'] == m.source.digest({k: v for k, v in e.items() if k != 'event_sha256'})
        previous = e['event_sha256']
    assert previous == r['journal_chain_tail']


@pytest.mark.parametrize('fault', ['terminal', 'truncate', 'guard', 'render', 'substitute', 'throw'])
def test_native_earlystop_engineering_failure_never_pad(fault):
    r, store, envs = execute(lambda: ArrayEnv(fault=fault))
    assert len(r['rows']) == 4 and all(e.closed and len(e.commands) == 11 for e in envs)
    assert r['full_horizons'] == 0 and all([c['step'] for c in row['checkpoints']] == [0] for row in r['rows'])
    if fault in ('terminal', 'truncate'):
        assert r['status'] == 'matched_sampling_physical_recorded_not_admitted' and r['early_stops'] == 4
    else: assert r['status'] == 'partial_or_failed' and r['engineering_failures'] == 4
    assert all(row['own_attempted'] == 1 and row['own_returned'] == (0 if fault == 'throw' else 1) for row in r['rows'])
    if fault == 'throw': assert r['counts']['public_step'] == dict(attempted=44, completed=40)
    else: assert r['counts']['physics_step']['completed'] == 456
    if fault in ('guard', 'render', 'substitute'):
        assert all(row['last_observation']['png'] in store.files for row in r['rows'])


@pytest.mark.parametrize('field', ['raw', 'RGB', 'state', 'snapshot', 'goal'])
def test_precise_original_history_goal_binding_before_dispatch(field):
    p, d = fixture(); ref = d['references'][0]
    if field == 'raw': ref['raw_history'] = np.zeros((10, 2), np.float32)
    elif field == 'RGB': ref['history'][1, 0, 0, 0] ^= 1
    elif field == 'state': ref['states'][1, 0] += 1e-8
    elif field == 'snapshot': ref['dynamics'][1]['state_float64_bits'] = 'bad'
    else: ref['goal'][0] += 1
    r, _, _ = execute(p=p, data=d)
    assert all(row['own_attempted'] == 0 and not row['initial_exact'] for row in r['rows'][:2])
    assert all(row['full_horizon'] for row in r['rows'][2:])


@pytest.mark.parametrize('branch', range(4))
@pytest.mark.parametrize('fault', ['constructor', 'close', 'step', 'truncate'])
def test_each_one_failure_keeps_ALL_four_denominator(branch, fault):
    index = 0
    def factory():
        nonlocal index
        i = index; index += 1
        if i == branch:
            if fault == 'constructor': raise RuntimeError('partial constructor unknown')
            if fault == 'close': return ArrayEnv(close_error=True)
            return ArrayEnv(fault='throw' if fault == 'step' else 'truncate')
        return ArrayEnv()
    r, _, envs = execute(factory)
    assert index == len(r['rows']) == 4 and all(e.closed for e in envs)
    assert all(row['full_horizon'] for i, row in enumerate(r['rows']) if i != branch)
    row = r['rows'][branch]
    if fault == 'constructor':
        assert row['final_state'] is row['native_success'] is row['final_metrics'] is None
        assert row['constructor_internal_work'] == 'unknown'
        assert r['total_internal_physics_scope'] == 'unknown_constructor_work'
        assert r['counts']['environment_construct'] == dict(attempted=4, completed=3)
    elif fault == 'close': assert row['full_horizon'] and row['first_error']['stage'] == 'close'
    elif fault == 'step': assert row['own_attempted'] == 1 and row['own_returned'] == 0
    else: assert row['own_returned'] == 1 and row['status'] == 'native_stopped'


@pytest.mark.parametrize('stage', ['construct_return', 'step_intent', 'step_return', 'branch_closed', 'run_finished', 'report'])
def test_durable_failure_retains_actual_entries_returns_and_cleanup(stage):
    store = MemoryStore(); emit, put = store.emit, store.put
    def event(kind, **v):
        rec = v.get('receipt', {})
        fail = ((stage == 'construct_return' and kind == 'operation_returned' and rec.get('resource') == 'environment_construct') or
                (stage == 'step_intent' and kind == 'operation_dispatch_intent' and rec.get('phase') == 'plan') or
                (stage == 'step_return' and kind == 'operation_returned' and rec.get('phase') == 'plan') or stage == kind)
        if fail: raise OSError('mock durable write failure')
        return emit(kind, **v)
    def final(name, value, final=False):
        if stage == 'report' and final: raise OSError('mock final report failure')
        return put(name, value, final)
    store.emit, store.put = event, final
    r, _, envs = execute(store=store)
    assert r['status'] == 'partial_or_failed' and len(r['rows']) == 4 and all(e.closed for e in envs)
    if stage == 'step_intent': assert r['rows'][0]['own_attempted'] == r['rows'][0]['own_returned'] == 0
    if stage == 'step_return': assert r['rows'][0]['own_attempted'] == r['rows'][0]['own_returned'] == 1
    if stage == 'branch_closed': assert len(envs) == 1 and r['not_started'] == 3
    if stage == 'construct_return': assert len(envs) == 1 and r['counts']['environment_close']['completed'] == 1


def producer_rows():
    _, data = fixture(); rows = []
    for s in (0, 1):
        for arm in m.ARMS:
            r = dict(schema=m.AUTHORITY+'.model_plan', source_slot=s, arm=arm, task_id='task'+str(s),
                blind_sha256=data['references'][s]['blind_sha256'], task_export='/MOCK/task'+str(s), task_sha256='f'*64,
                task_contract_sha256='d'*64, commands=np.full((10, 2), .25, np.float32).tolist(),
                with_target=False, physical_execution=False, selection_uses_physics=False, P1_admission=False, official_MPPI=False)
            rows.append(dict(r, selection_sha256=m.source.digest(r), whole_run_report_sha256='c'*64))
    return data, rows


def entry():
    return dict(authority=m.AUTHORITY, output='/MOCK/newrun', report_sha256='c'*64,
                source_sha256='a'*64, tests_sha256='a'*64, innerdoc_sha256='a'*64)


def manifest_fixture():
    data, rows = producer_rows()
    manifest = dict(schema=m.NAME+'.selection', sourcepack='/MOCK/sourcepack.json', sourcepack_sha256=m.SOURCE_SHA,
        source_prepared='/MOCK/source_prepare.json', source_prepare_sha256='b'*64,
        roots=[dict(root_slot=i, root_sha256=r['root_sha256']) for i, r in enumerate(data['references'])],
        arms=list(m.ARMS), authorities=[entry()], selection_fixed_before_physics=True,
        physical_outcome_selection=False, wall_seconds=600)
    return manifest, data, rows


def test_exact_consume_call_new_authority_and_external_FINAL_sha_only():
    _, rows = producer_rows(); calls = []
    receipt = dict(schema=m.AUTHORITY+'.consume', rows=rows, P1_admission=False, equal_total_compute=False,
        official_MPPI=False, prediction_calls_and_samples_matched_between_sampling_arms=True,
        resources={k: dict(attempted=576, completed=576) for k in ('predict_API', 'predictor', 'action_encoder', 'pred_proj')},
        predictor_samples=dict(attempted=576, completed=576))
    path = m.ROOT/'experiments/spectral_reachability'/(m.AUTHORITY+'.py')
    module = NS(__file__=str(path), consume=lambda o, h: (calls.append((o, h)) or receipt))
    report = dict(prepared='/MOCK/prepare.json', prepare_sha256='b'*64, artifact_sha256={})
    prep = dict(schema=m.AUTHORITY+'.prepare', slots=[{k: v for k, v in rows[2*s].items() if k in (
        'source_slot', 'task_id', 'blind_sha256', 'task_export', 'task_sha256', 'task_contract_sha256')} for s in (0, 1)], bindings_sha256={})
    def hashfile(p):
        return ('a'*64 if str(p) in m.authority_files(entry()) else 'b'*64 if str(p).endswith('prepare.json') else 'c'*64)
    with patch.object(m, 'sha', side_effect=hashfile), patch.object(m, 'read', side_effect=lambda p, *a: report if str(p).endswith('report.json') else prep), \
         patch.object(m.importlib, 'import_module', return_value=module):
        selected, bindings = m.ownconsume(entry())
        assert selected == rows and calls == [(Path('/MOCK/newrun'), 'c'*64)]
        assert str(path) in bindings
        rows[0]['commands'][0][0] = .5
        with pytest.raises(ValueError, match='seal'): m.ownconsume(entry())


@pytest.mark.parametrize('key', ['source_sha256', 'tests_sha256', 'innerdoc_sha256'])
def test_unfinished_authority_never_imported_or_report_invented(key):
    e = entry(); e[key] = None
    with patch.object(m.importlib, 'import_module', side_effect=AssertionError('unfinished import')), \
         patch.object(m, 'sha', side_effect=AssertionError('no guessed hash')), pytest.raises(ValueError, match='external literal'):
        m.ownconsume(e)


def test_selection_new_plans_join_and_exact_original_seals():
    manifest, data, rows = manifest_fixture()
    with patch.object(m, 'ownconsume', return_value=(rows, {})), patch.object(m, 'task_bridge'):
        plans, _ = m.selection(manifest, data)
    assert [(p['root_slot'], p['arm']) for p in plans] == [(s, a) for s in (0, 1) for a in m.ARMS]
    assert [p['compiler_selection_sha256'] for p in plans] == [r['selection_sha256'] for r in rows]


@pytest.mark.parametrize('fault', ['single', 'reorder', 'missing', 'duplicate', 'boolslot', 'oldarm', 'rounded', 'outside', 'rootorder', 'outcome'])
def test_new_four_plan_join_failclosed(fault):
    manifest, data, rows = manifest_fixture()
    if fault == 'single': manifest['arms'] = m.ARMS[:1]
    elif fault == 'reorder': manifest['arms'].reverse()
    elif fault == 'missing': rows.pop()
    elif fault == 'duplicate': rows.append(deepcopy(rows[0]))
    elif fault == 'boolslot': rows[0]['source_slot'] = True
    elif fault == 'oldarm': rows[0]['arm'] = 'moving_gradient_open_loop'
    elif fault == 'rounded': rows[0]['commands'][0][0] = .1
    elif fault == 'outside': rows[0]['commands'][0][0] = 1.1
    elif fault == 'rootorder': manifest['roots'].reverse()
    else: manifest['physical_outcome_selection'] = True
    with patch.object(m, 'ownconsume', return_value=(rows, {})), patch.object(m, 'task_bridge'), pytest.raises(ValueError):
        m.selection(manifest, data)


@pytest.mark.parametrize('fault', ['RGB', 'goalRGB', 'raw', 'blind', 'contract', 'logits'])
def test_saved_task_bridge_exact_warm_goal_raw_contract(fault):
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as encoder
    _, data = fixture(); ref = data['references'][0]
    task = dict(root_raw_action_history=m.kernel.raw_history(ref['raw_history']), projection=np.eye(192),
                packet_logit_cap=float(.75*np.sqrt(5)), task_contract_sha256='d'*64)
    tc = dict(pixel_sha256=[m.kernel.moving.image_sha(a) for a in [*ref['history'], ref['goal_rgb']]],
              raw_action_sha256=m.kernel.moving.image_sha(ref['raw_history']))
    row = dict(schema=m.AUTHORITY+'.model_plan', source_slot=0, blind_sha256=ref['blind_sha256'], task_id='task',
        task_export='/MOCK/task', task_sha256='f'*64, task_contract_sha256='d'*64,
        logits=np.zeros((2, 10)).tolist(), commands=np.zeros((10, 2)).tolist())
    env = dict(source_slot=0, blind_sha256=ref['blind_sha256'], task_contract=tc)
    if fault == 'RGB': tc['pixel_sha256'][0] = 'bad'
    elif fault == 'goalRGB': tc['pixel_sha256'][-1] = 'bad'
    elif fault == 'raw': task['root_raw_action_history'] = np.zeros((2, 10))
    elif fault == 'blind': row['blind_sha256'] = 'bad'
    elif fault == 'contract': row['task_contract_sha256'] = 'bad'
    else: row['logits'][0][0] = 1.
    with patch.object(encoder, 'load_task', return_value=dict(gn_task=task, task_id='task')), \
         patch.object(encoder, 'read', return_value=env), pytest.raises(ValueError): m.task_bridge(row, ref, {})


def test_actual_attempts_not_intent_and_failed_dispatch_no_refund():
    p, _ = fixture(); store = MemoryStore(); ops = m.Ops(p, store)
    def fail(): raise RuntimeError('RAM failure')
    with pytest.raises(RuntimeError): ops.call('goal_bind', fail)
    assert ops.counts['goal_bind'] == dict(attempted=1, completed=0)
    store.emit = lambda *a, **k: (_ for _ in ()).throw(OSError('intent failure'))
    with pytest.raises(OSError): ops.call('goal_bind', lambda: pytest.fail('must not dispatch'))
    assert ops.counts['goal_bind'] == dict(attempted=1, completed=0)


def test_authorization_wall_and_cap_before_factory_or_step():
    p, d = fixture()
    with pytest.raises(ValueError): m.execute(p, d, MemoryStore(), lambda: pytest.fail('factory'), authorization='wrong')
    with pytest.raises(ValueError): m.run('none', 'none', 'none', lambda: pytest.fail('factory'), authorization='wrong')
    ops = m.Ops(p, MemoryStore()); ops.p['contract']['caps']['public_step'] = 0
    with pytest.raises(ValueError): ops.call('public_step', lambda: pytest.fail('step'))
    ops = m.Ops(fixture()[0], MemoryStore()); ops.started -= 601
    with pytest.raises(ValueError): ops.call('public_step', lambda: pytest.fail('step'))


def test_32MiB_store_floor_cap_wall_FINAL_no_file_writes():
    store = object.__new__(m.Store); store.started = 1.; store.wall = 600
    store.path = Path('/MOCK/output')
    with patch.object(m.time, 'monotonic', return_value=2), patch.object(m.Store, 'used', return_value=0), \
         patch.object(m.shutil, 'disk_usage', return_value=NS(free=m.FLOOR-1)), pytest.raises(ValueError): store.check(1)
    with patch.object(m.time, 'monotonic', return_value=2), patch.object(m.Store, 'used', return_value=m.CAP-m.FINAL), \
         patch.object(m.shutil, 'disk_usage', return_value=NS(free=m.FLOOR+m.CAP)), pytest.raises(ValueError): store.check(1)
    with patch.object(m.time, 'monotonic', return_value=602), pytest.raises(ValueError): store.check(1)
    with patch.object(m.Store, 'used', return_value=m.CAP), patch.object(m.shutil, 'disk_usage', return_value=NS(free=m.FLOOR+m.CAP)), \
         patch.object(Path, 'open', side_effect=AssertionError('no write')), pytest.raises(ValueError): store.write('report.json', b'x', final=True)


def test_capacity_88_actual_RGB_and_complete_report_RAM():
    from PIL import Image
    r, store, _ = execute(); sizes = []
    for value in store.files.values():
        if isinstance(value, np.ndarray):
            b = io.BytesIO(); Image.fromarray(value).save(b, format='PNG'); sizes.append(len(b.getvalue()))
        else: sizes.append(len(value))
    sizes.append(sum(len(m.raw(e)) for e in store.events))
    assert len([n for n in store.files if n.endswith('.png')]) == 88
    assert sum(sizes) < m.CAP and len(m.raw(r)) < m.FINAL


def test_no_old_sealed_mutations_no_neural_simulator_import_or_prepare():
    assert not any(n in sys.modules for n in ('torch', 'gymnasium', 'stable_worldmodel', 'pymunk'))
    assert m.helper.native_constructor_contract()[1] == 'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2'
    assert Path(m.__file__).resolve() in m.physical_files()
    assert m.CAP == 32*1024**2 and m.FLOOR == 200*1024**2
    assert m.contract()['caps']['physics_step'] == 816


def test_canonical_SHA_interoperates_with_producer_without_importing_neural():
    from experiments.spectral_reachability import moving_prediction_matched_sampling_cuda_20261027_v1 as producer
    value = {'真实': [-.0, .25], 'raw10': np.zeros((10, 2)).tolist()}
    import hashlib
    expected = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert m.source.digest(value) == producer.core.digest(value) == expected
    assert not any(n in sys.modules for n in ('torch', 'gymnasium', 'stable_worldmodel', 'pymunk'))


@pytest.mark.parametrize('fault', [None, 'terminal', 'truncate', 'throw', 'guard'])
def test_downstream_assemble_actual7_checkpoints_and_partial_denominator(fault):
    p, d = fixture(); r, store, _ = execute(lambda: ArrayEnv(fault=fault), p=p, data=d)
    cp = m.assemble(r, store.events, p)
    assert len(r['rows']) == 4 and len(cp) == (12 if fault is None else 4)
    assert [(c['root_slot'], c['arm'], c['checkpoint']['step']) for c in cp] == [
        (s, a, step) for s in (0, 1) for a in m.ARMS for step in ((0, 5, 10) if fault is None else (0,))]
    assert all(len(c['checkpoint']['state']) == 7 for c in cp)
    assert all(c['selected_plan']['root_sha256'] == d['references'][c['root_slot']]['root_sha256'] for c in cp)


@pytest.mark.parametrize('fault', ['meter', 'receipt', 'raw', 'root', 'goal', 'state6', 'step', 'plan', 'guard', 'denominator', 'native', 'terminal'])
def test_downstream_assemble_rejects_cross_join_missing_axes_or_replacement(fault):
    p, d = fixture(); r, store, _ = execute(p=p, data=d); events = deepcopy(store.events)
    if fault == 'meter': r['counts']['physics_step']['completed'] -= 1
    elif fault == 'receipt': r['actual_operation_receipts'].pop()
    elif fault == 'raw':
        planstep = next(x for x in r['actual_operation_receipts'] if x.get('phase') == 'plan' and x['resource'] == 'public_step')
        planstep['raw_action'][0] = 0.
    elif fault == 'root': r['rows'][0]['root_state'][0] += 1
    elif fault == 'goal': r['rows'][0]['goal'][0] += 1
    elif fault == 'state6': r['rows'][0]['checkpoints'][1]['state'].pop()
    elif fault == 'step': r['rows'][0]['checkpoints'][1]['step'] = 4
    elif fault == 'plan': r['rows'][0]['selected_plan']['commands'][0][0] = 0.
    elif fault == 'guard':
        next(e for e in events if e['kind'] == 'guard_receipt' and e['phase'] == 'plan')['accepted'] = False
    elif fault == 'denominator': r['rows'].pop()
    elif fault == 'native': r['rows'][0]['final_metrics']['native_distance'] += 1
    else: events.pop()
    with pytest.raises(ValueError): m.assemble(r, events, p)


def test_downstream_consume_SHA_gate_before_any_prepare_source_or_images():
    with patch.object(Path, 'iterdir', return_value=[]), patch.object(m, 'sha', return_value='a'*64), \
         patch.object(m, 'verify', side_effect=AssertionError('no unchecked verify')), \
         patch.object(m, 'read', side_effect=AssertionError('no unchecked read')), pytest.raises(ValueError, match='external actual physical'):
        m.consume('/MOCK/output', 'b'*64)
