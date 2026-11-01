"""RAM fake native environment and saved-byte fixtures; no physics/neural."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from experiments.spectral_reachability import agent_terminal_witness_physical_20261028_v1 as m
from test_agent_allmode_physical_20261028_v1 import fixture as old_fixture, MemoryStore, Env, fake_place, h, reseal


def fixture():
    prior, data, _, compiled, _ = old_fixture()
    analysis = dict(schema=m.selector.NAME + '.analysis', new_algorithm=True, P1_admission=False,
        independent_confirmation=False, status='new_exploratory_model_only_selector_not_admitted', panels=[])
    for i, slot in enumerate(compiled['slots']):
        context = dict(RAM='context' + str(i)); goal = [float(i)] * 192; q = np.eye(192).tolist()
        slot['packet'] = dict(root_context=context, goal=goal); slot['q'] = q
        # Explicit RAM full222 denominator; not a scientific score/asset.
        commands = deepcopy(prior['plans'][i]['commands']); rawsha = hashlib.sha256(m.mixed.raw10(commands).tobytes()).hexdigest()
        candidates = [dict(index=j, nonlinear_terminal_Q=float(abs(j - (i + 3))),
            raw_float32_sha256=rawsha, model_prefix_uphill_from_anchor=False) for j in range(222)]
        panel = dict(family='mixedmode', index=i, root_slot=slot['root_slot'], condition=slot['condition'],
            task_sha256=slot['task_sha256'], candidate_denominator=222, terminal_available=222,
            candidates=candidates, literal_forward_requests=[], status='complete_model_only', P1_admission=False,
            context_sha256=m.selector.sha_object(context), goal_sha256=m.selector.sha_object(goal),
            q_sha256=m.selector.base.digest(q), original_stage_plus_ridge=dict(index=0),
            nonlinear_terminal_Q=dict(index=i + 3, score=0., commands=commands,
                raw_float32_sha256=rawsha, model_prefix_uphill_from_anchor=False))
        analysis['panels'].append(panel)
    compiled['schema'] = m.mixed.PRODUCER + '.prepare'
    plans = m.plans_from_analysis(analysis, compiled, data)
    authority = dict(selector=m.selector.NAME, analysis=str(m.ANALYSIS), analysis_sha256=m.ANALYSIS_SHA,
        source_sha256={}, exact_build_verified=True, family='mixedmode', new_algorithm=True,
        selection_uses_physics=False, P1_admission=False, plans_sha256=m.mixed.core.digest(plans))
    p = dict(schema=m.NAME + '.prepare', contract=m.contract(), plans=plans, selector_authority=authority,
        source_report_sha256=m.mixed.SOURCE_SHA, source_slots=deepcopy(prior['source_slots']),
        source_roots=deepcopy(prior['source_roots']), warm_traces=deepcopy(prior['warm_traces']), bindings={})
    return p, data, analysis, compiled


def test_newplan_authority_is_real_analysis_not_old_selector_prepare():
    p, _, _, _ = fixture(); m.validate_execution(p)
    assert [x['selected_index'] for x in p['plans']] == [3, 4, 5, 6]
    assert p['selector_authority']['analysis_sha256'] == m.ANALYSIS_SHA
    assert not set(p['selector_authority']) & {'consume_sha256', 'prepare_sha256', 'report_sha256'}
    assert all(x['schema'] == m.NAME + '.execution_plan' for x in p['plans'])


@pytest.mark.parametrize('damage', ['family', 'order', 'missing', 'count', 'task', 'context', 'goal', 'Q',
    'winner', 'raw', 'digest', 'partial', 'authority', 'admission', 'old_schema'])
def test_analysis_or_newplan_mutation_refused(damage):
    p, data, analysis, compiled = fixture(); panel = analysis['panels'][0]
    if damage == 'family': panel['family'] = 'allmode'
    elif damage == 'order': analysis['panels'].reverse()
    elif damage == 'missing': panel['literal_forward_requests'] = [dict(index=1)]
    elif damage == 'count': panel['candidates'].pop()
    elif damage == 'task': panel['task_sha256'] = h('wrong')
    elif damage == 'context': panel['context_sha256'] = h('wrong')
    elif damage == 'goal': panel['goal_sha256'] = h('wrong')
    elif damage == 'Q': panel['q_sha256'] = h('wrong')
    elif damage == 'winner': panel['nonlinear_terminal_Q']['index'] += 1
    elif damage == 'raw': panel['nonlinear_terminal_Q']['commands'][0][0] = .125
    elif damage == 'digest': p['plans'][0]['selection_digest'] = h('wrong')
    elif damage == 'partial': panel['terminal_available'] = 221
    elif damage == 'authority': p['selector_authority']['analysis_sha256'] = h('invented')
    elif damage == 'admission': analysis['P1_admission'] = True
    elif damage == 'old_schema': p['schema'] = m.mixed.NAME + '.prepare'
    with pytest.raises(ValueError):
        if damage in ('digest', 'authority', 'old_schema'): m.validate_execution(p)
        else: m.plans_from_analysis(analysis, compiled, data)


def test_allmode_scores_outcomes_do_not_choose_library_or_mode():
    p, data, analysis, compiled = fixture(); expected = deepcopy(p['plans'])
    analysis['panels'].insert(0, dict(family='allmode', physical_success=True, terminal_score=-1e90))
    analysis['native_outcomes'] = [1e99] * 4
    for panel in analysis['panels'][1:]:
        panel['native_success'] = True; panel['nonlinear_terminal_Q']['physical_reward'] = 1e99
    assert m.plans_from_analysis(analysis, compiled, data) == expected


def execute(*, fault=None, at=11, close_error=False):
    p, data, _, _ = fixture(); store = MemoryStore(); envs = []
    def factory():
        env = Env(fault=fault, at=at, close_error=close_error); envs.append(env); return env
    with pytest.MonkeyPatch.context() as mp:
        # Explicit RAM-only placement and validation seams; production source
        # NEVER patches imported globals or rewrites an old schema/receipt.
        mp.setattr(m.mixed, 'place', fake_place); mp.setattr(m.mixed, 'validate_env', lambda _: None)
        mp.setattr(m, 'sha', lambda f: hashlib.sha256(store.files[Path(f).name]).hexdigest())
        report = m.execute(p, data, store, factory, authorization=m.AUTH, prepared='/RAM_ONLY/native_prepare.json',
                           prepare_sha256=h('native prepare'))
    return report, store, p, data, envs


@lru_cache(maxsize=1)
def complete(): return execute()


def test_actual_lowlevel_episode_full4_all10_own_selector_arm():
    r, s, p, data, envs = complete()
    assert r['status'] == m.RECORDED, r['errors']
    assert len(envs) == len(m.validate_saved(r, s.events, p)) == 4
    assert r['counts'] == {k: dict(attempted=v, completed=v) for k, v in m.contract()['caps'].items()}
    assert r['counts']['public_step']['completed'] == 80 and r['counts']['physics_step']['completed'] == 816
    assert len([x for x in s.files if x.endswith('.png')]) == 264
    assert all(x['own_returned'] == 10 and x['warmup_returned'] == 10 and
               x['prefix51']['complete'] and x['missing_checkpoints'] == [] for x in r['rows'])
    for env, plan in zip(envs, p['plans'], strict=True):
        assert env.closed and env.dispatch_flags == [False] * 20 and env.with_target is False
        assert np.asarray(env.commands[10:], np.float32).tobytes() == m.mixed.raw10(plan['commands']).tobytes()
        assert '_set_state' not in env.__dict__ and 'reset' not in env.__dict__


@pytest.mark.parametrize('fault', ['terminal', 'truncate', 'throw', 'substitute', 'guard', 'render'])
def test_native_or_engineering_stop_preserves4_and_does_not_pad(fault):
    r, s, p, _, envs = execute(fault=fault, at=15)
    assert len(r['rows']) == 4 and all(x.closed for x in envs)
    assert all(row['own_returned'] <= 10 for row in r['rows'])
    if fault in ('terminal', 'truncate'):
        assert r['status'] == m.RECORDED, r['errors']
        assert all(len(x.commands) == 15 for x in envs)
        assert all(row['own_returned'] == 5 and row['missing_checkpoints'] == [10] for row in r['rows'])
        m.validate_saved(r, s.events, p)
    else:
        assert r['status'] == 'partial_or_failed'


@pytest.mark.parametrize('damage', ['raw', 'scope', 'task', 'state', 'snapshot', 'reward', 'goal', 'triplet',
    'RGB', 'trajectory', 'unknown_event', 'counts', 'seal', 'first5only', 'old_schema', 'admission'])
def test_saved_resealed_mutations_refused(damage):
    original, store, prepared, _, _ = complete()
    r, p, es = deepcopy(original), deepcopy(prepared), deepcopy(store.events)
    def event(kind): return next(e for e in es if e['kind'] == kind)
    if damage == 'raw': next(e for e in es if e['kind'] == 'operation_dispatch_intent' and
        e['receipt']['resource'] == 'public_step')['receipt']['raw_bits'] = '00'
    elif damage == 'scope': event('fresh_observation')['condition'] = 'on'
    elif damage == 'task': r['rows'][0]['task_id'] = h('wrong')
    elif damage == 'state': event('fresh_observation')['receipt']['state'][0] += 1
    elif damage == 'snapshot': event('fresh_observation')['receipt']['snapshot']['state'][0] += 1
    elif damage == 'reward': r['rows'][0]['responses'][0]['reward'] = 0
    elif damage == 'goal': r['rows'][0]['goal_pose'][0] += 1
    elif damage == 'triplet': event('fresh_observation')['receipt']['overlay_index'] = 2
    elif damage == 'RGB': event('fresh_observation')['receipt']['pixel_sha256'] = h('bad')
    elif damage == 'trajectory': r['rows'][0]['trajectory'].pop()
    elif damage == 'unknown_event': event('fresh_observation')['kind'] = 'unknown'
    elif damage == 'counts': r['counts']['physics_step']['completed'] -= 1
    elif damage == 'seal': r['execution_binding']['selector_authority']['analysis_sha256'] = h('bad')
    elif damage == 'first5only': r['rows'][0]['own_returned'] = 5
    elif damage == 'old_schema': r['schema'] = m.mixed.NAME + '.report'
    elif damage == 'admission': r['P1_admission'] = True
    reseal(es, r)
    with pytest.raises((ValueError, KeyError, StopIteration)): m.validate_saved(r, es, p)


def test_saved_complete_consume_png_bytes_and_extra_artifact(monkeypatch):
    r, store, p, data, _ = complete(); files = deepcopy(store.files)
    expected = hashlib.sha256(files['report.json']).hexdigest()
    monkeypatch.setattr(m, 'verify', lambda *a: (deepcopy(p), deepcopy(data)))
    monkeypatch.setattr(m.mixed, 'saved_artifacts', lambda *a: deepcopy(files))
    out = m.consume('/RAM_ONLY', expected)
    assert out['fixed_denominator'] == 4 and out['saved_PNG_verified'] == 264
    assert out['P1_admission'] is out['freshness'] is False
    files['unexpected.txt'] = b'unknown'
    with pytest.raises(ValueError, match='unaccounted'): m.consume('/RAM_ONLY', expected)
    files.pop('unexpected.txt')
    name = next(x for x in files if x.endswith('.png')); files[name] += b'bad'
    with pytest.raises(ValueError): m.consume('/RAM_ONLY', expected)


def test_MAIN_authority_rejected_before_reads_or_imports(monkeypatch):
    monkeypatch.setattr(m, 'verify', lambda *a: pytest.fail('no unauthorized reads'))
    with pytest.raises(ValueError): m.run(None, None, None, lambda: pytest.fail('no env'), authorization=None)
    with pytest.raises(ValueError): m.main(['run', '--authorization', 'wrong'])


def test_cold_import_no_foreign_runtime():
    script = '''
import builtins
old=builtins.__import__
def guard(name,*a,**k):
    if name.split('.')[0] in ('torch','gymnasium','pymunk','stable_worldmodel'):raise AssertionError(name)
    return old(name,*a,**k)
builtins.__import__=guard
from experiments.spectral_reachability import agent_terminal_witness_physical_20261028_v1 as m
assert m.contract()['inheritance'] is False
try:m.main(['--help'])
except SystemExit as e:assert e.code==0
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=m.ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_native_joint_success_not_native7_reward_threshold():
    value = m.mixed.metrics([0.] * 7, [1., 1., 1., 1., .01, 10000., -10000.])
    assert value['native_success'] is True and value['native7_reward'] < -10000


def test_pinned_real_analysis_metadata_only_read():
    # Explicit analysis JSON/source hashes only; no model arrays/sourcepack,
    # original physics outcomes, selector.build or production prepare/run.
    assert m.sha(m.ANALYSIS) == m.ANALYSIS_SHA
    analysis = m.read(m.ANALYSIS, 8 * 1024**2)
    panels = [x for x in analysis['panels'] if x['family'] == 'mixedmode']
    assert [(x['root_slot'], x['condition']) for x in panels] == m.ORDER
    assert [x['nonlinear_terminal_Q']['index'] for x in panels] == [51, 47, 61, 69]
    assert [x['original_stage_plus_ridge']['index'] for x in panels] == [51, 51, 23, 23]
    assert m.sha(m.mixed.PREPARED) == m.mixed.PREP_SHA
    compiled = m.read(m.mixed.PREPARED, m.selector.mixed.PREP_CAP)
    data = dict(exports={(x['root_slot'], x['condition']): dict(sha256=x['blind_sha256']) for x in compiled['slots']})
    plans = m.plans_from_analysis(analysis, compiled, data)
    assert [x['selected_index'] for x in plans] == [51, 47, 61, 69]
    for plan, panel in zip(plans, panels, strict=True):
        assert plan['commands'] == panel['nonlinear_terminal_Q']['commands']
        assert plan['selection_digest'] == m.mixed.core.digest({k: v for k, v in plan.items() if k != 'selection_digest'})


def test_material_verify_real_entrypoints_with_RAM_authority(monkeypatch):
    _, data, analysis, compiled = fixture(); data['bindings'] = {}; compiled['bindings'] = {}
    parent = dict(family='mixedmode', prepared=str(m.mixed.PREPARED), prepare_sha256=m.mixed.PREP_SHA,
        report=str(m.ROOT / 'results/RAM_mixed/report.json'), report_sha256=m.mixed.REPORT_SHA)
    analysis.update(inputs=[parent], source_sha256={str(Path(m.selector.__file__).resolve()): h('RAM selector source')})
    records = {str(m.ANALYSIS): analysis, parent['prepared']: compiled}
    hashes = {str(m.ANALYSIS): m.ANALYSIS_SHA, parent['prepared']: m.mixed.PREP_SHA,
        parent['report']: m.mixed.REPORT_SHA, **analysis['source_sha256']}
    for slot in compiled['slots']: hashes[slot['task_file']] = slot['task_sha256']
    calls = []
    monkeypatch.setattr(m, 'read', lambda path, *a: deepcopy(records[str(path)]))
    monkeypatch.setattr(m, 'sha', lambda path: hashes.get(str(path), h(str(path))))
    monkeypatch.setattr(m.selector, 'build', lambda: calls.append('exact build') or deepcopy(analysis))
    monkeypatch.setattr(m.mixed, 'source_material', lambda: deepcopy(data))
    monkeypatch.setattr(m.mixed.allmode, 'constructor_contract', lambda: ('/RAM/native.py', h('/RAM/native.py')))
    args = dict(selector_analysis=str(m.ANALYSIS), selector_analysis_sha256=m.ANALYSIS_SHA)
    p, _ = m.material(args)
    assert calls == ['exact build']
    assert p['schema'] == m.NAME + '.prepare' and p['selector_authority']['family'] == 'mixedmode'
    records['/RAM/native_prepare.json'] = p
    verified, _ = m.verify('/RAM/native_prepare.json', m.sha('/RAM/native_prepare.json'))
    assert verified == p and calls == ['exact build', 'exact build']
    changed = deepcopy(analysis); changed['P1_admission'] = True
    monkeypatch.setattr(m.selector, 'build', lambda: changed)
    with pytest.raises(ValueError, match='entire real selector.build'): m.material(args)


def test_prepare_rejects_existing_before_selector_build(tmp_path, monkeypatch):
    existing = tmp_path / 'existing.json'; existing.write_text('{}')
    monkeypatch.setattr(m, 'material', lambda *a: pytest.fail('existing target must fail before reconstruction'))
    with pytest.raises(ValueError, match='exclusive prepare'): m.prepare(str(existing), {})


def test_cli_preserves_actual_analysis_names(monkeypatch):
    called = []
    monkeypatch.setattr(m, 'prepare', lambda output, arguments: called.append((output, arguments)) or dict(status='RAM'))
    assert m.main(['prepare', '--output', '/RAM/p.json', '--analysis', str(m.ANALYSIS), '--analysis-sha', m.ANALYSIS_SHA]) == 0
    assert called == [('/RAM/p.json', dict(selector_analysis=str(m.ANALYSIS), selector_analysis_sha256=m.ANALYSIS_SHA))]
