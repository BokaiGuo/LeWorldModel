from contextlib import nullcontext
import hashlib
import json
import numpy as np

import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_native_outer_20261029_v1 as outer
from experiments.spectral_reachability import agent_p3_c3_k17_native_meter_20261029_v1 as meter
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import p3_k17_live_source_budget_seam_20261029 as seam
from experiments.spectral_reachability import p3_k17_saved_source_budget_consumer_20261029 as consumer
from experiments.spectral_reachability import p3_k17_host_bundle_runner_20261029 as runner

ORIGINAL_ADMISSION = runner._admission

def source():
    rows = [({'slot': slot, 'root': root, 'mode': mode, 'K': 17,
              'first5_raw_f32_hex': bytes(range(40)).hex()}, {'task': root}, {'ref': root})
            for slot, root, mode in outer.ORDER]
    return {'roots': [{}, {}], 'source_plan': {}}, rows


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.render_action = False
        self.render_size = 224
        self.render_mode = 'rgb_array'
        self.closed = False
    def close(self): self.closed = True


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(runner, '_admission', lambda *args: None)
    monkeypatch.setattr(outer, 'material', source)
    monkeypatch.setattr(outer.h3, 'validate_env', lambda body: None)
    monkeypatch.setattr(outer.h3, 'native_hooks', lambda *args: nullcontext())
    monkeypatch.setattr(runner, 'environment_evidence', lambda env: {'fixture': True})
    def episode(*args): raise RuntimeError('fixture episode failure')
    monkeypatch.setattr(outer.episode, 'episode', episode)
    data, rows = source()
    authority = {'source_files_sha256': consumer.sources(), 'bundle_code_sha256': runner.bundle_codes(),
                 'guard_source_sha256': runner.strict.source_files(),
                 'caps': dict(meter.CAPS), 'wall_seconds': meter.WALL,
                 'order': [list(x) for x in outer.ORDER], 'source_sha256': seam.source_digest(data, rows),
                 'environment': {'fixture': True}}
    return authority, runner.digest(transport.canonical(authority))


def test_public_refusal_before_output_or_factory(tmp_path):
    with pytest.raises(ValueError, match='explicit P3 scope'):
        runner.run_host(tmp_path/'bundle', {}, '0'*64, 'no')
    assert not (tmp_path/'bundle').exists()
    with pytest.raises(ValueError, match='explicit P3 scope'):
        runner._execute_admitted(tmp_path/'bundle', {}, '0'*64, FakeEnv)


def test_real_admission_accepts_only_explicit_pinned_scope(tmp_path, setup, monkeypatch):
    authority, _ = setup
    monkeypatch.setattr(runner, '_admission', ORIGINAL_ADMISSION)
    authority.update(schema='p3_k17_host_prepare.v1', scope=runner.CONFIRM,
                     physical_authorized=False, scientific_pass=False)
    anchor = runner.digest(transport.canonical(authority))
    runner._admission(authority, anchor, runner.CONFIRM)
    with pytest.raises(ValueError, match='explicit P3 scope'):
        runner._admission(authority, anchor, 'different')
    with pytest.raises(ValueError, match='explicit P3 scope'):
        runner._admission(authority, '0'*64, runner.CONFIRM)
    result = runner._execute_admitted(tmp_path/'bundle', authority, anchor, FakeEnv, confirmation=runner.CONFIRM)
    assert result['completion']['status'] == 'saved_incomplete'
    assert runner.consume_bundle(tmp_path/'bundle', result['completion_sha256'], authority, anchor)['status'] == 'saved_incomplete'


def test_real_place_interface_has_separate_reset_identity(tmp_path, setup, monkeypatch):
    del setup
    native = runner.native_store.NativeStore(tmp_path/'native')
    bound = seam.SourceBoundStore.from_saved(native)
    scoped = runner.scoped_outer.ScopedStore(bound)
    scoped.emit('run_started', denominator=6, order=[list(x) for x in outer.ORDER])
    met = meter.Meter(scoped)
    met.where = scoped.where = {'branch': 0, 'root_slot': 0, 'mode': 'fixedH1'}
    class Block: shapes = []
    env = FakeEnv()
    env.block = Block()
    env.reset = lambda **kw: ({'state': np.zeros(7)}, {'goal_state': np.zeros(7)})
    monkeypatch.setattr(outer.h3.source, 'sample_state', lambda *args: np.zeros(7))
    met.call('constructor', lambda: env)
    slot = {'reset_seed': 2, 'root_seed': 3}
    outer.h3.place(env, {}, slot, met, scoped)
    events = consumer.read_events(native.path/'journal.jsonl', runner.digest((native.path/'journal.jsonl').read_bytes()))
    placement = next(e['payload'] for e in events if e['kind'] == 'placement_recipe')
    assert placement['slot'] == 3 and placement['reset_identity'] == slot


def test_failure_is_saved_incomplete_and_independently_consumed(tmp_path, setup):
    authority, sha = setup
    path = tmp_path/'bundle'
    result = runner._execute_admitted(path, authority, sha, FakeEnv)
    assert result['completion']['status'] == 'saved_incomplete'
    check = runner.consume_bundle(path, result['completion_sha256'], authority, sha)
    assert check['status'] == 'saved_incomplete' and not check['production_authorized']
    assert check['audit']['branches'][0]['status'] == 'engineering_failed'
    assert not check['native_provenance_verified'] and not check['scientific_pass']
    with pytest.raises(ValueError, match='exclusive'): runner._execute_admitted(path, authority, sha, FakeEnv)


def test_environment_mismatch_fails_and_closes(tmp_path, setup, monkeypatch):
    authority, sha = setup
    seen = []
    def create():
        env = FakeEnv()
        seen.append(env)
        return env
    monkeypatch.setattr(runner, 'environment_evidence', lambda env: {'fixture': False})
    result = runner._execute_admitted(tmp_path/'bundle', authority, sha, create)
    assert seen[0].closed
    native_path = tmp_path/'bundle'/'native'
    events = consumer.read_events(native_path/'journal.jsonl', runner.digest((native_path/'journal.jsonl').read_bytes()))
    operations = [e['payload']['receipt'] for e in events if e['kind'] == 'operation_returned']
    assert [r['resource'] for r in operations] == ['constructor', 'close']
    check = runner.consume_bundle(tmp_path/'bundle', result['completion_sha256'], authority, sha)
    assert check['status'] == 'failed' and check['audit'] is None


def test_missing_forged_and_unanchored_files_fail(tmp_path, setup):
    authority, sha = setup
    path = tmp_path/'bundle'
    result = runner._execute_admitted(path, authority, sha, FakeEnv)
    with pytest.raises(ValueError, match='completion SHA'):
        runner.consume_bundle(path, '0'*64, authority, sha)
    with pytest.raises(ValueError, match='authority SHA'):
        runner.consume_bundle(path, result['completion_sha256'], authority, '0'*64)
    (path/'native'/'report.json').write_text('{}\n')
    with pytest.raises(ValueError, match='inventory mismatch'):
        runner.consume_bundle(path, result['completion_sha256'], authority, sha)


def test_source_and_budget_mismatch_block_before_output(tmp_path, setup):
    authority, _ = setup
    authority['caps'] = dict(authority['caps'], public_step=999)
    sha = runner.digest(transport.canonical(authority))
    with pytest.raises(ValueError, match='source/budget'):
        runner._execute_admitted(tmp_path/'bundle', authority, sha, FakeEnv)
    assert not (tmp_path/'bundle').exists()


def test_saved_consumer_failure_publishes_failed_manifest(tmp_path, setup, monkeypatch):
    authority, sha = setup
    def fail(*args, **kwargs): raise ValueError('synthetic audit failure')
    monkeypatch.setattr(consumer, 'consume', fail)
    path = tmp_path/'bundle'
    result = runner._execute_admitted(path, authority, sha, FakeEnv)
    assert result['completion']['status'] == 'failed'
    manifest = json.loads((path/'manifest.json').read_bytes())
    assert manifest['failure']['message'] == 'synthetic audit failure'
    check = runner.consume_bundle(path, result['completion_sha256'], authority, sha)
    assert check['status'] == 'failed' and check['audit'] is None


def test_six_slot_fixture_bundle_remains_engineering_only(tmp_path, setup, monkeypatch):
    authority, sha = setup
    monkeypatch.setattr(runner.strict, 'audit', lambda path, journal, pins, material: {
        'schema': 'fixture', 'guards_recomputed': 6, 'frames_checked': 306})
    def episode(env, root, plan, item, task, ref, row, met, bound):
        def observation(index):
            name = f'b{row["index"]}_{index}.png'
            body = name.encode()
            bound.store.store.inner.png(name, body)
            bound.emit('fresh_observation', receipt={'png': name, 'file_sha256': hashlib.sha256(body).hexdigest()})
        def step(phase, index, bits):
            met.call('public_step', lambda: None, phase=phase, step=index, raw_bits=bits)
            bound.emit('returned_transition', **met.where, phase=phase, step=index, requested_bits=bits,
                       actual_bits=bits, actual_dtype='<f4', terminated=False, truncated=False)
        for index in range(1, 11): step('warmup', index, (bytes([index])*8).hex())
        for index in range(36): observation(index)
        bound.emit('k17_guard', accepted=True, proof={'physical_dispatch_ready': False,
                   'history': {'accepted': True}, 'action': {'actual_replay_tokens_exact': True}})
        raw = bytes.fromhex(item['first5_raw_f32_hex'])
        for index in range(1, 6):
            step('plan', index, raw[(index-1)*8:index*8].hex())
            for j in range(3): observation(36+(index-1)*3+j)
        return {'status': 'horizon_completed', 'returned': 5, 'missing_steps': []}
    monkeypatch.setattr(outer.episode, 'episode', episode)
    path = tmp_path/'bundle'
    result = runner._execute_admitted(path, authority, sha, FakeEnv)
    assert result['completion']['status'] == 'saved_engineering_only'
    checked = runner.consume_bundle(path, result['completion_sha256'], authority, sha)
    assert checked['audit']['six_slot_event_coverage']
    assert not checked['audit']['complete_receipt_verified']
    assert not checked['native_provenance_verified'] and not checked['scientific_pass']
