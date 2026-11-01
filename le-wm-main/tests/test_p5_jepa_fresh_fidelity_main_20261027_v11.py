"""Array/protocol checks only, no model or simulator."""
import numpy as np
import pytest
import json
from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace

from experiments.spectral_reachability import p5_jepa_fresh_fidelity_main_20261027_v11 as fresh


def test_env_arguments_not_divided_by100():
    rgb = np.zeros((4, 224, 224, 3), np.uint8)
    prop = np.ones((4, 4), np.float32)
    commands = np.full((15, 2), .2, np.float64)
    stats = dict(ACTION_MEAN=[.1, -.1], ACTION_STD=[.2, .1], PROPRIO_MEAN=[0]*4, PROPRIO_STD=[2]*4)
    p, packets = fresh.normalized_inputs(rgb, prop, commands, stats)
    np.testing.assert_array_equal(p, .5)
    np.testing.assert_allclose(packets.reshape(15, 2), np.tile([.5, 3.], (15, 1)), atol=1e-6)


def test_last_packet_only_and_axis_order():
    packet = np.arange(30, dtype=np.float32).reshape(3, 10)
    bank = fresh.controls(packet)
    for arm in fresh.ARMS:
        np.testing.assert_array_equal(bank[arm][:2], packet[:2])
    np.testing.assert_array_equal(bank['reverse_steps'][-1].reshape(5, 2), packet[-1].reshape(5, 2)[::-1])
    np.testing.assert_array_equal(bank['negative_normalized'][-1], -packet[-1])
    assert np.all(bank['normalized_zero'][-1] == 0)
    np.testing.assert_array_equal(packet, np.arange(30, dtype=np.float32).reshape(3, 10))


def test_zero_or_constant_last_packet_not_informative():
    with pytest.raises(ValueError, match='nontrivial'):
        fresh.controls(np.zeros((3, 10)))
    with pytest.raises(ValueError, match='nontrivial'):
        fresh.controls(np.ones((3, 10)))


def test_malformed_inputs():
    with pytest.raises(ValueError):
        fresh.controls(np.zeros((30,)))
    with pytest.raises(ValueError):
        fresh.numeric([float('nan')], (1,))


def test_contract_is_not_efficacy():
    p = fresh.protocol()
    assert not p['P5_admission'] and not p['target_optimized'] and not p['training']
    assert not p['cpu_neural_fallback'] and not p['prospective_control']
    assert p['counts'] == dict(official_encoder=1, proprio_encoder=1, predictor=4)
    assert p['frames'] == [0, 5, 10, 15]


def saved_factual(tmp_path):
    """Serialization fixture, never a simulator or efficacy sample."""
    directory = tmp_path / fresh.FACTUAL / 'fixture'
    directory.mkdir(parents=True)
    source = directory.parent / 'dummy_source.txt'
    source.write_text('test serialization fixture, not physics')
    schedule = np.arange(30).reshape(3, 5, 2) / 100.
    protocol = dict(seed=2026102701, frames=[0, 5, 10, 15], schedule=schedule.tolist())
    manifest = dict(schema=fresh.FACTUAL, protocol=protocol, seals={str(source): fresh.parent.sha(source)})
    frozen = directory.parent / 'manifest_frozen.json'
    frozen.write_text(json.dumps(manifest))
    payloads = []
    for role in ('a', 'b'):
        for frame in range(16):
            dest = directory / role / f'frame_{frame:02d}'
            dest.mkdir(parents=True)
            for name, array in dict(rgb=np.zeros((224, 224, 3), np.uint8),
                state=np.zeros(7, np.float32), proprio=np.zeros(4, np.float32)).items():
                np.save(dest / (name + '.npy'), array, allow_pickle=False)
            for name in ('full_public_state.json', 'transition.json'):
                (dest / name).write_text('{}')
            files = {p.name: dict(sha256=fresh.parent.sha(p)) for p in dest.iterdir()}
            payloads.append(dict(phase='assets_durable', role=role, record=dict(frame=frame, files=files)))
        for frame in range(1, 16):
            (directory / role / f'command_{frame:02d}.json').write_text(json.dumps(dict(
                frame=frame, raw_env_argument=schedule.reshape(15, 2)[frame-1].tolist())))
    report = dict(schema=fresh.FACTUAL, status='fresh_native_nonzero_factual_recorded', error=None,
        source_hashes_unchanged=True, manifest_file_sha256=fresh.parent.sha(frozen), protocol=protocol,
        trajectory_frames={'a': 16, 'b': 16}, rows=[dict(frame=i, all_exact=True, exact={'fixture': True}) for i in range(16)],
        native_call_receipts=dict(python_call_entries=dict(env_construct=2, seed=4, reset=2,
            control_step=30, space_step=302, native_render=32, native_get_obs=32, close=2)))
    payloads.append(dict(phase='terminal', summary_sha256=fresh.digest(report)))
    entries, previous = [], None
    for i, event in enumerate(payloads):
        entry = dict(sequence=i, previous_sha256=previous, event=event)
        entry['entry_sha256'] = fresh.digest(entry)
        previous = entry['entry_sha256']
        entries.append(json.dumps(entry))
    journal = directory / 'journal.jsonl'
    journal.write_text('\n'.join(entries) + '\n')
    report['journal_sha256'] = fresh.parent.sha(journal)
    (directory / 'report.json').write_text(json.dumps(report))
    return directory


def test_saved_factual_identity_and_units(tmp_path):
    directory = saved_factual(tmp_path)
    rgb, prop, commands, bindings = fresh.factual_inputs(directory)
    assert rgb.shape == (4, 224, 224, 3) and prop.shape == (4, 4)
    np.testing.assert_array_equal(commands, np.arange(30).reshape(15, 2) / 100.)
    assert len(bindings) > 190


@pytest.mark.parametrize('tamper', ['asset', 'command', 'journal'])
def test_saved_factual_tamper_rejected(tmp_path, tamper):
    directory = saved_factual(tmp_path)
    if tamper == 'asset':
        np.save(directory / 'a/frame_02/state.npy', np.ones(7, np.float32), allow_pickle=False)
    elif tamper == 'command':
        path = directory / 'a/command_01.json'
        value = json.loads(path.read_text())
        value['raw_env_argument'] = [99, 99]
        path.write_text(json.dumps(value))
    else:
        with (directory / 'journal.jsonl').open('a') as stream:
            stream.write('{}\n')
    with pytest.raises(ValueError):
        fresh.factual_inputs(directory)


def fake_cuda_interfaces(monkeypatch, *, fail_prediction=None, source_error=False):
    """Constant-array interfaces, NOT neural modules, physics, or CUDA evidence."""
    class Tensor:
        is_cuda = True  # Interface contract fixture, never a device claim.
        def __init__(self, value): self.value = np.asarray(value)
        @property
        def shape(self): return self.value.shape
        def permute(self, *dims): return Tensor(self.value.transpose(dims))
        def to(self, *args): return self
        def __truediv__(self, value): return Tensor(self.value / value)
        def __getitem__(self, item): return Tensor(self.value[item])
        def reshape(self, *shape): return Tensor(self.value.reshape(shape))
        def unsqueeze(self, axis): return Tensor(np.expand_dims(self.value, axis))
        def expand(self, *shape): return Tensor(np.broadcast_to(self.value, shape))
        def cpu(self): return self
        def numpy(self): return self.value
        def double(self): return Tensor(self.value.astype(np.float64))
        def __sub__(self, other): return Tensor(self.value - other.value)
        def square(self): return Tensor(self.value**2)
        def mean(self): return Tensor(self.value.mean())
        def item(self): return float(self.value)
    class Handle:
        def remove(self): return None
    class Interface:
        def __init__(self, kind): self.kind, self.pre, self.post, self.calls = kind, [], [], 0
        def register_forward_pre_hook(self, fn): self.pre.append(fn); return Handle()
        def register_forward_hook(self, fn): self.post.append(fn); return Handle()
        def __call__(self, *args, **kwargs):
            for fn in self.pre: fn(self, args)
            self.calls += 1
            if self.kind == 'predictor' and self.calls == fail_prediction:
                raise RuntimeError('injected forward failure, not a neural call')
            if self.kind == 'official_encoder': out = dict(x_norm_patchtokens=Tensor(np.zeros((4, 256, 384), np.float32)))
            elif self.kind == 'proprio_encoder': out = Tensor(np.zeros((1, 3, 1, 16), np.float32))
            else: out = (Tensor(np.zeros((1, 3, 256, 384), np.float32)),)
            for fn in self.post: fn(self, args, out)
            return out
    fake_torch = ModuleType('torch')
    fake_torch.from_numpy = Tensor
    fake_torch.float32 = np.float32
    fake_torch.isfinite = lambda value: np.isfinite(value.value)
    fake_torch.inference_mode = nullcontext
    attention = ModuleType('torch.nn.attention')
    attention.sdpa_kernel = lambda *a: nullcontext()
    attention.SDPBackend = SimpleNamespace(MATH='mock-math')
    transform = ModuleType('app.plan_common.datasets.transforms')
    transform.make_transforms = lambda **kw: (lambda x: x)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'torch.nn.attention', attention)
    monkeypatch.setitem(sys.modules, 'app.plan_common.datasets.transforms', transform)
    monkeypatch.delenv('PYTHONPATH', raising=False)
    modules = {k: Interface(k) for k in ('official_encoder', 'proprio_encoder', 'predictor')}
    receipts = {k: {'fixture': True} for k in modules}
    v4 = SimpleNamespace(model_receipt=lambda m: {'fixture': True})
    v5 = SimpleNamespace(import_v4=lambda r: v4)
    v7 = SimpleNamespace(dependencies=lambda r: (None, v5, None))
    monkeypatch.setattr(fresh.parent, 'helper', lambda r: v7)
    monkeypatch.setattr(fresh.parent, 'require_cuda', lambda: None)
    monkeypatch.setattr(fresh.parent, 'load_cuda', lambda *a: (modules, {}, {}, receipts))
    manifest = dict(factual='fixture', parent_manifest={}, stats={}, bindings={'fixture': 'expected'})
    monkeypatch.setattr(fresh, 'prepare', lambda p: manifest)
    monkeypatch.setattr(fresh, 'factual_inputs', lambda p: (np.zeros((4,224,224,3),np.uint8), None, None, {}))
    packets = np.arange(30, dtype=np.float32).reshape(3,10)
    monkeypatch.setattr(fresh, 'normalized_inputs', lambda *a: (np.zeros((4,4),np.float32), packets))
    manifest['controls'] = {k: v.tolist() for k,v in fresh.controls(packets).items()}
    def source_hash(path):
        if source_error: raise OSError('injected source disappearance')
        return 'expected'
    monkeypatch.setattr(fresh.parent, 'sha', source_hash)
    return manifest


def test_later_forward_failure_retains_completed_arrays_and_attempts(monkeypatch):
    manifest = fake_cuda_interfaces(monkeypatch, fail_prediction=2)
    result = fresh.run_cuda(manifest)
    assert result['status'] == 'fresh_fidelity_failed'
    assert result['attempted_counts']['predictor'] == 2
    assert result['counts']['predictor'] == 1
    assert set(result['artifact_arrays']) == {'official_patch_features', 'actual_successor'}
    assert result['costs'] == {'actual': 0.}
    assert result['arm_status']['actual'] == 'completed'
    assert result['arm_status']['normalized_zero'] == 'attempted'
    assert 'injected forward failure' in result['error']


def test_final_source_failure_preserves_original_error(monkeypatch):
    manifest = fake_cuda_interfaces(monkeypatch, fail_prediction=2, source_error=True)
    result = fresh.run_cuda(manifest)
    assert 'injected forward failure' in result['error']
    assert not result['sources_unchanged']
    assert 'source disappearance' in result['source_check_errors'][0]
    assert 'actual_successor' in result['artifact_arrays']


def test_complete_interface_counts_are_consistent(monkeypatch):
    result = fresh.run_cuda(fake_cuda_interfaces(monkeypatch))
    assert result['status'] == 'fresh_native_fidelity_recorded'
    assert result['counts_match'] and result['model_unchanged'] and result['sources_unchanged']
    assert len(result['artifact_arrays']) == 5 and len(result['costs']) == 4

