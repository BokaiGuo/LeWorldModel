"""NumPy/PIL RAM fixtures and saved metadata only; no model/GPU/physics."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import agent_p2_midpoint_teacher_20261028_v1 as m


@pytest.fixture(scope='module')
def actual():
    return m.collect()


def test_actual_full4x2_and_no_existing_step5(actual):
    assert [(f['root_slot'], f['arm']) for f in actual['frames']] == [(i, a) for i in range(4) for a in m.ARMS]
    assert len(actual['existing_encoding_inventory']['files_sha256']) == 44
    assert not actual['existing_encoding_inventory']['matched_existing_step5']
    assert actual['unique_step5_image_count'] == 7  # root0 both genuine plans share a prefix
    assert actual['encoder_abi'] == dict(image_shape=[224, 224], patch_shape=[14, 14], tokens=257,
                                        encoder_dim=192, native_dim=192, cls_index=0)
    assert set(actual['image_statistics']) == {'mean', 'std'}
    assert actual['contract']['encoder_cap'] == actual['contract']['existing_projector_cap'] == 8
    assert actual['contract']['predict_cap'] == actual['contract']['AD_cap'] == actual['contract']['physics_cap'] == 0
    for f in actual['frames']:
        assert f['primitive_step'] == 5 and f['predicted_history_index'] == 1
        assert f['predicted_vector'] == f['predicted_context']['latent_history'][-1]
        assert f['executed_prefix_actions'] == f['raw_actions'][:5]
        assert f['frame_binding_sha256'] == m.digest({k: v for k, v in f.items() if k != 'frame_binding_sha256'})


@pytest.fixture(scope='module')
def real_midpoint():
    prepared = m.read(m.ROOT / m.gap.INPUTS['prepare'][0])
    root = m.read(m.ROOT / 'results/agent_p2_runtime_guard_20261028_v1_known_MAIN01/root_0.json')
    p = m.read(m.ROOT / 'results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01/protocol.json')
    return prepared['plans'][0]['execution_plan'], root['results'][0], p['context']['normalization']


@pytest.mark.parametrize('damage', ['missing_mid', 'terminal_as_mid', 'wrong_latent', 'wrong_token',
                                    'wrong_root', 'wrong_actions', 'wrong_std', 'no_paid_replay', 'bad_shape'])
def test_midpoint_rejects_substitution_and_missing(real_midpoint, damage):
    plan, result, norm = deepcopy(real_midpoint)
    win = result['winner']
    if damage == 'missing_mid': win['segmented_contexts'].pop(1)
    if damage == 'terminal_as_mid': win['segmented_contexts'][1] = deepcopy(win['whole_context'])
    if damage == 'wrong_latent': win['segmented_contexts'][1]['latent_history'][2][0] += 1
    if damage == 'wrong_token': win['segmented_contexts'][1]['action_history'][1][0] += 1
    if damage == 'wrong_root': plan['predictor_context']['latent_history'][0][0] += 1
    if damage == 'wrong_actions': plan['commands'][0][0] = .125
    if damage == 'wrong_std': norm['std'][0] = 1
    if damage == 'no_paid_replay': result['ledger'][-4]['completed'] = False
    if damage == 'bad_shape': win['segmented_contexts'][1]['latent_history'][0].pop()
    with pytest.raises(ValueError): m.midpoint(plan, result, norm)


def fake_tensor(kind='encoder', device='cuda', value=None):
    if value is None:
        value = np.zeros((1, 3, 224, 224) if kind == 'encoder' else (1, 192), np.float32)
    return SimpleNamespace(device=SimpleNamespace(type=device), shape=value.shape, value=value)


def eight():
    return [dict(frame_id='f' + str(i)) for i in range(8)]


def fake_persist(name, tensor):
    return dict(file=name + '.npy', sha256='RAM', dtype='float32', shape=list(tensor.shape))


@pytest.mark.parametrize('damage', ['cpu', 'shape', 'wrong_frame', 'projector_first'])
def test_native_guard_precedes_callback(damage):
    calls, events = [], []
    ledger = m.Ledger(eight(), events.append, fake_persist)
    ledger.active = 'f0'
    tensor, kind = fake_tensor(), 'encoder'
    if damage == 'cpu': tensor.device.type = 'cpu'
    if damage == 'shape': tensor.shape = (2, 3, 224, 224)
    if damage == 'wrong_frame': ledger.active = 'f1'
    if damage == 'projector_first': kind, tensor = 'existing_projector', fake_tensor('existing_projector')
    with pytest.raises(ValueError): ledger.dispatch(kind, tensor, lambda: calls.append(1), (), {})
    assert not calls and not ledger.entries and ledger.blocked


def test_failure_nonrefundable_attempt_and_first_error():
    count = 0
    def journal(event):
        nonlocal count
        count += 1
        if count == 3: raise OSError('secondary journal')
    ledger = m.Ledger(eight(), journal, fake_persist)
    ledger.active = 'f0'
    def fail(): raise RuntimeError('original native failure')
    with pytest.raises(RuntimeError, match='original native failure'):
        ledger.dispatch('encoder', fake_tensor(), fail, (), {})
    assert ledger.entries[0]['native_entered'] and not ledger.entries[0]['native_returned']
    assert ledger.journal_errors and ledger.counts()['encoder']['attempted'] == 1
    with pytest.raises(ValueError): ledger.dispatch('encoder', fake_tensor(), lambda: 1, (), {})


def test_attempt_journal_failure_no_native_entry():
    def fail(event): raise OSError('journal unavailable')
    ledger = m.Ledger(eight(), fail, fake_persist)
    ledger.active = 'f0'
    calls = []
    with pytest.raises(OSError): ledger.dispatch('encoder', fake_tensor(), lambda: calls.append(1), (), {})
    assert not calls and not ledger.entries[0]['native_entered']
    assert ledger.counts()['encoder']['wall_seconds'] == 0  # actually zero native entries


@pytest.mark.parametrize('kind', ['predict', 'predictor', 'action_encoder', 'pred_proj', 'AD.grad', 'AD.jvp'])
def test_forbidden_native_entries(kind):
    ledger = m.Ledger(eight(), lambda _: None, fake_persist)
    with pytest.raises(RuntimeError): ledger.forbid(kind)()
    assert len(ledger.blocked) == 1


def ram_fixture(folder, encoder_tokens=257, wrong_normalization=False):
    """NOT research evidence: synthetic RGB and module outputs only."""
    frames = []
    for i in range(8):
        path = folder / ('input' + str(i) + '.png')
        rgb = np.full((224, 224, 3), i*16, np.uint8)
        Image.fromarray(rgb).save(path)
        _, pixel = m.pixels(path, m.sha(path))
        f = dict(frame_id='f' + str(i), rgb_path=str(path), rgb_file_sha256=m.sha(path), pixel_sha256=pixel)
        f.update(root_slot=i//2, root_sha256='RAM_root_' + str(i//2), arm=m.ARMS[i%2],
                 actual_state=[0.]*7, actual_goal_state=[0.]*7,
                 goal_vector=[0.]*192, active_metric_id='RAM_Q', predicted_vector=[0.]*192)
        f['frame_binding_sha256'] = m.digest(f)
        frames.append(f)
    p = dict(frames=frames, model_bindings={'RAM': 'not_a_model'}, image_statistics=m.image_statistics(),
             encoder_abi=dict(tokens=257, encoder_dim=192))
    rows, errors = [], []
    with (folder / 'events.jsonl').open('x') as stream:
        journal = m.Journal(stream)
        def persist(name, tensor): return m.save_array(folder, name, tensor.value)
        ledger = m.Ledger(frames, journal, persist, precision=m.precision_contract)
        def adapter(f):
            rgb, _ = m.pixels(f['rgb_path'], f['rgb_file_sha256'])
            x = m.normalized_rgb(rgb, p['image_statistics'])
            if wrong_normalization: x = x + np.float32(.001)
            output = fake_tensor(value=np.zeros((1, encoder_tokens, 192), np.float32))
            enc = ledger.dispatch('encoder', fake_tensor(value=x[0]),
                                  lambda: SimpleNamespace(last_hidden_state=output), (), {})
            proj = fake_tensor(value=enc.last_hidden_state.value[:, 0])
            z = ledger.dispatch('existing_projector', proj, lambda: proj, (), {})
            return dict(rgb=rgb, normalized_pixels=x, native192=z.value[0])
        m.encode_frames(p, folder, adapter, ledger, journal, rows, errors)
        assert not errors and len(rows) == 8
        r = dict(schema=m.NAME + '.report', status='all8_step5_encoded', contract=m.contract(), errors=[], blocked=[],
                 journal_errors=[], first_error=None, model_hash_before='RAM', model_hash_after='RAM',
                 model_unchanged=True, source_verified=True, host=dict(device='cuda:0'), model_bindings=p['model_bindings'],
                 new_predict_calls=0, new_AD_calls=0, new_physics_steps=0, P1_admission=False, P2_admission=False,
                 scientific_admission=False, rows=rows, entries=ledger.entries, counts=ledger.counts(),
                 total_wall_seconds=1., journal_sha256=m.sha(folder / 'events.jsonl'), journal_tail_sha256=journal.previous)
        r.update(precision_before=m.precision_contract(), precision_after=m.precision_contract())
    return r, p


def test_full_ram16_actual_counted_entries_and_strict_arrays(tmp_path):
    r, p = ram_fixture(tmp_path)
    arrays = m.validate_run(tmp_path, r, p)
    assert len(arrays) == 8
    assert all(r['counts'][k]['entered'] == r['counts'][k]['returned'] == r['counts'][k]['saved'] == 8
               for k in ('encoder', 'existing_projector'))


def test_wrong197_rejected_with_all_array_hashes_and_journal_valid(tmp_path):
    r, p = ram_fixture(tmp_path, encoder_tokens=197)
    with pytest.raises(ValueError, match='buffer closure'): m.validate_run(tmp_path, r, p)


def test_wrong_normalization_rejected_even_if_encoder_input_matches(tmp_path):
    r, p = ram_fixture(tmp_path, wrong_normalization=True)
    with pytest.raises(ValueError, match='normalization arithmetic'): m.validate_run(tmp_path, r, p)


@pytest.mark.parametrize('damage', ['missing_row', 'order', 'entry_count', 'CPU', 'unknown_hash', 'changed_model',
                                    'cost_scope', 'frame_binding', 'counts', 'array_hash', 'journal', 'dtype', 'precision'])
def test_strict_ram_run_mutations_failclosed(tmp_path, damage):
    r, p = ram_fixture(tmp_path)
    if damage == 'missing_row': r['rows'].pop()
    if damage == 'order': r['rows'].reverse()
    if damage == 'entry_count': r['entries'].pop()
    if damage == 'CPU': r['host']['device'] = 'cpu'
    if damage == 'unknown_hash': r['model_hash_before'] = r['model_hash_after'] = None
    if damage == 'changed_model': r['model_hash_after'] = 'changed'
    if damage == 'cost_scope': r['new_predict_calls'] = 1
    if damage == 'frame_binding': p['frames'][0]['pixel_sha256'] = 'wrong'
    if damage == 'counts': r['counts']['encoder']['entered'] = 7
    if damage == 'array_hash': r['rows'][0]['arrays']['native192']['sha256'] = 'wrong'
    if damage == 'journal': r['journal_sha256'] = 'wrong'
    if damage == 'dtype': r['entries'][0]['arrays']['input']['dtype'] = 'float64'
    if damage == 'precision': r['precision_after']['sdpa_flash'] = True
    with pytest.raises(ValueError): m.validate_run(tmp_path, r, p)


def test_full_denominator_retained_after_first_failure(tmp_path):
    p = dict(frames=[dict(frame_id='f'+str(i), frame_binding_sha256=str(i)) for i in range(8)])
    rows, errors, calls = [], [], []
    def fail(f):
        calls.append(f['frame_id'])
        raise RuntimeError('RAM failure')
    ledger = m.Ledger(p['frames'], lambda _: None, fake_persist)
    m.encode_frames(p, tmp_path, fail, ledger, lambda _: None, rows, errors)
    assert calls == ['f0'] and len(rows) == 8 and rows[0]['status'] == 'failed'
    assert [r['status'] for r in rows[1:]] == ['blocked_not_attempted']*7


def test_numeric_uses_original_q_no_fitting():
    zero, one = np.zeros(192), np.ones(192)
    result = m.numeric(one, zero, zero, (2*np.eye(192)).tolist())
    assert result == dict(teacher_Q=4., predicted_Q=0., latent_prediction_mse=1., projected_prediction_mse=4.)
    with pytest.raises(ValueError): m.numeric(one, zero, zero, None)


def test_actual_config_image_and_patch_derive257(actual):
    config = m.read(Path(actual['policy']) / 'config.json')
    assert m.encoder_abi(config)['tokens'] == 1 + (224 // 14)**2 == 257
    damaged = deepcopy(config)
    damaged['encoder']['patch_size'] = 0
    with pytest.raises(ValueError): m.encoder_abi(damaged)
    assert config['projector']['input_dim'] == config['projector']['output_dim'] == 192


def test_public_normalization_byte_arithmetic():
    rgb = np.full((224, 224, 3), 255, np.uint8)
    stats = m.image_statistics()
    x = m.normalized_rgb(rgb, stats)
    assert x.shape == (1, 1, 3, 224, 224) and x.dtype == np.float32
    for k in range(3):
        expected = (np.float32(1) - np.float32(stats['mean'][k])) / np.float32(stats['std'][k])
        assert np.all(x[0, 0, k] == expected)


@pytest.mark.parametrize('damage', ['flash', 'efficient', 'cudnn_sdp', 'matmul_tf32', 'cudnn_tf32', 'autocast'])
def test_precision_guard_ram_no_torch(damage):
    values = dict(m.precision_contract())
    key = dict(flash='sdpa_flash', efficient='sdpa_mem_efficient', cudnn_sdp='sdpa_cudnn',
               matmul_tf32='matmul_tf32', cudnn_tf32='cudnn_tf32', autocast='autocast_cuda_enabled')[damage]
    values[key] = True
    cuda = SimpleNamespace(math_sdp_enabled=lambda: values['sdpa_math'], flash_sdp_enabled=lambda: values['sdpa_flash'],
                           mem_efficient_sdp_enabled=lambda: values['sdpa_mem_efficient'],
                           cudnn_sdp_enabled=lambda: values['sdpa_cudnn'], matmul=SimpleNamespace(allow_tf32=values['matmul_tf32']))
    torch = SimpleNamespace(backends=SimpleNamespace(cuda=cuda, cudnn=SimpleNamespace(allow_tf32=values['cudnn_tf32'])),
                            is_autocast_enabled=lambda _: values['autocast_cuda_enabled'])
    with pytest.raises(ValueError, match='precision guard'): m.checked_precision(torch)


def test_inventory_ignores_own_analyze_output(tmp_path, monkeypatch):
    results = tmp_path / 'results'
    results.mkdir()
    m.write(results / (m.NAME + '_analyze.json'), {'not': 'an encoding'})
    m.write(results / (m.NAME + '.encoding.json'), {'terminal_rgb_sha256': 'target'})
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    assert m.encoding_inventory([dict(rgb_file_sha256='target')])['files_sha256'] == {}
    m.write(results / 'p2_other.encoding.json', {'terminal_rgb_sha256': 'target'})
    with pytest.raises(ValueError, match='already encoded'): m.encoding_inventory([dict(rgb_file_sha256='target')])


def test_cold_saved_import_and_metadata_no_torch():
    code = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','gymnasium','pymunk','stable_worldmodel','requests'}:
            raise AssertionError('forbidden runtime import ' + fullname)
sys.meta_path.insert(0, Guard())
from experiments.spectral_reachability import agent_p2_midpoint_teacher_20261028_v1 as m
p = m.collect()
assert len(p['frames']) == 8
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, '-B', '-c', code], cwd=m.ROOT, capture_output=True, check=True, timeout=30,
                   env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'})


def test_cli_run_needs_explicit_main_authorization():
    result = subprocess.run([sys.executable, '-B', '-m', 'experiments.spectral_reachability.' + m.NAME, 'run'],
                            cwd=m.ROOT, capture_output=True, text=True, timeout=30,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode != 0 and 'MAIN HOST CUDA authorization' in result.stderr


def test_public_prepare_verify_with_ram_inputs_only(tmp_path, monkeypatch):
    frozen = dict(schema=m.NAME + '.protocol', contract=m.contract(), frames=eight(), bindings={}, model_bindings={},
                  historical_sources={}, existing_encoding_inventory=dict(files_sha256={}))
    monkeypatch.setattr(m, 'collect', lambda: deepcopy(frozen))
    monkeypatch.setattr(m, 'source_files', lambda: [Path(m.__file__).resolve()])
    output = tmp_path / 'prepared'
    receipt = m.prepare(output)
    assert m.verify(output, receipt['prepare_sha256'])['frames'] == eight()
    with pytest.raises(ValueError): m.verify(output, '0'*64)
    with pytest.raises(ValueError): m.prepare(output)


def test_public_consume_ram_end_to_end(tmp_path, monkeypatch):
    folder = tmp_path / 'RAM_run'
    folder.mkdir()
    r, p = ram_fixture(folder)
    compilation = tmp_path / 'RAM_compilation.json'
    m.write(compilation, dict(rows=[dict(root_sha256='RAM_root_'+str(i), compiled=dict(
        goal_vectors=[[0.]*192], scoring=dict(metric=dict(sha256='RAM_Q', projection=np.eye(192).tolist()))))
        for i in range(4)]))
    p.update(compilation_path=str(compilation), compilation_sha256=m.sha(compilation), inherited_resources='RAM only')
    g = dict(rows=[dict(root_slot=i//2, arm=m.ARMS[i%2], prediction={'RAM': True}, observed={'RAM': True}) for i in range(8)],
             pairs=[dict(composition_minus_joint={'RAM': True}) for i in range(4)])
    gap_path = tmp_path / 'RAM_gap.json'
    m.write(gap_path, g)
    monkeypatch.setattr(m, 'GAP', gap_path)
    monkeypatch.setattr(m, 'verify', lambda *args: p)
    r.update(prepare_sha256='RAM_prepare_SHA', gap_report_sha256=m.GAP_SHA)
    m.write(folder / 'report.json', r)
    m.write(folder / 'execution_binding.json', dict(prepared='RAM', prepare_sha256='RAM_prepare_SHA', gap_report_sha256=m.GAP_SHA))
    output = tmp_path / 'RAM_consume.json'
    result = m.consume(folder, m.sha(folder/'report.json'), output)
    saved = m.read(output)
    assert saved['root_denominator'] == 4 and len(saved['rows']) == 8
    assert saved['rows'][0]['midpoint']['teacher_Q'] == 0.
    assert saved['scientific_admission'] is False and result['sha256'] == m.sha(output)
    with pytest.raises(FileExistsError): m.consume(folder, m.sha(folder/'report.json'), output)
    with pytest.raises(ValueError): m.consume(folder, '0'*64, tmp_path/'never.json')
