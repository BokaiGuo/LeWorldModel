"""RAM/saved fixtures only; never call production model, environment or runner."""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v2 as m
import agent_p3_same_feedback_physical20_20261028_v1_tests as fixtures


def image_sources():
    pure = Path(sysconfig.get_paths()['purelib'])
    paths = [m.base.ROOT / 'experiments/run_pusht_counterfactual.py',
             pure / 'stable_pretraining/data/dataset_stats.py',
             pure / 'torchvision/transforms/v2/_misc.py',
             pure / 'torchvision/transforms/v2/functional/_misc.py',
             pure / 'torchvision/transforms/v2/functional/_geometry.py']
    return dict(sources={str(p): m.sha(p) for p in paths})


@pytest.fixture(scope='module')
def ram_full():
    report, store, _, _ = fixtures.execute_ram()
    assert report['completed20'] == 4
    return report, store.events


def rechain(events):
    previous = None
    for i, e in enumerate(events):
        e.update(index=i, previous_sha256=previous)
        e['event_sha256'] = m.base.old.source.digest({k: v for k, v in e.items() if k != 'event_sha256'})
        previous = e['event_sha256']
    return b''.join(m.base.raw(e) + b'\n' for e in events)


def test_complete_all_event_and_all_resource_census(ram_full):
    r, es = ram_full
    result = m.census(es, r)
    assert result['complete']
    assert result['event_counts']['fresh_observation'] == result['event_counts']['guard_passed'] == 128
    assert result['event_counts']['handler_event'] == 80
    assert result['event_counts']['boundary_sealed'] == 16
    assert result['counts'] == r['counts']
    for eid, per in result['per_episode'].items():
        assert per['set_state'] == dict(attempted=4, completed=4)
        assert per['goal_setter'] == dict(attempted=3, completed=3)
        assert per['goal_render'] == dict(attempted=2, completed=2)
        assert per['physics_step'] == dict(attempted=304, completed=304)


def test_valid_rechained_JSONL_reaches_complete_census(ram_full, tmp_path):
    r, es = deepcopy(ram_full)
    (tmp_path / 'journal.jsonl').write_bytes(rechain(es))
    r.update(journal_sha256=m.sha(tmp_path / 'journal.jsonl'), journal_chain_tail=es[-1]['event_sha256'])
    assert m.census(m.load_events(tmp_path, r), r)['complete']


@pytest.mark.parametrize('kind', ['guard_passed', 'warmup_return', 'control_return', 'fresh_observation',
    'handler_event', 'boundary_encoded', 'boundary_sealed', 'authorized_dispatch', 'episode_closed',
    'run_started', 'run_finished'])
@pytest.mark.parametrize('action', ['duplicate', 'remove'])
def test_missing_or_extra_known_events_rejected(ram_full, kind, action):
    r, es = deepcopy(ram_full); i = next(i for i, e in enumerate(es) if e['kind'] == kind)
    if action == 'duplicate': es.insert(i, deepcopy(es[i]))
    else: es.pop(i)
    with pytest.raises((ValueError, KeyError, IndexError)): m.census(es, r)


@pytest.mark.parametrize('damage', ['unknown', 'extra_field', 'wrong_episode', 'handler_namespace',
    'nested_foreign_event', 'handler_extra', 'seal_hash', 'encoding_hash', 'checkpoint', 'guard',
    'region', 'units', 'bool_units', 'operation_id', 'resource', 'reset_census', 'goal_census', 'orphan_return'])
def test_rehashed_physical_mutations_rejected(ram_full, tmp_path, damage):
    r, es = deepcopy(ram_full)
    def event(kind): return next(e for e in es if e['kind'] == kind)
    if damage == 'unknown': event('guard_passed')['kind'] = 'unmetered_physics'
    elif damage == 'extra_field': event('guard_passed')['reward'] = 0
    elif damage == 'wrong_episode': event('guard_passed')['episode_id'] = 's1_adaptive'
    elif damage == 'handler_namespace': event('handler_event')['namespace'] = 'other_b00_encode'
    elif damage == 'nested_foreign_event': event('handler_event')['event']['native']['kind'] = 'unknown'
    elif damage == 'handler_extra': event('handler_event')['event']['native']['entry']['state'] = [1]
    elif damage == 'seal_hash': event('boundary_sealed')['seal_sha256'] = '0' * 64
    elif damage == 'encoding_hash': event('boundary_encoded')['encoding_sha256'] = '0' * 64
    elif damage == 'checkpoint': r['rows'][0]['checkpoints'][0]['pixel_sha256'] = '0' * 64
    elif damage == 'guard': event('guard_passed')['canonical_agent_pixels'] = 0
    elif damage in ('region', 'units', 'bool_units', 'operation_id', 'resource'):
        e = event('operation_attempted')
        key, value = {'region': ('region', 'encoding'), 'units': ('units', 2), 'bool_units': ('units', True),
                      'operation_id': ('operation_id', 80), 'resource': ('resource', 'unknown')}[damage]
        e[key] = value
    elif damage in ('reset_census', 'goal_census'):
        # Forge report AND bookend totals to retain a self-consistent report;
        # journal independent counts still must reject it.
        key = 'reset' if damage == 'reset_census' else 'goal_setter'
        r['counts'][key]['completed'] -= 1; es[-1]['counts'] = deepcopy(r['counts'])
    elif damage == 'orphan_return': es.remove(event('operation_attempted'))
    payload = rechain(es); (tmp_path / 'journal.jsonl').write_bytes(payload)
    r.update(journal_sha256=m.sha(tmp_path / 'journal.jsonl'), journal_chain_tail=es[-1]['event_sha256'])
    with pytest.raises(ValueError): m.census(m.load_events(tmp_path, r), r)


def test_partial_unknown_event_still_rejected(ram_full, tmp_path):
    r, es = deepcopy(ram_full); r.update(status='partial_or_failed', errors=[{'stage': 'RAM'}])
    es[-1]['status'] = r['status']; es[3]['kind'] = 'unknown'
    (tmp_path / 'journal.jsonl').write_bytes(rechain(es))
    r.update(journal_sha256=m.sha(tmp_path / 'journal.jsonl'), journal_chain_tail=es[-1]['event_sha256'])
    with pytest.raises(ValueError, match='unknown physical'): m.load_events(tmp_path, r)


@pytest.mark.parametrize('stop', [5, 10, 15, 20])
def test_clean_native_stop_keeps_actual_denominator(stop):
    r, s, _, _ = fixtures.execute_ram(fault='terminal', at=10 + stop)
    c = m.census(s.events, r)
    assert c['complete'] and r['completed20'] == (4 if stop == 20 else 0)
    assert c['counts']['public_step']['completed'] == 4 * (10 + stop)


def test_partial_prefix_not_upgraded(ram_full):
    r, es = deepcopy(ram_full); r['status'] = 'partial_or_failed'; r['errors'] = [{'stage': 'RAM'}]
    es[-1]['status'] = r['status']
    out = m.census(es, r)
    assert out['complete'] is False and 'not certified' in out['reason']


def test_original_transform_AST_and_no_guessed_stats():
    p = image_sources(); abi = m.image_contract(p)
    assert abi['layout'] == [1, 1, 3, 224, 224]
    assert abi['statistics'] == dict(mean=[.485, .456, .406], std=[.229, .224, .225])
    broken = deepcopy(p); broken['sources'].pop(next(iter(broken['sources'])))
    with pytest.raises(ValueError, match='source SHA'): m.image_contract(broken)
    broken = deepcopy(p); broken['sources'][next(iter(broken['sources']))] = '0' * 64
    with pytest.raises(ValueError, match='source SHA'): m.image_contract(broken)


def test_float32_separate_ops_all_uint8_values():
    rgb = (np.arange(224 * 224 * 3) % 256).astype(np.uint8).reshape(224, 224, 3)
    stats = m.image_contract(image_sources())['statistics']; got = m.preprocess(rgb, stats)
    for ch in range(3):
        table = np.array([np.float32(np.float32(np.float32(i) * np.float32(1 / 255)) -
            np.float32(stats['mean'][ch])) / np.float32(stats['std'][ch]) for i in range(256)], np.float32)
        assert got[0, 0, ch].tobytes() == table[rgb[:, :, ch]].tobytes()
    assert not np.array_equal(got, ((rgb.transpose(2, 0, 1).astype(np.float32) / np.float32(255) -
        np.asarray(stats['mean'], np.float32)[:, None, None]) /
        np.asarray(stats['std'], np.float32)[:, None, None])[None, None])


def save_array(path, value):
    stream = io.BytesIO(); np.save(stream, value, allow_pickle=False); path.write_bytes(stream.getvalue())


@pytest.fixture
def rgb_fixture(tmp_path):
    eid = 's0_fixed'; ns = eid + '_b00'; folder = tmp_path / (ns + '_encode'); folder.mkdir()
    path = tmp_path / (eid + '_plan_00.png'); alias = tmp_path / (ns + '_plan_05.png')
    rgb = (np.arange(224 * 224 * 3) % 256).astype(np.uint8).reshape(224, 224, 3)
    Image.fromarray(rgb).save(path); alias.write_bytes(path.read_bytes())
    png = dict(path=str(path), file_sha256=m.sha(path), pixel_sha256=m.base.old.array_sha(rgb))
    response = dict(latent=dict(path=str(folder / 'native192.npy')))
    er = dict(actual_step=0, mapping=dict(actual_step=0, original_png=png, alias_png=str(alias)),
        request=dict(png=dict(png, path=str(alias))), response=response)
    (tmp_path / (ns + '_encoding.json')).write_bytes(m.base.raw(er))
    (folder / 'receipt.json').write_bytes(m.base.raw(response))
    p = image_sources(); value = m.preprocess(rgb, m.image_contract(p)['statistics'])
    save_array(folder / 'preprocessed_pixels.npy', value)
    report = dict(rows=[dict(episode_id=eid, encodings=[dict(step=0, receipt_sha256=m.base.old.source.digest(er))])])
    return tmp_path, report, p, value


def test_saved_RGB_bitwise_recompute(rgb_fixture):
    output, r, p, value = rgb_fixture; result = m.audit_rgb(output, r, p)
    assert result['frames'] == 1 and result['arrays'][0]['values_checked'] == value.size
    assert result['arrays'][0]['exact_float32_bits']
    assert result['original_preprocessed_file_SHA_in_v1_receipt'] is False


@pytest.mark.parametrize('damage', ['ulp', 'float64', 'wrong_shape', 'nan', 'trailing', 'channel',
    'missing', 'symlink', 'receipt', 'png', 'alias', 'mapping', 'latent_namespace'])
def test_saved_RGB_mutations_rejected(rgb_fixture, damage):
    output, r, p, value = rgb_fixture; path = output / 's0_fixed_b00_encode/preprocessed_pixels.npy'
    if damage == 'ulp': value.flat[0] = np.nextafter(value.flat[0], np.float32(np.inf)); save_array(path, value)
    elif damage == 'float64': save_array(path, value.astype(np.float64))
    elif damage == 'wrong_shape': save_array(path, value[0])
    elif damage == 'nan': value.flat[0] = np.nan; save_array(path, value)
    elif damage == 'trailing': path.write_bytes(path.read_bytes() + b'trailing')
    elif damage == 'channel': save_array(path, value[:, :, ::-1])
    elif damage == 'missing': path.rename(path.with_suffix('.absent'))
    elif damage == 'symlink':
        other = path.with_suffix('.other'); path.rename(other); path.symlink_to(other)
    elif damage == 'receipt': (path.parent / 'receipt.json').write_text('{}')
    elif damage == 'png': (output / 's0_fixed_plan_00.png').write_bytes(b'not PNG')
    elif damage == 'alias': (output / 's0_fixed_b00_plan_05.png').write_bytes(b'stale PNG')
    elif damage in ('mapping', 'latent_namespace'):
        f = output / 's0_fixed_b00_encoding.json'; er = json.loads(f.read_bytes())
        if damage == 'mapping': er['mapping']['actual_step'] = 5
        else: er['response']['latent']['path'] = '/other/native192.npy'
        f.write_bytes(m.base.raw(er)); r['rows'][0]['encodings'][0]['receipt_sha256'] = m.base.old.source.digest(er)
    with pytest.raises((ValueError, FileNotFoundError)): m.audit_rgb(output, r, p)


def test_cold_import_and_CLI_no_neural_or_physics():
    script = '''
import builtins
original = builtins.__import__
def guard(name, *a, **k):
    if name.split('.')[0] in ('torch','torchvision','gymnasium','pymunk','stable_worldmodel','stable_pretraining'):
        raise AssertionError(name)
    return original(name,*a,**k)
builtins.__import__=guard
from experiments.spectral_reachability import agent_p3_same_feedback_physical20_20261028_v2 as m
assert not hasattr(m,'run')
try: m.main(['--help'])
except SystemExit as e: assert e.code==0
'''
    proc = subprocess.run([sys.executable, '-B', '-c', script], cwd=m.base.ROOT,
                          text=True, capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_saved_native_stop_full_arrays_v2_roundtrip(tmp_path, monkeypatch):
    output = tmp_path / 'RAM_SAVED'; output.mkdir()
    r, store, _, _ = fixtures.execute_ram(fault='truncate', at=11, real_array_plan=True, path=str(output))
    p = store.test_p; p['sources'] = image_sources()['sources']
    p['modelbinding'] = m.base.moving.suffix.modelbinding(); p['expected_model_hash'] = p['modelbinding']['tensor_sha256']
    prepared = str(tmp_path / 'RAM_PREPARE.json'); prepare_sha = 'a' * 64
    hashes = dict(original_state_dict=p['expected_model_hash'], all_buffers='b' * 64)
    precision = dict(matmul_TF32=False, cudnn_TF32=False, autocast=False, math=True,
                     flash=False, efficient=False, cudnn_sdp=False)
    r.update(prepared=prepared, prepare_sha256=prepare_sha, source_manifest_sha256=m.digest(p['sources']),
        model_before=hashes, model_after=hashes, precision_before=precision, precision_after=precision,
        CUDA_peak_bytes=[0, 0], per_episode_neural=m.base.node_accounting(r['nodes']))
    store.files['execution_binding.json'] = m.base.raw(dict(prepared=prepared, prepare_sha256=prepare_sha))
    for name, payload in store.files.items():
        path = output / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
    # Mock visual tensors were zeros; replace only this explicit RAM fixture's
    # preprocessing arrays with source-defined arithmetic, no model execution.
    for row in r['rows']:
        for enc in row['encodings']:
            ns = row['episode_id'] + f"_b{enc['step']:02}"
            with Image.open(output / (row['episode_id'] + f"_plan_{enc['step']:02}.png")) as image:
                rgb = np.asarray(image).copy()
            save_array(output / (ns + '_encode/preprocessed_pixels.npy'),
                       m.preprocess(rgb, m.image_contract(p)['statistics']))
    report = output / 'report.json'; report.write_bytes(m.base.raw(r))
    monkeypatch.setattr(m.base, 'verify', lambda *a: p)
    monkeypatch.setattr(m.base, 'source_data', lambda *a: store.test_data)
    result = m.consume(output, m.sha(report), prepared, prepare_sha)
    assert result['complete'] and result['physical_census']['complete']
    assert result['RGB_preprocessing']['frames'] == 4
    assert result['original_consume']['completed20'] == 0
    assert not any(result[k] for k in ('P1_admission', 'P3_admission', 'freshness', 'scientific_admission'))
    assert len(result['original_consume']['audited']) == 4
