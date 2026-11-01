import copy
import hashlib

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import codex_p1_null_same_action_error_cuda_20261030_v1 as exp


def row_fixture():
    raw = np.arange(30, dtype='<f4').reshape(15, 2) / 100
    warm = np.arange(20, dtype='<f4').reshape(10, 2) / 100
    mean, std = np.array([.1, -.2], np.float32), np.array([.5, 2], np.float32)
    hist = (warm.reshape(2, 10)-np.tile(mean, 5))/np.tile(std, 5)
    return dict(raw15_float32_hex=raw.tobytes().hex(),
                raw15_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                warmup_raw_float32_hex=warm.tobytes().hex(), action_mean=mean.tolist(),
                action_std=std.tolist(), action_history=hist.tolist(),
                model_history=np.zeros((3, 192), np.float32).tolist())


def test_action_normalization_and_original_history():
    raw, mean, std, history, z = exp.action_inputs(row_fixture())
    assert raw.shape == (3, 10) and z.shape == (3, 192) and history.shape == (2, 10)
    assert np.allclose((raw[0] - np.tile(mean, 5)) / np.tile(std, 5),
                       ((np.arange(10, dtype=np.float32)/100).reshape(5, 2)-mean).reshape(10)
                       / np.tile(std, 5))


@pytest.mark.parametrize('field,value', [
    ('raw15_sha256', '0'*64), ('action_std', [0, 1]), ('action_mean', [1]),
    ('model_history', [[0]]), ('warmup_raw_float32_hex', '00'*80),
])
def test_reject_action_or_history_tamper(field, value):
    row = row_fixture()
    row[field] = value
    with pytest.raises(ValueError):
        exp.action_inputs(row)


def test_refresh_context_exact_times():
    history = np.stack([np.full(192, n, np.float32) for n in (0, 1, 2)])
    actual = np.stack([np.full(192, n, np.float32) for n in (10, 11, 12, 13)])
    assert exp.contexts(history, actual)[:, :, 0].tolist() == [[1, 10, 11], [10, 11, 12]]
    with pytest.raises(ValueError):
        exp.contexts(history, np.full((4, 192), np.nan))


def test_action_history_packet_order():
    history = np.stack([np.full(10, n, np.float32) for n in (1, 2)])
    tokens = np.stack([np.full(10, n, np.float32) for n in (3, 4, 5)])
    assert exp.action_contexts(history, tokens)[:, :, 0].tolist() == [
        [1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_time_matched_latent_errors():
    actual = np.stack([np.full(192, x, np.float32) for x in (0, 1, 2, 3)])
    opens = np.stack([np.full(192, x, np.float32) for x in (3, 5, 7)])
    refreshed = np.stack([np.full(192, x, np.float32) for x in (3, 5)])
    result = exp.compare_latents(actual, opens, refreshed)
    assert [r['step'] for r in result] == [5, 10, 15]
    assert [r['open_mse'] for r in result] == [4, 9, 16]
    assert [r['refreshed_mse'] for r in result] == [None, 1, 4]
    assert result[2]['refreshed_l2'] == pytest.approx(2 * np.sqrt(192))


def test_rgb_file_pixel_and_mode_integrity(tmp_path):
    path = tmp_path / 'frame.png'
    array = np.zeros((224, 224, 3), np.uint8)
    Image.fromarray(array).save(path)
    item = dict(path=str(path), file_sha256=exp.sha(path),
                pixel_sha256=hashlib.sha256(array.tobytes()).hexdigest())
    assert np.array_equal(exp.rgb(item), array)
    bad = copy.deepcopy(item)
    bad['pixel_sha256'] = '0'*64
    with pytest.raises(ValueError, match='pixel'):
        exp.rgb(bad)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='file'):
        exp.rgb(item)


def test_exclusive_write_does_not_replace(tmp_path):
    path = tmp_path / 'claim'
    exp.write_exclusive(path, b'first')
    with pytest.raises(FileExistsError):
        exp.write_exclusive(path, b'second')
    assert path.read_bytes() == b'first'


def test_failed_run_keeps_exclusive_marker(tmp_path, monkeypatch):
    (tmp_path / 'results').mkdir()
    monkeypatch.setattr(exp, 'ROOT', tmp_path)
    prepared = tmp_path / 'results' / (exp.NAME + '_prepare_fixture.json')
    payload = exp.encode_json({'bindings': {}})
    prepared.write_bytes(payload)
    monkeypatch.setattr(exp, 'material', lambda: {'bindings': {}})
    def fail(_):
        raise RuntimeError('synthetic failure')
    monkeypatch.setattr(exp, 'cuda_predictions', fail)
    output = tmp_path / 'results' / (exp.NAME + '_run_fixture.json')
    digest = hashlib.sha256(payload).hexdigest()
    with pytest.raises(RuntimeError, match='synthetic failure'):
        exp.run(str(prepared), digest, str(output))
    assert output.read_bytes() == b''
    with pytest.raises(ValueError, match='exclusive'):
        exp.run(str(prepared), digest, str(output))


def test_wrong_report_sha_fails_before_consumer(monkeypatch):
    monkeypatch.setattr(exp, 'REPORT_SHA', '0'*64)
    with pytest.raises(ValueError, match='physical report SHA'):
        exp.material()


def test_real_saved_physical_and_model_sources_cpu_only():
    data = exp.material()
    assert data['fixed_rows'] == len(data['rows']) == 16
    assert data['physical_execution'] is False and data['training'] is False
    assert data['predictor_calls'] == 80 and data['image_encoder_calls'] == 64
    for i, row in enumerate(data['rows']):
        assert row['index'] == i
        assert [x['step'] for x in row['images']] == [0, 5, 10, 15]
        raw = bytes.fromhex(row['raw15_float32_hex'])
        if row['role'] == 'zero_prefix':
            assert raw[:40] == bytes(40)
            assert raw[40:] == bytes.fromhex(data['rows'][i-1]['raw15_float32_hex'])[40:]
        exp.action_inputs(row)


def test_cpu_disabled_is_not_a_fallback(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    with pytest.raises(ValueError, match='CUDA explicitly disabled'):
        exp.cuda_predictions({})
