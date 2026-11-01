import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_null_same_action_saved_audit_20261030_v1 as audit


def fixture():
    exp = audit.exp
    raw = np.zeros((15, 2), dtype='<f4')
    warm = np.zeros((10, 2), dtype='<f4')
    model_hash = {'original_state_dict': 'a'*64, 'all_buffers': 'b'*64}
    source, rows = [], []
    for i in range(16):
        row = dict(index=i, task_id=f'task-{i}', role='zero_prefix', slot=i//4,
                   root_slot=i//8, condition='off', arm='arm', physical_root_sha256='c'*64,
                   raw15_sha256=hashlib.sha256(raw.tobytes()).hexdigest(), images=[],
                   native_status='horizon_completed', duplicate_action_task_group=i,
                   raw15_float32_hex=raw.tobytes().hex(), warmup_raw_float32_hex=warm.tobytes().hex(),
                   action_mean=[0., 0.], action_std=[1., 1.],
                   action_history=np.zeros((2, 10), np.float32).tolist(),
                   model_history=np.zeros((3, 192), np.float32).tolist())
        actual = np.zeros((4, 192), np.float32)
        opens, refreshed = np.ones((3, 192), np.float32), np.ones((2, 192), np.float32)
        result = {k: copy.deepcopy(row[k]) for k in audit.FIELDS}
        result.update(actual_root_latent=actual[0].tolist(),
                      normalized_tokens=np.zeros((3, 10), np.float32).tolist(),
                      open_action_contexts=np.zeros((3, 3, 10), np.float32).tolist(),
                      comparisons=exp.compare_latents(actual, opens, refreshed))
        source.append(row)
        rows.append(result)
    boundary = dict(training=False, physical_execution=False, P1_admission=False,
                    independent_confirmation=False, freshness=False)
    prepared = dict(schema=exp.NAME+'.prepare', fixed_rows=16, rows=source, model_hash=model_hash,
                    **boundary)
    report = dict(schema=exp.NAME+'.report', fixed_rows=16, rows=rows, prepared='/fake/prepare',
                  prepare_sha256='d'*64, model_before=model_hash, model_after=copy.deepcopy(model_hash),
                  host={'device': 'cuda:0'},
                  calls={'encoder': 64, 'projector': 64, 'predictor': 80, 'action_encoder': 80},
                  **boundary)
    return report, prepared


def test_full_synthetic_report_is_checked():
    report, prepared = fixture()
    result = audit.check(report, prepared, '/fake/prepare', 'd'*64)
    assert result['rows'] == 16 and result['P1_admission'] is False
    assert result['report_claimed_cuda_not_independently_attested'] is True


@pytest.mark.parametrize('change', ['missing_row', 'reorder', 'task', 'action', 'history',
                                    'time', 'error', 'refreshed', 'count', 'model', 'device',
                                    'admission', 'partial'])
def test_tampered_report_is_rejected(change):
    report, prepared = fixture()
    if change == 'missing_row':
        report['rows'].pop()
    elif change == 'reorder':
        report['rows'][0], report['rows'][1] = report['rows'][1], report['rows'][0]
    elif change == 'task':
        report['rows'][0]['task_id'] = 'other'
    elif change == 'action':
        report['rows'][0]['normalized_tokens'][0][0] = .5
    elif change == 'history':
        report['rows'][0]['open_action_contexts'][1][0][0] = .5
    elif change == 'time':
        report['rows'][0]['comparisons'][1]['step'] = 15
    elif change == 'error':
        report['rows'][0]['comparisons'][1]['open_mse'] = 2.
    elif change == 'refreshed':
        report['rows'][0]['comparisons'][0]['refreshed_prediction'] = [0]*192
    elif change == 'count':
        report['calls']['predictor'] = 79
    elif change == 'model':
        report['model_after']['all_buffers'] = '0'*64
    elif change == 'device':
        report['host']['device'] = 'cpu'
    elif change == 'admission':
        report['P1_admission'] = True
    else:
        report['rows'][0]['comparisons'].pop()
    with pytest.raises((ValueError, KeyError, TypeError)):
        audit.check(report, prepared, '/fake/prepare', 'd'*64)


def test_external_sha_and_empty_partial_rejected(tmp_path, monkeypatch):
    exp = audit.exp
    (tmp_path/'results').mkdir()
    monkeypatch.setattr(exp, 'ROOT', tmp_path)
    path = tmp_path/'results'/(exp.NAME+'_run_fixture.json')
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='digest'):
        audit.read_bounded(path, 'f'*64, 'run')
    with pytest.raises(ValueError):
        audit.read_bounded(path, hashlib.sha256(b'').hexdigest(), 'run')


def test_pinned_files_and_source_rebuild(tmp_path, monkeypatch):
    exp = audit.exp
    (tmp_path/'results').mkdir()
    monkeypatch.setattr(exp, 'ROOT', tmp_path)
    report, prepared = fixture()
    prep_path = tmp_path/'results'/(exp.NAME+'_prepare_fixture.json')
    report_path = tmp_path/'results'/(exp.NAME+'_run_fixture.json')
    prep_data = exp.encode_json(prepared)
    prep_sha = hashlib.sha256(prep_data).hexdigest()
    prep_path.write_bytes(prep_data)
    report.update(prepared=str(prep_path), prepare_sha256=prep_sha)
    report_data = exp.encode_json(report)
    report_sha = hashlib.sha256(report_data).hexdigest()
    report_path.write_bytes(report_data)
    monkeypatch.setattr(exp, 'material', lambda: prepared)
    assert audit.audit(str(report_path), report_sha, str(prep_path), prep_sha)['rows'] == 16
    monkeypatch.setattr(exp, 'material', lambda: {})
    with pytest.raises(ValueError, match='sources differ'):
        audit.audit(str(report_path), report_sha, str(prep_path), prep_sha)


def test_cli_failure_is_nonzero():
    cmd = [sys.executable, '-B', '-s', '-m', audit.__name__, '--report', '/no-file',
           '--report-sha', '0'*64, '--prepared', '/no-file', '--prepare-sha', '0'*64]
    proc = subprocess.run(cmd, cwd=audit.exp.ROOT, capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0
    assert json.loads(proc.stderr)['status'] == 'blocked_invalid'
