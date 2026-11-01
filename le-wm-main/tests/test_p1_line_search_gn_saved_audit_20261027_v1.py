"""Real saved-array arithmetic and adversarial mutations only; no neural imports."""
from copy import deepcopy
import gzip
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p1_line_search_gn_saved_audit_20261027_v1 as a


@pytest.fixture(scope='module')
def bundle(): return a.load_bundle()


def find(b, kind, occurrence=0):
    return [e for e in b['events'] if e['kind'] == kind][occurrence]


def test_actual_full_saved_recompute(bundle):
    r = a.audit(bundle)
    assert r['all_checks'] and r['event_count'] == 5660
    assert r['raw_derivative_rows'] == 1536 and r['fullH2_rollouts'] == 70 and r['AD_fullH2_forwards'] == 8
    assert r['resources']['predict'] == r['resources']['action_encoder'] == dict(attempted=156, completed=156)
    assert [x['final_cost'] for x in r['roots']] == [1.1405115382133657, 1.9203170978443598]
    assert all([s['selected_scale'] for s in x['steps']] == [1., 1., .25, .25] for x in r['roots'])
    assert r['new_neural_calls'] == r['new_physical_steps'] == 0 and not r['P1_admission']


@pytest.mark.parametrize('mutant', [
    'raw_row_value', 'raw_row_missing', 'raw_row_order', 'double_chain', 'stale_J_point',
    'missing_scale', 'wrong_scale_logits', 'raw_commands', 'history_value', 'history_layout',
    'latent_order', 'sameQ_cost', 'return_missing', 'return_duplicate', 'extra_forward',
    'resource_summary', 'root_result', 'selection', 'root_identity', 'step_order', 'rootraw',
])
def test_adversarial_saved_mutations_rejected(bundle, mutant):
    b = deepcopy(bundle)
    if mutant == 'raw_row_value': find(b, 'jacobian_row')['raw_control_derivative'][0] += .001
    elif mutant == 'raw_row_missing': b['events'].remove(find(b, 'jacobian_row'))
    elif mutant == 'raw_row_order': find(b, 'jacobian_row')['row'] = 1
    elif mutant == 'double_chain':
        joint = find(b, 'joint_logit_jacobian', 1)
        d = find(b, 'derivative_complete', 1)['derivative']
        factor = 1-np.tanh(np.asarray(d['at_logits'], np.float32)).reshape(20).astype(float)**2
        joint['jacobian'] = (np.asarray(joint['jacobian'])*factor).tolist()
    elif mutant == 'stale_J_point': find(b, 'jacobian_forward_complete', 1)['logits'] = np.zeros((2, 10)).tolist()
    elif mutant == 'missing_scale': find(b, 'line_proposals_frozen')['scales'].pop()
    elif mutant == 'wrong_scale_logits': find(b, 'line_proposals_frozen')['logits'][1][0][0] += .01
    elif mutant == 'raw_commands': find(b, 'evaluation_request')['commands'][0][0] += .01
    elif mutant == 'history_value': find(b, 'native_packet')['context']['action_history'][0][0] += .01
    elif mutant == 'history_layout': find(b, 'native_packet')['context']['layout'] = 'padded'
    elif mutant == 'latent_order': find(b, 'native_packet')['context']['latent_history'][0][0] += .01
    elif mutant == 'sameQ_cost': find(b, 'evaluation_complete')['prediction']['cost'] += .01
    elif mutant == 'return_missing': b['events'].remove(find(b, 'operation_completed'))
    elif mutant == 'return_duplicate':
        e = find(b, 'operation_completed'); b['events'].insert(b['events'].index(e), deepcopy(e))
    elif mutant == 'extra_forward':
        e = deepcopy(find(b, 'operation_attempted')); b['events'].append(e)
    elif mutant == 'resource_summary': b['top']['resources']['predict']['completed'] -= 1
    elif mutant == 'root_result': b['roots'][0]['final_prediction']['commands'][0][0] += .1
    elif mutant == 'selection': find(b, 'selection_complete')['selected_index'] = 0
    elif mutant == 'root_identity': find(b, 'native_packet')['root_sha256'] = '0'*64
    elif mutant == 'step_order': find(b, 'native_packet')['packet'] = 1
    elif mutant == 'rootraw': b['tasks'][0]['root_raw_action_history'][0][0] += .1
    with pytest.raises(ValueError): a.audit(b)


def test_input_SHA_rejects_without_reading_other_inputs(tmp_path):
    p = tmp_path/'report.json'; p.write_bytes(b'{}')
    with pytest.raises(ValueError, match='SHA mismatch'): a.load_bundle(p, '0'*64)


def test_original_source_drift_rejected(bundle, monkeypatch):
    path = next(p for p in bundle['bindings'] if p.endswith('p1_line_search_gn_cuda_20261027_v1.py'))
    original = a.sha
    monkeypatch.setattr(a, 'sha', lambda p: '0'*64 if str(p) == path else original(p))
    with pytest.raises(ValueError, match='SHA mismatch'): a.load_bundle()


def test_gzip_truncation_crc_and_trailing_data_rejected(tmp_path):
    p = tmp_path/'tiny.json.gz'; encoded = gzip.compress(b'{"saved":true}')
    p.write_bytes(encoded[:-5])
    with pytest.raises(ValueError): a.read(p)
    p.write_bytes(encoded+b'garbage')
    with pytest.raises(ValueError): a.read(p)
    broken = bytearray(encoded); broken[-8] ^= 1; p.write_bytes(broken)
    with pytest.raises(a.zlib.error): a.read(p)
    p.write_bytes(encoded); assert a.read(p) == dict(saved=True)


def test_small_exclusive_saved_output(tmp_path):
    p = tmp_path/'audit.json'; result = a.execute(a.INPUT, a.INPUT_SHA, p)
    assert result['bytes'] < 256*1024 and result['sha256'] == a.sha(p)
    assert json.loads(p.read_bytes())['all_checks'] is True
    with pytest.raises(ValueError, match='exclusive'): a.execute(a.INPUT, a.INPUT_SHA, p)


def test_postread_TOCTOU_detected(bundle, monkeypatch):
    path = next(iter(bundle['bindings'])); original = a.sha
    monkeypatch.setattr(a, 'sha', lambda p: '0'*64 if str(p) == path else original(p))
    with pytest.raises(ValueError, match='TOCTOU'): a.audit(bundle)
