"""Non-neural contracts only; no model load, CUDA, simulator or efficacy test."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import pusht_geometry_feedback_response_20261027_v1 as m
from experiments.spectral_reachability.p4_feedback_matched_20261027_v2 import advance_observed


def bank():
    rows = [dict(probe_id='zero',epsilon=None,axis=None,sign=0,logits=[0.]*10)]
    for i,e in enumerate((.05,.1)):
        for axis in range(10):
            for sign in (-1,1):
                u = np.zeros(10); u[axis] = e*sign
                rows.append(dict(probe_id=f'e{i}_a{axis}_{"minus" if sign < 0 else "plus"}',
                                 epsilon=e,axis=axis,sign=sign,logits=u.tolist()))
    return rows


def context():
    return dict(latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())


def test_exact_history_and_reencoding_comparison():
    imagined, actual = context(), context()
    actual['latent_history'][0][0] = 1e-6
    actual['latent_history'][2][0] = 17
    combined, delta = m.pair_contexts(imagined,actual)
    assert delta == 1e-6
    assert combined['latent_history'][:2] == imagined['latent_history'][:2]
    assert combined['latent_history'][2][0] == 17
    assert actual['latent_history'][0][0] == 1e-6  # caller not mutated
    actual['latent_history'][0][0] = .01
    with pytest.raises(ValueError,match='old latents'):
        m.pair_contexts(imagined,actual)


def test_incoming_history_must_be_exact_not_allclose():
    a, b = context(), context(); b['action_history'][0][0] = 1e-10
    with pytest.raises(ValueError,match='incoming'):
        m.pair_contexts(a,b)


def test_actual_shift_uses_exact_step5_and_prefix_packet():
    frames = np.stack([np.full((224,224,3),i,dtype=np.uint8) for i in range(3)])
    between = np.arange(20).reshape(2,5,2)/100
    prefix = np.arange(10).reshape(5,2)/20
    rgb = np.full((224,224,3),255,dtype=np.uint8)
    execution = dict(status='completed',completed_commands=5,remaining_commands_not_executed=0)
    shifted, raw = advance_observed(frames,between,rgb,prefix,execution)
    np.testing.assert_array_equal(shifted,np.stack([frames[1],frames[2],rgb]))
    np.testing.assert_array_equal(raw,np.stack([between[1],prefix]))
    execution['completed_commands'] = 4
    with pytest.raises(ValueError,match='partial'):
        advance_observed(frames,between,rgb,prefix,execution)


def test_tanh_fd_not_raw_jacobian_and_epsilon_sensitivity():
    b = bank(); a = np.zeros((192,10)); a[:10] = np.eye(10)
    vectors = {r['probe_id']:(a@np.tanh(r['logits'])).tolist() for r in b}
    k = m.central_fd(vectors,b,np.eye(192))
    for e, matrix in zip((.05,.1),k):
        np.testing.assert_allclose(matrix,a*np.tanh(e)/e,rtol=1e-14,atol=1e-14)
    assert np.linalg.norm(k[0]-k[1]) > 0


def test_nonzero_probe_compares_corresponding_physical_vector():
    b = bank(); real = {r['probe_id']:np.zeros(192).tolist() for r in b}
    pred = deepcopy(real); pred[b[1]['probe_id']][0] = 8
    result = m.analyze(pred,real,b,np.eye(192),np.zeros(192))
    assert result['zero_prediction_residual']['projected_mse'] == 0
    assert result['probes'][1]['projected_mse'] == 64/192
    assert result['grids'][0]['frobenius_error'] > 0
    assert result['grids'][0]['cosine'] is None
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('mutation',['missing','extra','nonfinite','shape'])
def test_bad_probe_vectors_fail_closed(mutation):
    b = bank(); v = {r['probe_id']:np.zeros(192).tolist() for r in b}
    if mutation == 'missing':
        del v['zero']
    elif mutation == 'extra':
        v['other'] = v['zero']
    elif mutation == 'nonfinite':
        v['zero'][0] = np.nan
    else:
        v['zero'] = [0]
    with pytest.raises(ValueError):
        m.central_fd(v,b,np.eye(192))


def test_receipt_protocol_tamper_fails_before_runtime(tmp_path):
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'prepare_receipt.json').write_text(json.dumps(dict(status='prepared_not_run',protocol_sha256='bad')))
    with pytest.raises(ValueError,match='protocol changed'):
        m.verify_frozen(tmp_path)


def test_receipt_requires_sources(tmp_path):
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'prepare_receipt.json').write_text(json.dumps(dict(status='prepared_not_run',
        protocol_sha256=m.sha(tmp_path/'protocol.json'),source_seals=[])))
    with pytest.raises(ValueError,match='sources'):
        m.verify_frozen(tmp_path)


def test_no_environment_or_fit_entrypoint():
    import inspect
    source = inspect.getsource(m)
    for forbidden in ('gym.make(', 'env.step(', 'autograd.functional', 'bounded_ridge(', 'optimizer.step('):
        assert forbidden not in source


def test_history_shape_nonfinite_rejected():
    a, b = context(), context(); b['latent_history'][0][0] = float('inf')
    with pytest.raises(ValueError):
        m.pair_contexts(a,b)
