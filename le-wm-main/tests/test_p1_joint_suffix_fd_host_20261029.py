import numpy as np
import pytest

from experiments.spectral_reachability import p1_joint_suffix_fd_host_20261029 as producer
from experiments.spectral_reachability import p1_joint_suffix_fd_consume_20261029 as consumer

estimate = producer.estimate


def test_directional_derivative_exact_linear():
    j = np.arange(1920, dtype=float).reshape(192, 10)/1000
    d = np.arange(10, dtype=np.float32)/20
    base = np.arange(192, dtype=float)
    eps = .01
    result = estimate(j, d, eps, base+eps*(j@d), base-eps*(j@d))
    assert result['relative_error'] < 1e-10


def test_invalid_shape_rejected():
    with pytest.raises(ValueError):
        estimate(np.zeros((1, 10)), np.zeros(10), .1, np.zeros(192), np.zeros(192))


def test_consumer_recomputes_and_rejects_forgery(monkeypatch):
    direction = np.zeros(10, dtype=np.float32)
    jac = np.zeros((192, 10))
    zero = np.zeros(192)
    c = {'suffix': direction.reshape(5, 2).tolist(),
         'signal': {'jacobian': jac.tolist(), 'nominal_context': {'latent_history': [zero.tolist()]}},
         'end': {'latent_history': [zero.tolist()]}}
    monkeypatch.setattr(producer, 'candidates', lambda *_: [(i//4, i%4, {}, c) for i in range(16)])
    scales = [dict(scale=s, epsilon_float32=float(np.float32(s)), plus=zero.tolist(), minus=zero.tolist(),
                   fd_l2=0., reference_l2=0., error_l2=0., relative_error=None) for s in producer.SCALES]
    rows = [dict(slot=i//4, iteration=i%4, direction=direction.tolist(), zero=zero.tolist(), full=zero.tolist(),
                 nominal_max_abs=0., full_max_abs=0., scales=[dict(x) for x in scales]) for i in range(16)]
    result = dict(source_report_sha256=producer.REPORT_SHA, source_receipt_sha256=producer.RECEIPT_SHA,
                  model_calls=160, rows=rows)
    assert consumer.verify(result, {}, {})['verified_saved_arithmetic']
    rows[0]['scales'][0]['error_l2'] = 1.
    with pytest.raises(ValueError, match='finite difference mismatch'):
        consumer.verify(result, {}, {})
