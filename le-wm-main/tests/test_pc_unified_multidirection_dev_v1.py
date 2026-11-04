import numpy as np
import pytest

from experiments.spectral_reachability import pc_unified_multidirection_dev_v1 as api


def test_meter_preserves_model_dtype_and_charges_actual_horizon():
    def predict(actions):
        return np.ones((len(actions), 2), dtype=np.float32), np.ones(len(actions), dtype=np.float32)

    meter = api._Meter(predict)
    pos, scores = meter.predict(np.zeros((3, 10, 2)))
    assert pos.dtype == np.float32 and scores.dtype == np.float32
    assert (meter.calls, meter.samples, meter.token_samples) == (1, 3, 6)


def test_rejects_invalid_callback_response_and_bad_carry():
    meter = api._Meter(lambda actions: (np.full((len(actions), 2), np.nan), np.zeros(len(actions))))
    with pytest.raises(ValueError, match="nonfinite"):
        meter.predict(np.zeros((1, 5, 2)))
    assert meter.calls == meter.token_samples == 0
    with pytest.raises(ValueError, match="invalid carry"):
        api.compile_decision(root=0, step=1, target=[0., 0.], predict=lambda x: None,
                             context_id="ctx", carry=np.zeros((5, 2)))


def test_rejects_out_of_domain_and_unmetered_overflow():
    meter = api._Meter(lambda x: (np.zeros((len(x), 2)), np.zeros(len(x))))
    with pytest.raises(ValueError, match="action batch"):
        meter.predict(np.full((1, 5, 2), 1.1))
    with pytest.raises(ValueError, match="token cap"):
        meter.predict(np.zeros((161, 15, 2)))
    assert meter.calls == meter.token_samples == 0
