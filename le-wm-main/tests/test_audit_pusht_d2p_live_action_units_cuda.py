import copy

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import audit_pusht_d2p_live_action_units_cuda as a


def fixture():
    return [{"episode": i, "observed_vector": [0.]*192,
        "predicted_vectors": {"extra_div100": [1.]*192, "live_units": [.5]*192}} for i in range(16)]


def test_units_contrast_is_exactly_one_extra_divisor_not_packet_interleaving():
    raw = torch.tensor(np.arange(10).reshape(1, 5, 2)/10, requires_grad=True)
    kwargs = {"mean": [0., 0.], "std": [1., 1.]}
    correct = a.model_packets(raw, "live_units", **kwargs)
    old = a.model_packets(raw, "extra_div100", **kwargs)
    torch.testing.assert_close(correct, raw.reshape(1, 10))
    torch.testing.assert_close(old, correct/100)
    correct.sum().backward()
    assert torch.equal(raw.grad, torch.ones_like(raw))


def test_full_root_identity_and_no_p1_admission():
    rows = fixture()
    result = a.summarize(rows)
    assert result["improved"] == 16
    assert result["p1_admitted"] is False
    assert result["training_stats_provenance_certified"] is False
    rows.reverse()
    assert a.summarize(rows) == result


@pytest.mark.parametrize("fault", ["partial", "duplicate", "vector", "nan", "variant"])
def test_partial_or_changed_forecasts_fail_closed(fault):
    rows = copy.deepcopy(fixture())
    if fault == "partial": rows.pop()
    elif fault == "duplicate": rows[-1]["episode"] = 0
    elif fault == "vector": rows[0]["observed_vector"].pop()
    elif fault == "nan": rows[0]["predicted_vectors"]["live_units"][0] = float("nan")
    else: rows[0]["predicted_vectors"].pop("extra_div100")
    with pytest.raises(ValueError): a.summarize(rows)
