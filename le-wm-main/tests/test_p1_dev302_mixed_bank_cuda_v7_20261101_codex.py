import numpy as np

from experiments.spectral_reachability import (
    p1_dev302_mixed_bank_cuda_v7_20261101_codex as subject,
)


def test_v7_has_endpoint_preserving_delayed_support_variants():
    value = subject.protocol()
    rows = value["prefix_bank"]["candidates"]
    assert len(rows) == 13
    reference = rows[8]["rgb_dynamic_predicted_pusher_centers_xy"][-1]
    for row in rows[9:]:
        assert np.allclose(
            row["rgb_dynamic_predicted_pusher_centers_xy"][-1],
            reference,
            atol=1e-5,
        )
        assert row["contact_feasible_prior"] is True
    assert value["P1_admission"] is False
