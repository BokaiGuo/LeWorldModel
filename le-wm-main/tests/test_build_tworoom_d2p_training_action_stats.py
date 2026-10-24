import json

import numpy as np
import pytest

from experiments.spectral_reachability.build_tworoom_d2p_training_action_stats import (
    ENVIRONMENT,
    SCHEMA,
    training_packet_statistics,
    validate_training_semantic_receipt,
)


def test_raw_two_dimensional_statistics_are_tiled_in_time_major_packet_order():
    actions = np.asarray([[1.0, 3.0], [5.0, 7.0], [np.nan, np.nan]], dtype=np.float64)
    mean, std = training_packet_statistics(actions, action_block=5)
    np.testing.assert_allclose(mean, np.asarray([3.0, 5.0] * 5))
    np.testing.assert_allclose(std, np.asarray([np.sqrt(8.0), np.sqrt(8.0)] * 5))


def test_position_specific_legacy_calibration_is_rejected(tmp_path):
    payload = {
        "schema": SCHEMA,
        "status": "training_semantics_validated",
        "reports": [{
            "environment": ENVIRONMENT,
            "raw_action_dim": 2,
            "action_block": 5,
            "calibration_action_mean": [0.0, 0.0] * 5,
            "calibration_action_std": [1.0, 1.0] * 5,
            "all_packet_slots_share_physical_statistics": True,
        }],
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_training_semantic_receipt(receipt)["environment"] == ENVIRONMENT
    payload["reports"][0]["calibration_action_mean"][2] = 0.1
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="position-specific"):
        validate_training_semantic_receipt(receipt)
