import json

import pytest

from experiments.summarize_cube_temporal_action_ranking import summarize


def test_ranking_summary_recomputes_tie_aware_agreement(tmp_path):
    payload = {
        "protocol": {"oracle_tolerance_m": 1e-6},
        "held_out": {
            "records": [{
                "root_index": 0,
                "actual_candidate_error_m": [1.0, 1.0 + 1e-7],
                "latent_static": {"selected_index": 1},
                "latent_temporal": {"selected_index": 0},
            }]
        },
    }
    path = tmp_path / "ranking.json"
    path.write_text(json.dumps(payload))
    result = summarize([path], seed=2)
    assert result["aggregate"]["latent_static"]["oracle_agreement_rate"] == 1.0
    assert result["aggregate"]["paired_temporal_minus_static"]["mean_regret_delta_m"] == pytest.approx(-1e-7)
