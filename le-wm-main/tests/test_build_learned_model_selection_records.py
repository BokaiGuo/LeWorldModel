from __future__ import annotations

import json

import numpy as np

from experiments.empirical_spectral_audit.build_learned_model_selection_records import build_record
from experiments.empirical_spectral_audit.enriched_record_schema import make_payload


def _payload(path, *, episodes, offset: float) -> None:
    rows = []
    for episode in episodes:
        source = [float(episode), float(episode + 1), float(episode + 2)]
        target = [value + offset for value in source]
        rows.append({
            "sample_id": f"{episode}:0",
            "episode_id": episode,
            "source_frame": 0,
            "source_latent": source,
            "source_state_raw": [float(episode), 0.0],
            "source_proprio_raw": [float(episode)],
            "action_blocks_raw": [[0.0, 1.0]],
            "action_blocks_normalized": [[0.0, 1.0]],
            "latent_by_horizon": {
                "1": {"predicted": target, "target": target, "target_state_raw": [float(episode) + offset, 0.0]},
                "2": {"predicted": target, "target": target, "target_state_raw": [float(episode) + offset, 0.0]},
            },
        })
    path.write_text(json.dumps(make_payload(
        architecture="jepa_wm",
        checkpoint="ckpt.pth",
        environment="PointMaze",
        seed=0,
        training_stage=50,
        latent_dimension=3,
        action_distribution="pointmaze/observed/v1",
        split={"name": "diagnostic" if offset == 0.1 else "heldout", "episode_ids": list(episodes)},
        horizons=[1, 2],
        enriched_records=rows,
    )))


def test_builder_keeps_fit_and_outcome_disjoint(tmp_path) -> None:
    fit_path, outcome_path = tmp_path / "fit.json", tmp_path / "outcome.json"
    _payload(fit_path, episodes=range(0, 8), offset=0.1)
    _payload(outcome_path, episodes=range(8, 16), offset=0.2)
    record = build_record(fit_path, outcome_path, fit_horizon=1, outcome_horizon=2)
    assert record["record_schema"] == "real_checkpoint_selection.v1"
    diagnostics = record["record"]["diagnostics"]
    assert np.isfinite(diagnostics["prediction_mse"])
    assert len(diagnostics["predictor_spectrum"]) == 3
    assert np.isfinite(record["record"]["heldout"]["planning_loss"])
    assert record["protocol"]["runtime_successor_verification"] is False
    operator = diagnostics["transfer_operator"]
    assert operator["schema"] == "learned_transfer_operator.v1"
    assert set(operator["G_h"]) == {"1", "2"}
    assert operator["risk_envelope"]["bound_allowed_for_control"] is False

