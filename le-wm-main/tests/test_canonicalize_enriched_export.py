from __future__ import annotations

import json

from experiments.empirical_spectral_audit.canonicalize_enriched_export import canonicalize


def test_canonicalize_pusht_recovers_declared_proprio_convention(tmp_path) -> None:
    row = {
        "episode_id": 0,
        "window_start": 0,
        "source_frame": 0,
        "source_state_raw": [1, 2, 3, 4, 5, 6, 7],
        "source_latent": [0.0, 1.0],
        "action_blocks_raw": [[0.1, 0.2]],
        "action_blocks_normalized": [[0.5, 0.6]],
        "latent_by_horizon": {"1": {"predicted": [0.0, 1.0], "target": [0.0, 1.0], "target_state_raw": [1, 2, 3, 4, 5, 6, 7]}},
    }
    source = tmp_path / "metrics.json"
    source.write_text(json.dumps({"enriched_records": [row], "protocol": {"split": "val"}}))
    payload = canonicalize(
        source,
        architecture="jepa_wm",
        checkpoint="ckpt",
        environment="PushT",
        seed="0",
        training_stage=50,
        action_distribution="shared/v1",
    )
    assert payload["record_schema"] == "enriched_learned_model.v1"
    assert payload["enriched_records"][0]["source_proprio_raw"] == [1.0, 2.0, 6.0, 7.0]


def test_canonicalize_rejects_vector_only_artifact(tmp_path) -> None:
    source = tmp_path / "vectors.json"
    source.write_text(json.dumps({"operator_fit_vectors": {"predicted": {"1": [[0.0]]}}}))
    try:
        canonicalize(source, architecture="dino_wm", checkpoint="ckpt", environment="PushT", seed="0", training_stage=50, action_distribution="shared/v1")
    except ValueError as exc:
        assert "enriched_records" in str(exc)
    else:
        raise AssertionError("vector-only exports must remain ineligible")
