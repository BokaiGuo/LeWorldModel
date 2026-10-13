from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.empirical_spectral_audit.enriched_record_schema import (
    make_payload,
    validate_payload,
)
from experiments.empirical_spectral_audit.visual_adapter_export_metadata import (
    architecture_name,
    training_stage,
)


def _row(sample_id: str, episode: int, dim: int = 3) -> dict:
    vector = [float(i + episode) for i in range(dim)]
    return {
        "sample_id": sample_id,
        "episode_id": episode,
        "source_frame": 10,
        "source_latent": vector,
        "source_state_raw": [0.0, 1.0],
        "source_proprio_raw": [0.0],
        "action_blocks_raw": [[0.1, 0.2], [0.2, 0.3]],
        "action_blocks_normalized": [[0.5, 0.6], [0.6, 0.7]],
        "latent_by_horizon": {
            "1": {"predicted": vector, "target": vector, "target_state_raw": [1.0, 1.0]},
            "2": {"predicted": vector, "target": vector, "target_state_raw": [2.0, 1.0]},
        },
    }


def test_schema_validates_cross_architecture_payload() -> None:
    payload = make_payload(
        architecture="jepa_wm",
        checkpoint="checkpoint.pth",
        environment="PointMaze",
        seed=0,
        training_stage=50,
        latent_dimension=3,
        action_distribution="pointmaze/observed/v1",
        split={"name": "diagnostic", "episode_ids": [0]},
        horizons=[1, 2],
        enriched_records=[_row("0:0", 0)],
    )
    assert payload["record_schema"] == "enriched_learned_model.v1"
    assert payload["latent_dimension"] == 3
    assert validate_payload(json.loads(json.dumps(payload))) == payload


def test_schema_rejects_missing_source_latent() -> None:
    row = _row("0:0", 0)
    del row["source_latent"]
    with pytest.raises(ValueError, match="source_latent"):
        make_payload(
            architecture="dino_wm",
            checkpoint="checkpoint.pth",
            environment="PointMaze",
            seed=0,
            training_stage=50,
            latent_dimension=3,
            action_distribution="pointmaze/observed/v1",
            split={"name": "diagnostic", "episode_ids": [0]},
            horizons=[1, 2],
            enriched_records=[row],
        )


def test_snapshot_export_preserves_architecture_and_stage() -> None:
    checkpoint = Path(
        "results/strong_claim_protocol_v2_20261009/checkpoints/phase_legacy/"
        "seed_8411/snapshots/epoch_0020.pt"
    )
    saved = {"epochs": 100}
    assert architecture_name(saved, checkpoint) == "causal_visual_phase_legacy"
    assert training_stage(checkpoint, saved, None) == 20


def test_final_export_uses_saved_total_stage() -> None:
    checkpoint = Path(
        "results/strong_claim_protocol_v2_20261009/checkpoints/phase_legacy/"
        "seed_8411/causal_visual_world_model.pt"
    )
    assert architecture_name({}, checkpoint) == "causal_visual_phase_legacy"
    assert training_stage(checkpoint, {"epochs": 100}, None) == 100
