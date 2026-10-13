import json

from experiments.empirical_spectral_audit.audit_dino_vector_artifact_join import audit


def _payload():
    rows = []
    for episode in range(8):
        rows.append({
            "sample_id": f"{episode}:0",
            "episode_id": episode,
            "source_frame": 10,
            "source_state_raw": [0.0, 0.0],
            "source_latent": [0.0],
            "action_blocks_raw": [[0.0]],
            "latent_by_horizon": {
                h: {"predicted": [0.0], "target": [0.0]}
                for h in ("1", "2", "4")
            },
        })
    return {
        "protocol": {
            "episodes_used": list(range(8)),
            "horizons": [1, 2, 4],
            "context_length": 3,
            "frame_stride": 5,
            "action_mode": "observed",
        },
        "operator_fit_vectors": {
            "predicted": {h: [[0.0] for _ in range(8)] for h in ("1", "2", "4")},
            "target": {h: [[0.0] for _ in range(8)] for h in ("1", "2", "4")},
        },
    }, rows


def test_vector_only_artifact_is_rejected_for_formal_join(tmp_path):
    vector, rows = _payload()
    vector_path = tmp_path / "vector.json"
    enriched_path = tmp_path / "enriched.json"
    vector_path.write_text(json.dumps(vector))
    enriched_path.write_text(json.dumps({"enriched_records": rows}))

    report = audit(vector_path, enriched_path)

    assert report["decision"]["formal_action_conditioned_join"] == "rejected_fail_closed"
    assert report["decision"]["vector_only_joined_diagnostic"] == "available"
    assert report["metadata"]["vector_has_sample_ids"] is False
