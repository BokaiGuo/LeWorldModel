import json

import numpy as np

from experiments.decision_benchmark.audit_candidate_task_contrast import audit
from experiments.empirical_spectral_audit.enriched_record_schema import make_payload


def test_offline_candidate_contrast_keeps_rgb_boundary(tmp_path):
    dataset = tmp_path / "candidate.json"
    arrays = tmp_path / "arrays.npz"
    fit_export = tmp_path / "fit.json"
    records = []
    values = {}
    enriched = []
    for split, root in (("calibration", "c"), ("outcome", "o")):
        for candidate in range(2):
            key = f"{split}::{root}/{candidate}"
            values[key] = np.asarray([float(candidate), 0.0])
            records.append({
                "instance_id": root,
                "split": split,
                "candidate_id": candidate,
                "predicted_horizon": 4,
                "predicted_latent_ref": f"arrays.npz::{key}",
                "source_state_raw": [0.0, 0.0],
                "goal_raw": [1.0, 0.0],
                "real_task_cost": float(candidate),
            })
    for i in range(24):
        enriched.append({
            "sample_id": f"fit:{i}",
            "episode_id": i,
            "source_frame": 0,
            "source_latent": [float(i), 0.0],
            "source_state_raw": [0.0, 0.0],
            "source_proprio_raw": [0.0],
            "action_blocks_raw": [[0.1, 0.0]],
            "action_blocks_normalized": [[0.1, 0.0]],
            "latent_by_horizon": {
                str(h): {"predicted": [float(i), 0.0], "target": [float(i), 0.0], "target_state_raw": [0.0, 0.0]}
                for h in (1, 2, 4)
            },
        })
    dataset.write_text(json.dumps({"records": records}))
    fit_export.write_text(json.dumps(make_payload(
        architecture="fixture", checkpoint="fixture", environment="PointMaze", seed=0,
        training_stage=1, latent_dimension=2, action_distribution="fixture",
        split={"name": "fit"}, horizons=[1, 2, 4], enriched_records=enriched,
    )))
    np.savez(arrays, **values)
    report = audit(dataset, arrays, fit_export)
    assert report["protocol"]["rgb_realized_latent_available"] is False
    assert report["outcome"]["pair_count"] == 1
    assert report["certificate_preview"]["released_root_count"] == 0
