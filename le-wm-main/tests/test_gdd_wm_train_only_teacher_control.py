import json

import numpy as np

from experiments.gdd_wm.build_train_only_readout_teacher_package import build


def _dataset(path, context):
    rows = []
    for split, root in (("train", "r_train"), ("outcome", "r_test")):
        for candidate, action in (("a", [[0.1]]), ("b", [[-0.1]])):
            rows.append({
                "instance_id": root,
                "candidate_id": candidate,
                "action_sequence": action,
                "predicted_latent_ref": f"arrays.npz::predicted::{root}/{candidate}",
                "realized_latent_ref": f"realized::{root}/{candidate}",
                "real_task_cost": (0.2 if candidate == "a" else 0.8) + (0.01 if context else 0.0),
                "teacher_score": 99.0,
                "teacher_rank": 0,
                "teacher_model_id": "offline_environment_cost_v1",
                "teacher_set_id": f"{root}__source_{context}",
                "candidate_generator": "test",
                "split": split,
                "candidate_bank_hash": f"source_{context}",
                "candidate_bank_context_id": f"source_{context}",
                "source_latent_model": [0.0, 0.0],
                "predicted_latent_ref": f"arrays.npz::predicted::{root}/{candidate}",
            })
    path.write_text(json.dumps({"metadata": {"candidate_bank_hash": f"source_{context}"}, "records": rows}))
    return rows


def _arrays(path, context):
    values = {
        f"predicted::r_train/a": np.asarray([0.0 + context, 0.0]),
        f"predicted::r_train/b": np.asarray([1.0 + context, 0.0]),
        f"predicted::r_test/a": np.asarray([0.1 + context, 0.0]),
        f"predicted::r_test/b": np.asarray([0.9 + context, 0.0]),
    }
    np.savez(path, **values)


def test_train_only_teacher_is_frozen_and_not_source_cost_oracle(tmp_path):
    datasets = []
    arrays = []
    for context in (0, 1):
        dataset = tmp_path / f"dataset_{context}.json"
        array = tmp_path / f"arrays_{context}.npz"
        _dataset(dataset, context)
        _arrays(array, context)
        datasets.append(dataset)
        arrays.append(array)

    report = build(candidate_datasets=datasets, arrays_paths=arrays, output=tmp_path / "out")
    assert report["status"] == "completed_control_package"
    assert report["metadata"]["teacher_model_id"] == "train_only_predicted_latent_ridge_v1"
    payload = json.loads((tmp_path / "out" / "candidate_dataset.json").read_text())
    assert len(payload["records"]) == 8
    assert all(row["teacher_model_id"] == "train_only_predicted_latent_ridge_v1" for row in payload["records"])
    assert any(
        abs(float(row["teacher_score"]) + float(row["real_task_cost"])) > 1.0e-10
        for row in payload["records"]
    )

    action_report = build(
        candidate_datasets=datasets,
        arrays_paths=arrays,
        output=tmp_path / "out_action",
        feature_family="source_delta_action",
    )
    assert action_report["metadata"]["feature_family"] == "source_delta_action"
    assert action_report["summary"]["feature_dimension"] > 2

    residual_report = build(
        candidate_datasets=datasets,
        arrays_paths=arrays,
        output=tmp_path / "out_residual",
        feature_family="predicted_terminal",
        residualize_action=True,
    )
    assert residual_report["metadata"]["target_mode"] == "residualized_action_plus_feature"
    assert residual_report["summary"]["action_baseline_train_rmse"] is not None
