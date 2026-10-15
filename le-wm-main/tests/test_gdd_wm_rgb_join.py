import json

import numpy as np

from experiments.gdd_wm.join_rgb_realized_refs import join_artifact


def _write_fixture(tmp_path, *, bad_path=False, missing_key=False):
    candidate = tmp_path / "candidate.json"
    rows = [
        {
            "instance_id": "root0",
            "candidate_id": 0,
            "split": "outcome",
            "teacher_set_id": "set0",
            "teacher_model_id": "teacher",
            "teacher_score": -0.2,
            "teacher_rank": 1,
            "real_task_cost": 0.2,
            "real_success": True,
            "action_sequence": [[1.0, 0.0]],
            "realized_latent_ref": None,
        },
        {
            "instance_id": "root0",
            "candidate_id": 1,
            "split": "outcome",
            "teacher_set_id": "set0",
            "teacher_model_id": "teacher",
            "teacher_score": -0.4,
            "teacher_rank": 2,
            "real_task_cost": 0.4,
            "real_success": False,
            "action_sequence": [[0.0, 1.0]],
            "realized_latent_ref": None,
        },
    ]
    candidate.write_text(json.dumps({"metadata": {}, "records": rows}), encoding="utf-8")
    rgb_report = tmp_path / "rgb_report.json"
    rgb_report.write_text(
        json.dumps(
            {
                "protocol": {"candidate_dataset": str(candidate.resolve() if not bad_path else tmp_path / "other.json")},
                "gates": {
                    "realized_rgb_latents_available": True,
                    "pca_projected_successor_comparison": True,
                    "renderer_consistency": True,
                },
                "rows": [
                    {
                        "instance_id": "root0",
                        "candidate_id": 0,
                        "split": "outcome",
                        "realized_terminal_projection": [1.0, 2.0],
                    },
                    {
                        "instance_id": "root0",
                        "candidate_id": 1,
                        "split": "outcome",
                        "realized_terminal_projection": [3.0, 4.0],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    npz = tmp_path / "realized.npz"
    arrays = {"projection::root0/000": np.array([1.0, 2.0], dtype=np.float32)}
    if not missing_key:
        arrays["projection::root0/001"] = np.array([3.0, 4.0], dtype=np.float32)
    np.savez(npz, **arrays)
    return candidate, rgb_report, npz


def test_join_is_copy_only_and_covers_all_keys(tmp_path):
    candidate, report, npz = _write_fixture(tmp_path)
    result = join_artifact(candidate, report, npz, tmp_path / "out")
    assert result["status"] == "passed"
    joined = json.loads((tmp_path / "out" / "candidate_dataset.json").read_text())
    assert [row["realized_latent_ref"] for row in joined["records"]] == [
        "realized_latents.npz::projection::root0/000",
        "realized_latents.npz::projection::root0/001",
    ]
    original = json.loads(candidate.read_text())
    assert all(row["realized_latent_ref"] is None for row in original["records"])


def test_join_rejects_report_for_different_dataset(tmp_path):
    candidate, report, npz = _write_fixture(tmp_path, bad_path=True)
    result = join_artifact(candidate, report, npz, tmp_path / "out")
    assert result["status"] == "rejected_fail_closed"
    assert "does not exactly match" in result["errors"][0]
    assert not (tmp_path / "out" / "candidate_dataset.json").exists()


def test_join_rejects_incomplete_npz_coverage(tmp_path):
    candidate, report, npz = _write_fixture(tmp_path, missing_key=True)
    result = join_artifact(candidate, report, npz, tmp_path / "out")
    assert result["status"] == "rejected_fail_closed"
    assert "NPZ key is missing" in result["errors"][0]
