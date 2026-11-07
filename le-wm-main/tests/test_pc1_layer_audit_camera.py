"""Unit/integrity checks; these are not empirical PC1 outcomes."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pc1_layer_audit import camera
from experiments.pc_input_permissions import file_hash


def test_fixed_panel_and_support_do_not_overlap():
    assert len(camera.ROOTS) == 32
    assert not set(camera.ROOTS[:16]) & set(camera.ROOTS[16:])
    assert camera.HORIZONS == (5, 10, 25)
    assert np.max(np.abs(camera.ACTIONS)) <= 1


def test_map_uses_initial_and_goal_only():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(32, 2, 11, 2, 4))
    truth = rng.normal(size=(32, 3, 3))
    result, mapping = camera.representation_audit(x, truth)
    changed = x.copy()
    changed[:, :, 1:10] += rng.normal(size=changed[:, :, 1:10].shape)
    _, other = camera.representation_audit(changed, truth + 7)
    np.testing.assert_array_equal(mapping.rotation, other.rotation)
    np.testing.assert_array_equal(mapping.source_mean, other.source_mean)
    assert result["mapping"]["fit_rows"] == 16*2*2
    assert result["mapping"]["within_source_score_max_error"] < 1e-12
    assert all("offline_retrieval" not in row["horizons"][0] for row in result["rows"][:16])
    assert all(row["horizons"][0]["neighbor_root"] < 1848 for row in result["rows"][16:])


def test_camera_changes_do_not_become_physics_samples():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(32, 1, 11, 2, 4)).repeat(2, axis=1)
    truth = rng.normal(size=(32, 3, 3))
    result, _ = camera.representation_audit(x, truth)
    assert result["summary"]["view_distance"]["mean"] == 0
    assert result["summary"]["view_distance"]["n_roots"] == 16
    assert all(h["view_disagreement_roots"] == 0 for h in result["summary"]["horizons"])


def test_stage_refuses_hash_drift(tmp_path):
    source = tmp_path / "dependency.txt"
    source.write_text("original")
    record = {"sources": {str(source): file_hash(source)}}
    camera.validate_sources(record)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source drift"):
        camera.validate_sources(record)


def test_existing_pusht_result_binds_saved_arrays_and_sources():
    path = camera.ROOT / "results/pc1_full_token_orthogonal_pusht_20261107/result.json"
    if not path.exists():
        pytest.skip("optional completed empirical artifact not installed")
    result = json.loads(path.read_text())
    assert file_hash(path.with_name("diagnostics.npz")) == result["arrays_sha256"]
    for name, digest in result["sources"].items():
        assert file_hash(name) == digest
    assert len(result["roots"]) == len(result["predictor_transfer"]) == 32
    assert len(result["representations"]) == 25
    with np.load(path.with_name("diagnostics.npz"), allow_pickle=False) as data:
        for representation in result["representations"].values():
            mapping = representation["mapping"]
            assert not set(mapping["calibration_roots"]) & set(mapping["evaluation_roots"])
            assert representation["summary"]["within_source_distance_max_error"] < 1e-7
        for r, row in enumerate(result["predictor_transfer"]):
            for v in range(2):
                for s in range(2):
                    case = row["conditions"][v*2+s]
                    choice = int(np.argmin(data["predicted_costs"][r, v, s]))
                    truth = data["native_costs"][r, v, s]
                    assert case["predicted"]["choice"] == choice
                    assert case["predicted"]["regret"] == pytest.approx(truth[choice]-truth.min())
