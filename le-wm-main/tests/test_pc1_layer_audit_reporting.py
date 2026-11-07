import json

import numpy as np
import pytest

from experiments.pc1_layer_audit import camera, summarize
from experiments.pc_input_permissions import file_hash


def test_signs_exclude_physical_ties_and_count_prediction_ties():
    assert summarize.sign_agreement([1., 0., 2.], [5., 3., 8.]) == 1
    assert summarize.sign_agreement([1., 1., 1.], [5., 3., 8.]) == .5
    assert summarize.sign_agreement([0., 1.], [2., 2.]) is None
    assert summarize.grouped([None, None])["n_roots"] == 0


def test_completed_camera_artifact_contract_when_installed():
    folder = summarize.CAMERA.parent
    if not summarize.CAMERA.exists():
        pytest.skip("empirical artifact not installed")
    for stage in ("prepare", "predict", "replay", "analyze"):
        path = folder / f"{stage}.json"
        data = json.loads(path.read_text())
        assert data["track"] == "official_state_assisted"
        assert file_hash(path.with_suffix(".npz")) == data["arrays_sha256"]
        for source, digest in data["sources"].items():
            assert file_hash(source) == digest
    pred = json.loads((folder / "predict.json").read_text())
    receipts = pred["receipts"]
    assert receipts["execution_status"] == "model_calls_recorded"
    # Check runtime shape/state inputs in successful call receipts.
    assert len(receipts["input_receipts"]) == 64
    with np.load(folder / "prepare.npz", allow_pickle=False) as prepared, np.load(folder / "replay.npz", allow_pickle=False) as replay:
        np.testing.assert_array_equal(replay["traces"][:, :, 0], replay["traces"][:, :, 1])
        np.testing.assert_array_equal(replay["images"][:, :, :, 0],
                                     np.broadcast_to(prepared["frames"][:, None, 0], replay["images"][:, :, :, 0].shape))
        expected = np.linalg.norm(replay["traces"][:, :, 0][:, :, camera.HORIZONS, :2] -
                                  prepared["states"][:, None, None, 1, :2], axis=-1).transpose(0, 2, 1)
        np.testing.assert_array_equal(expected, replay["physical_costs"])
    analysis = json.loads(summarize.CAMERA.read_text())
    json.dumps(summarize.summarize_camera(analysis), allow_nan=False)
    assert analysis["encoding_batch_size"] == 1
    assert analysis["goal_encoding_max_error"] <= 1e-4
    assert len(analysis["representations"]) == 25
    assert len(analysis["predictor_transfer"]) == 32
    for value in analysis["representations"].values():
        assert value["mapping"]["fit_frames"] == ["initial", "goal"]
        assert not set(value["mapping"]["calibration_roots"]) & set(value["mapping"]["evaluation_roots"])


def test_paired_render_repair_preserves_actions_and_physics_when_installed():
    folder = camera.ROOT / "results/pc1_render_alignment_comparison_20261107"
    if not (folder / "result.json").exists():
        pytest.skip("paired artifact not installed")
    result = json.loads((folder / "result.json").read_text())
    assert result["physics_max_error"] == 0
    assert file_hash(folder / "cached_render.npz") == result["arrays_sha256"]
    assert file_hash(folder / "cached_features.npz") == result["features_sha256"]
    with np.load(folder / "cached_render.npz", allow_pickle=False) as old, np.load(summarize.CAMERA.parent / "replay.npz", allow_pickle=False) as new:
        np.testing.assert_array_equal(old["traces"], new["traces"])
        np.testing.assert_array_equal(old["times"], new["times"])
        np.testing.assert_array_equal(old["images"][:, :, :, 0], new["images"][:, :, :, 0])
        assert np.max(new["geometry_cache_lag"]) > 0


def test_figure_manifest_and_editable_text_when_installed():
    import xml.etree.ElementTree as ET
    path = camera.ROOT.parent / "docs/figures/pc1_layer_action_equivalence_20261107/manifest.json"
    if not path.exists():
        pytest.skip("figures not installed")
    data = json.loads(path.read_text())
    assert len(data["artifacts"]) == 12
    for name, digest in data["artifacts"].items():
        assert file_hash(name) == digest
        if name.endswith(".svg"):
            root = ET.parse(name).getroot()
            assert len(root.findall(".//{http://www.w3.org/2000/svg}text")) > 5
