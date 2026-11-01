"""Image/array contracts only: never load a model, simulator or hidden state."""
import ast
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p1_goal_orientation_contrast_20261027_v1 as m


def goal(index=0):
    with Image.open(m.GOAL_DIR/f'root_{index}.original_goal.png') as image:
        return np.asarray(image).copy()


@pytest.fixture(scope='module')
def actual_catalog():
    return m.catalog(goal())


def test_angle_table_is_full_circle_uniform_and_symmetric():
    assert m.ANGLES == tuple(range(-180, 180, 30))
    assert len(set(a % 360 for a in m.ANGLES)) == 12
    assert {a % 360 for a in m.ANGLES} == {(-a) % 360 for a in m.ANGLES}
    assert m.RESAMPLERS == ('nearest', 'bilinear')
    assert m.ROI == (48, 48, 176, 176)


@pytest.mark.parametrize('index', range(4))
def test_actual_goal_isolation_preserves_complete_foreground(index):
    original = goal(index); frozen = original.copy()
    clean, patch, info = m.isolate(original)
    np.testing.assert_array_equal(original, frozen)
    assert info['foreground_pixels'] > info['canonical_object_pixels'] == 183
    assert info['rotation_support_radius_bound'] < info['all_angle_clearance']
    assert info['agent_bbox'] == ([20,166,37,182], [66,102,83,119], [120,186,137,203], [30,17,47,34])[index]
    left, top, right, bottom = m.ROI
    np.testing.assert_array_equal(clean[top:bottom, left:right], patch)
    changed = np.any(clean != original, axis=-1)
    x0, y0, x1, y1 = info['agent_bbox']
    changed[y0:y1, x0:x1] = False
    assert not changed.any()


def test_complete_catalog_includes_identity_and_all_angle_sampler_pairs(actual_catalog):
    images = actual_catalog['images']
    assert len(images) == 26
    assert [(x['angle_degrees'], x['sampler']) for x in images[2:]] == [(a, s) for a in m.ANGLES for s in m.RESAMPLERS]
    assert images[0]['id'] == 'original' and images[1]['id'] == 'clean'
    for image in images[2:]:
        if image['angle_degrees'] == 0:
            np.testing.assert_array_equal(image['rgb'], images[1]['rgb'])
            assert image['artifact']['roundtrip_rgb_mse'] == 0


def test_rotation_changes_no_pixels_outside_roi(actual_catalog):
    left, top, right, bottom = m.ROI
    mask = np.ones((224,224), dtype=bool); mask[top:bottom, left:right] = False
    for image in actual_catalog['images'][2:]:
        np.testing.assert_array_equal(image['rgb'][mask], actual_catalog['images'][1]['rgb'][mask])
        assert m.white_perimeter(image['rgb'][top:bottom, left:right])


def test_artifacts_are_measured_not_called_physical_equivalence(actual_catalog):
    assert len(actual_catalog['sampler_differences']) == 12
    assert any(x['rgb_mse'] > 0 for x in actual_catalog['sampler_differences'])
    assert any(x['artifact']['roundtrip_rgb_mse'] > 0 for x in actual_catalog['images'][2:])
    for image in actual_catalog['images'][2:]:
        artifact = image['artifact']
        assert np.isfinite(list(artifact['ink_centroid_shift_pixels'])).all()
        assert artifact['ink_mass_ratio'] > 0
    contract = m.contract()
    assert contract['renderer_faithful_physical_states'] is contract['metric_validity_claim'] is False
    assert contract['new_Q'] is contract['physical_pose_input'] is False
    assert contract['actual_neural_calls'] == contract['actual_physical_steps'] == 0


@pytest.mark.parametrize('kind', ['wrong_dtype', 'wrong_shape', 'no_agent', 'no_object', 'foreign_roi', 'agent_occlusion', 'object_outside_roi'])
def test_refuse_ineligible_images_without_repair(kind):
    image = goal()
    if kind == 'wrong_dtype':
        image = image.astype(np.float32)
    elif kind == 'wrong_shape':
        image = image[:223]
    elif kind == 'no_agent':
        image[np.all(image == m.AGENT_COLOR, axis=-1)] = 255
    elif kind == 'no_object':
        image[np.all(image == m.OBJECT_COLOR, axis=-1)] = 255
    elif kind == 'foreign_roi':
        image[60:62, 60:62] = 100
    elif kind == 'agent_occlusion':
        image[166,20] = 20  # perimeter cannot be invented as white
    else:
        image[40,100] = m.OBJECT_COLOR
    before = image.copy()
    with pytest.raises(ValueError):
        m.catalog(image)
    np.testing.assert_array_equal(image, before)


def test_insufficient_all_angle_clearance_refuses_whole_catalog():
    image = goal(); image[60:95, 100:102] = m.OBJECT_COLOR
    # Extend the connected object towards fixed top boundary, rather than
    # allow a subset of apparently successful angles or a wider adaptive ROI.
    with pytest.raises(ValueError):
        m.catalog(image)


def test_no_outcome_or_pose_api_and_no_runtime_imports():
    import inspect
    assert list(inspect.signature(m.catalog).parameters) == ['goal']
    assert list(inspect.signature(m.isolate).parameters) == ['goal']
    tree = ast.parse(Path(m.__file__).read_text())
    modules = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(x.startswith(('torch', 'gym', 'stable_worldmodel', 'experiments.')) for x in modules)


def test_prepare_complete_source_bound_under_one_mb_and_verify(tmp_path):
    output = tmp_path/'prepared'
    receipt = m.prepare(output)
    assert receipt['bytes'] < 1_000_000 and receipt['images'] == 104
    assert receipt['unique_clean_object_patches'] == 1  # not four independent object poses
    assert m.verify(output, receipt['prepare_sha256'])['files'] == 106
    with pytest.raises(ValueError, match='exclusive'):
        m.prepare(output)
    victim = output/'root_0.angle_-180_nearest.png'
    raw = victim.read_bytes(); victim.write_bytes(raw+b'tamper')
    with pytest.raises(ValueError, match='drift'):
        m.verify(output, receipt['prepare_sha256'])


def test_source_hash_drift_is_rejected(tmp_path, monkeypatch):
    # No edits to actual source/PNG: redirect one expected hash in memory.
    wrong = list(m.GOAL_FILES); wrong[0] = '0'*64
    monkeypatch.setattr(m, 'GOAL_FILES', tuple(wrong))
    with pytest.raises(ValueError, match='goal file drift'):
        m.prepare(tmp_path/'refused')
    assert not (tmp_path/'refused').exists()
