import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_render_replay import array_fingerprint, numeric_fields, differing_paths


def test_fingerprints_include_shape_dtype_and_all_derived_coordinates():
    a = np.arange(9, dtype=np.float64).reshape(3, 3)
    f = array_fingerprint(a)
    assert f == array_fingerprint(a.copy())
    assert f["sha256"] == array_fingerprint(a.ravel())["sha256"]
    assert f["shape"] != array_fingerprint(a.ravel())["shape"]
    changed = a.copy()
    changed[2, 2] = np.nextafter(changed[2, 2], np.inf)
    assert f != array_fingerprint(changed)
    assert f != array_fingerprint(a.astype(np.float32))


def test_scene_fields_and_nested_difference_paths_are_bit_sensitive():
    class Scene:
        scalar = 1.
        count = 2
        vector = np.array([0., 1.], dtype=np.float32)
        _private = 3
        def ignored(self): pass
    fields = numeric_fields(Scene())
    assert set(fields) == {"scalar", "count", "vector"}
    assert fields["scalar"] == 1.0.hex()
    left = {"geoms": [{"pos": fields["vector"]}]}
    right = {"geoms": [{"pos": {**fields["vector"], "sha256": "changed"}}]}
    assert differing_paths(left, right) == ["/geoms/0/pos/sha256"]
