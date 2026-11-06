import numpy as np
import pytest

from experiments.pc_v25_multidepth_response_posthoc_v1 import audit_layers


def fixture():
    nominal = {"nominal": 0, "mode": None, "scale": 0.0}
    direction = {"nominal": 0, "mode": 0, "scale": 0.5}
    layer = {"depth": 0, "responses": [{"parent": 0, "nominal": 0,
             "matrix": [[1.0], [0.0], [1.0]]}], "retained": [0],
             "candidates": [
                 {"path": [nominal], "chain": [[0.0]], "feature": [0.0, 0.0, 0.0]},
                 {"path": [direction], "chain": [[0.5]], "feature": [0.5, 0.0, 0.5]}]}
    actions = np.array([[[0]], [[1]], [[-1]]], dtype=np.float32)
    features = np.array([[0, 0], [1, 0], [-1, 0]], dtype=float)
    scores = np.array([0, 1, -1], dtype=float)
    return [layer], [(actions, features, scores)]


def test_first_layer_empty_parent_and_exact_response():
    layers, calls = fixture()
    output = audit_layers(layers, calls, 1, 1, 1.0)
    record = output[0]["groups"][0]["candidates"][0]
    assert record["position_absolute_error"] == 0
    assert record["score_absolute_error"] == 0


def test_unmatched_response_is_rejected():
    layers, calls = fixture()
    layers[0]["responses"][0]["matrix"][0][0] = 2.0
    with pytest.raises(ValueError, match="no matching"):
        audit_layers(layers, calls, 1, 1, 1.0)
