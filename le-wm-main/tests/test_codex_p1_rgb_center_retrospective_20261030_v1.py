import inspect

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_rgb_center_retrospective_20261030_v1 as audit


def test_image_estimator_has_no_state_input_and_rejects_ambiguity():
    assert list(inspect.signature(audit.image_only_estimate).parameters) == ["image"]
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    decision, center = audit.image_only_estimate(image)
    assert decision["decision"] == "rejected"
    assert center is None
    with pytest.raises(TypeError):
        audit.image_only_estimate(image, np.zeros(7))


def test_posthoc_scale_and_invalid_values():
    assert audit.posthoc_error([112, 56], [256, 128, 0, 0, 0, 0, 0]) == 0.
    assert audit.posthoc_error(None, [0]*7) is None
    with pytest.raises(ValueError):
        audit.posthoc_error([1, 2], [float("nan")]*7)
