import numpy as np
import pytest
from experiments.spectral_reachability.pusht_rgb_observability_guard import RgbObservabilityGuard, ObservationUnavailable


def image(visible):
    rgb = np.full((224, 224, 3), 255, np.uint8)
    if visible: rgb[20, 20] = [65, 105, 225]
    return rgb


def test_each_command_requires_fresh_observation():
    g = RgbObservabilityGuard(root_id='r', agent_rgb=(65, 105, 225))
    with pytest.raises(ObservationUnavailable): g.dispatch([0., 0.])
    g.observe(image(True), primitive_step=0)
    np.testing.assert_array_equal(g.dispatch([.1, 0.]), [.1, 0.])
    with pytest.raises(ObservationUnavailable): g.dispatch([0., 0.])
    g.observe(image(True), primitive_step=1)
    assert g.receipt()['primitive_commands_authorized'] == 1


def test_absent_agent_invalidates_continuation_without_recovery():
    g = RgbObservabilityGuard(root_id='r', agent_rgb=(65, 105, 225))
    with pytest.raises(ObservationUnavailable): g.observe(image(False), primitive_step=0)
    with pytest.raises(ObservationUnavailable): g.dispatch([0., 0.])
    with pytest.raises(ObservationUnavailable): g.observe(image(True), primitive_step=0)
    assert g.receipt()['abstained']


def test_no_silent_change_to_noncanonical_renderer():
    with pytest.raises(ValueError): RgbObservabilityGuard(root_id='r', agent_rgb=(0, 0, 255))


@pytest.mark.parametrize('step', [False, True, 0., -1])
def test_explicit_integer_primitive_index_required(step):
    g = RgbObservabilityGuard(root_id='r', agent_rgb=(65, 105, 225))
    with pytest.raises(ObservationUnavailable): g.observe(image(True), primitive_step=step)
