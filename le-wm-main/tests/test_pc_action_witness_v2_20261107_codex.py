import numpy as np
import pytest
import torch

from experiments.pc_action_witness_v2_20261107_codex import ActionWitnessV2, describe_bound


def make(name="a", root="r0", clock="5env", actions=None):
    return ActionWitnessV2(name, root, clock, [[0.1, 0.0]] if actions is None else actions)


def test_original_mutation_cannot_change_bound_witness():
    source = [[0.1, 0.0]]
    item = make(actions=source)
    fingerprint = item.fingerprint
    source[0][0] = 0.9
    exported = item.array()
    exported[0, 0] = 0.8
    assert item.fingerprint == fingerprint
    np.testing.assert_allclose(item.array(), [[0.1, 0.0]])


def test_root_clock_and_actual_float32_identity():
    assert make(root="r0").fingerprint != make(root="r1").fingerprint
    assert make(clock="5env").fingerprint != make(clock="1env").fingerprint
    assert make(actions=[[0.1, 0.0]]).fingerprint == make(actions=np.array([[0.1, 0.0]], np.float32)).fingerprint


def test_mixed_context_fails_before_scoring():
    seen = []
    with pytest.raises(ValueError):
        describe_bound(lambda x: seen.append(x), make("hold"), [make("probe", root="r1")])
    assert not seen


def test_bound_descriptor_keeps_signed_effect_and_grad():
    scale = torch.tensor(1.0, requires_grad=True)
    result = describe_bound(lambda a: scale * torch.tensor(float(a[0, 0])),
                            make("hold", actions=[[0., 0.]]), [make("probe")])
    assert result["queries"] == 2
    result["effects"].sum().backward()
    assert scale.grad.item() == pytest.approx(-.1)


@pytest.mark.parametrize("actions", [[], [[float("nan"), 0]], [[1.1, 0]]])
def test_invalid_action_rejected(actions):
    with pytest.raises(ValueError):
        make(actions=actions)
