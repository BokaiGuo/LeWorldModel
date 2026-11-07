import numpy as np
import pytest
import torch

from experiments.pc_action_reference_contract_20261107_codex import certify_position_hold, describe_reference
from experiments.pc_action_witness_v2_20261107_codex import ActionWitnessV2


def w(name, root, x=0):
    return ActionWitnessV2(name, root, "five steps", [[x, 0]] * 5)


def test_zero_reference_is_not_automatically_hold():
    zero, plus = w("zero", "root"), w("plus", "root", .1)
    result = describe_reference(lambda a: torch.tensor(a[:, 0].sum()), zero, [plus])
    assert result["reference_semantics"] == "relative_action_reference"
    assert result["effects"].tolist() == pytest.approx([-.5])
    with pytest.raises(ValueError):
        certify_position_hold(zero, np.array([[0., 0.]] * 5 + [[.01, 0.]]),
                              tolerance=1e-6, source_sha256="a" * 64)


def test_hold_requires_full_trajectory_and_matching_root():
    zero, plus = w("zero", "root"), w("plus", "root", .1)
    cert = certify_position_hold(zero, np.zeros((6, 2)), tolerance=1e-6, source_sha256="a" * 64)
    result = describe_reference(lambda a: torch.tensor(a[:, 0].sum()), zero, [plus], position_hold=cert)
    assert result["reference_semantics"] == "witnessed_position_hold"
    with pytest.raises(ValueError):
        describe_reference(lambda a: torch.tensor(0.), w("zero", "other"), [], position_hold=cert)
    with pytest.raises(ValueError):
        certify_position_hold(zero, np.zeros((5, 2)), tolerance=1e-6, source_sha256="a" * 64)
    out_and_back = np.zeros((6, 2))
    out_and_back[2, 0] = .1
    with pytest.raises(ValueError):
        certify_position_hold(zero, out_and_back, tolerance=1e-6, source_sha256="a" * 64)
