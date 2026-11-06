import torch

from experiments.pc_prefix_candidate_set_audit_20261105_codex import candidate_set_difference


def test_unordered_set_and_tolerance():
    left = torch.tensor([[[[1.0]], [[2.0]]]])
    right = torch.tensor([[[[2.0]], [[1.0 + 1e-7]]]])
    a, b = candidate_set_difference(left, right)
    assert not a.any() and not b.any()


def test_new_candidate_is_reported_on_both_sides():
    left = torch.tensor([[[[1.0]], [[2.0]]]])
    right = torch.tensor([[[[1.0]], [[3.0]]]])
    a, b = candidate_set_difference(left, right)
    assert a.tolist() == [[False, True]]
    assert b.tolist() == [[False, True]]


def test_second_sign_flip_reorders_symmetric_combinations():
    d1 = torch.tensor([1.0, 0.0])
    d2 = torch.tensor([0.0, 1.0])
    left = torch.stack((d1 + d2, d1 - d2))[None, :, None, :]
    right = torch.stack((d1 - d2, d1 + d2))[None, :, None, :]
    a, b = candidate_set_difference(left, right)
    assert not a.any() and not b.any()
