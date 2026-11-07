import numpy as np
import pytest

from experiments import pc_all_split_support_20261108_codex as study


def nominals():
    return np.random.default_rng(71).uniform(-1, 1, (16, 15, 2))


def test_all_cut_diagonals_preserve_nominal_actions():
    actions = nominals()
    bank = study.prior.core.PairedBank(actions)
    for cut in range(16):
        np.testing.assert_array_equal(study.compose(actions, cut, bank.lineage), actions)


def test_k5_exactly_reproduces_previous_bank():
    actions = nominals()
    bank = study.prior.core.PairedBank(actions)
    np.testing.assert_array_equal(study.compose(actions, 5, study.PAIRS), bank.compose(study.PAIRS))


def test_endpoint_cuts_and_same_duration_hold():
    actions = nominals()
    pairs = np.asarray([[3, 0], [8, 6]])
    at_zero = study.compose(actions, 0, pairs)
    np.testing.assert_array_equal(at_zero[0], np.zeros((15, 2)))
    np.testing.assert_array_equal(at_zero[1], actions[5])
    np.testing.assert_array_equal(study.compose(actions, 15, pairs), actions[pairs[:, 0]])
    for cut in range(16):
        result = study.compose(actions, cut, pairs)
        np.testing.assert_array_equal(result[0, :cut], actions[3, :cut])
        np.testing.assert_array_equal(result[0, cut:], np.zeros((15-cut, 2)))


@pytest.mark.parametrize("cut", [-1, 16, True, 5., np.nan])
def test_invalid_cuts_rejected(cut):
    with pytest.raises(ValueError, match="integer cut"):
        study.compose(nominals(), cut, [[0, 1]])


def test_fractional_identity_and_illegal_actions_rejected():
    with pytest.raises(ValueError, match="integer action identities"):
        study.compose(nominals(), 5, [[0., 1.5]])
    bad = nominals()
    bad[0, 0, 0] = 1.01
    with pytest.raises(ValueError, match="illegal nominal"):
        study.compose(bad, 5, [[0, 1]])


def test_array_publication_is_immutable_and_recoverable(tmp_path):
    path = tmp_path / "arrays.npz"
    a = np.arange(12).reshape(3, 4)
    study.save_arrays(path, a=a)
    original = path.read_bytes()
    study.save_arrays(path, a=a)
    assert path.read_bytes() == original
    with pytest.raises(AssertionError):
        study.save_arrays(path, a=a+1)
    assert path.read_bytes() == original
