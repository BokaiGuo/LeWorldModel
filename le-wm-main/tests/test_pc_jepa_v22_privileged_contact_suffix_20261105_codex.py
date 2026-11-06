import numpy as np

from experiments import pc_jepa_v22_privileged_contact_suffix_20261105_codex as contact


def test_shared_suffixes_are_deterministic_and_separated(tmp_path, monkeypatch):
    witness = np.linspace(-0.8, 0.8, 30, dtype=np.float32).reshape(15, 2)
    path = tmp_path / "witness.npy"
    np.save(path, witness)
    monkeypatch.setattr(contact, "WITNESS", path)
    a, b = np.full((15, 2), 0.25, np.float32), np.full((15, 2), -0.25, np.float32)
    prefixes, suffixes, bank = contact.make_bank(a, b)
    assert bank.shape == (64, 15, 2)
    assert np.array_equal(bank, contact.make_bank(a, b)[2])
    assert np.array_equal(prefixes[3], witness[:5])
    assert np.array_equal(suffixes[4], witness[5:])
    assert not ({x.tobytes() for x in suffixes[1:8]} & {x.tobytes() for x in suffixes[8:]})
    for i in range(4):
        assert np.array_equal(bank[i*16:(i+1)*16, 5:], suffixes)
    assert np.max(np.abs(bank)) <= 1
