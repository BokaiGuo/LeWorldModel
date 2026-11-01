import numpy as np

from experiments.spectral_reachability import p1_backtrack_capability_prepare_20261030 as prepare


def test_exact_twenty_one_time_major_probes_and_fixed_scale():
    probes = prepare.probes()
    assert [x["name"] for x in probes] == ["zero"] + [
        f"{sign}{i}" for i in range(10) for sign in ("+", "-")]
    for row in probes:
        logits = np.frombuffer(bytes.fromhex(row["logits_float32_hex"]), dtype="<f4")
        actions = np.frombuffer(bytes.fromhex(row["raw5_float32_hex"]), dtype="<f4")
        assert logits.shape == actions.shape == (10,)
        assert actions.tobytes() == np.tanh(logits).astype("<f4").tobytes()
        assert np.count_nonzero(logits) == (0 if row["index"] == 0 else 1)
        assert np.linalg.norm(logits) <= .25
