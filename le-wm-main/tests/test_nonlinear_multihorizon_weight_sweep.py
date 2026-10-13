import json

from experiments.empirical_spectral_audit.run_nonlinear_multihorizon_weight_sweep import PROFILES


def test_weight_profiles_cover_one_step_short_and_long_regimes():
    assert PROFILES["one_step"] == {1: 1.0, 2: 0.0, 4: 0.0}
    assert PROFILES["short_favoring"][1] > PROFILES["short_favoring"][4]
    assert PROFILES["long_favoring"][4] > PROFILES["long_favoring"][1]
    assert all(value >= 0.0 for profile in PROFILES.values() for value in profile.values())
