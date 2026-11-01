import numpy as np

from experiments.spectral_reachability import (
    p1_dev302_v5_contact_tail_replan_20261101_codex as subject,
)


def test_protocol_has_per_arm_contact_seed_and_equal_budget():
    value = subject.protocol()
    assert [row["role"] for row in value["roles"]] == list(subject.ROLES)
    for row in value["roles"]:
        seed = np.asarray(row["contact_seed_tail"], np.float32)
        assert seed.shape == (2, 5, 2)
        assert subject._projection(
            seed[None], row["rgb_contact_geometry"]["contact_direction_raw_xy"]
        )[0] >= 0.15
    assert value["algorithm"]["minimum_contact_projection"] == 0.15
    assert value["post_cp5_physical_outcomes_used"] is False
    assert value["P1_admission"] is False
