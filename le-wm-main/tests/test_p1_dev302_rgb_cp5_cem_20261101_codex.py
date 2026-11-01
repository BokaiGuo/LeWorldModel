import numpy as np

from experiments.spectral_reachability import p1_dev302_rgb_cp5_cem_20261101_codex as subject


def test_contact_projection_filters_motion_away_from_object():
    bank = np.zeros((3, 5, 2), np.float32)
    direction = np.asarray([-1.0, 0.0], np.float32)
    bank[0, :2, 0] = -0.1
    bank[1, :2, 0] = 0.1
    score = subject.contact_projection(bank, direction)
    assert score[0] > subject.MIN_CONTACT_PROJECTION
    assert score[1] < 0
    assert score[2] == 0


def test_contact_seed_is_bounded_and_feasible():
    packet = subject.contact_seed([-0.9, -0.4])
    assert packet.shape == (5, 2)
    assert packet.dtype == np.float32
    assert np.max(np.abs(packet)) <= 1
    projection = subject.contact_projection(packet[None], [-0.9, -0.4])
    assert projection[0] >= subject.MIN_CONTACT_PROJECTION


def test_protocol_binds_actual_cp5_and_keeps_gate_closed():
    value = subject.protocol()
    assert value["actual_prefix_sha256"]
    assert value["seed_labels"][-1] == "rgb_contact_tail"
    assert value["algorithm"]["feasibility_role"].startswith("reject obvious")
    assert value["algorithm"]["final_objective"].startswith("robust nonlinear")
    assert value["development_only"] is True
    assert value["fair_total_cost"] is False
    assert value["independent_confirmation"] is False
    assert value["P1_admission"] is False
