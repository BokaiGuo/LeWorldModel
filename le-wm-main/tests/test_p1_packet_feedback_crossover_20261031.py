from experiments.spectral_reachability import p1_packet_feedback_crossover_20261031 as m
from experiments.spectral_reachability import p1_packet_feedback_crossover_consume_20261031 as c
import pytest


def test_crossover_full_roster_and_all_feedback_actions():
    fixed = m.build()
    source = m.feedback.build()
    assert len(fixed["anchors"]) == 24
    assert fixed["fixed_branches"] == 96
    for pair, bank in enumerate(fixed["banks"]):
        assert len(bank) == 4
        assert bank[0]["raw10_float32_hex"] == bytes(80).hex()
        assert [entry["name"] for entry in bank[1:]] == [
            "selected_feedback_tail", "zero_feedback_tail", "greedy_feedback_tail"]
        for role, entry in enumerate(bank[1:]):
            assert entry["raw10_float32_hex"] == source["roster"][3 * pair + role]["raw10_float32_hex"]


def test_out_of_domain_return_is_not_an_accepted_response():
    steps = [{"step": i} for i in range(1, 8)]
    c.validate_responses({"status": "censored_out_of_domain", "returned_suffix": 8}, steps)
    with pytest.raises(ValueError):
        c.validate_responses({"status": "probe_completed", "returned_suffix": 8}, steps)
    with pytest.raises(ValueError):
        c.validate_responses({"status": "censored_out_of_domain", "returned_suffix": 7}, steps)
