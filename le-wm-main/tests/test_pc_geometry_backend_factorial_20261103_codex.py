import json

import numpy as np

from experiments import pc_geometry_backend_factorial_20261103_codex as factorial
from experiments import pc_geometry_backend_replay_20261103_codex as audit
from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base


def test_stepwise_panel_and_original_replay():
    receipt = json.loads(audit.OUTPUT.read_text())
    source = json.loads(base.OUTPUT.read_text())
    assert receipt["source_sha256"] == base.digest(base.OUTPUT)
    assert receipt["protocol_sha256"] == base.digest(audit.PROTOCOL)
    assert receipt["worker_sha256"] == base.digest(audit.WORKER)
    assert len(receipt["rows"]) == len(source["rows"]) == 8
    for old, row in zip(source["rows"], receipt["rows"], strict=True):
        assert (old["start"], old["policy"]) == (row["start"], row["policy"])
        assert len(row["native"]) == len(row["modern"]) == len(old["trace"]) + 1
        for index, (a, b) in enumerate(zip(row["native"], row["modern"], strict=True)):
            delta = np.asarray(a["state"]) - np.asarray(b["state"])
            np.testing.assert_allclose(row["position_l2"][index], np.linalg.norm(delta[:2]), atol=1e-15)
            np.testing.assert_allclose(row["velocity_l2"][index], np.linalg.norm(delta[2:]), atol=1e-15)
            if index:
                np.testing.assert_array_equal(b["state"], old["trace"][index-1]["position"] +
                                              old["trace"][index-1]["velocity"])
    assert receipt["fixed_panel_consistent"] is False


def test_factorial_matches_stepwise_and_declared_pairs():
    receipt = json.loads(factorial.OUTPUT.read_text())
    earlier = json.loads(audit.OUTPUT.read_text())
    assert receipt["source_sha256"] == base.digest(base.OUTPUT)
    assert receipt["worker_sha256"] == base.digest(audit.WORKER)
    assert receipt["protocol_sha256"] == base.digest(factorial.PROTOCOL)
    assert receipt["script_sha256"] == base.digest(factorial.Path(factorial.__file__))
    assert len(receipt["runs"]) == 4
    for pair in receipt["comparisons"]:
        left, right = receipt["runs"][pair["left"]], receipt["runs"][pair["right"]]
        pos, vel, contacts = [], [], 0
        for a, b in zip(left["rows"], right["rows"], strict=True):
            for x, y in zip(a["trace"], b["trace"], strict=True):
                delta = np.asarray(x["state"]) - np.asarray(y["state"])
                pos.append(np.linalg.norm(delta[:2]))
                vel.append(np.linalg.norm(delta[2:]))
                contacts += audit.wall_contact(x["contacts"]) != audit.wall_contact(y["contacts"])
        np.testing.assert_allclose(pair["max_position_l2"], max(pos), atol=1e-15)
        np.testing.assert_allclose(pair["max_velocity_l2"], max(vel), atol=1e-15)
        assert pair["wall_contact_disagreements"] == contacts
    cross = receipt["comparisons"][2]
    assert cross["max_position_l2"] == earlier["max_position_l2"]
    assert cross["max_velocity_l2"] == earlier["max_velocity_l2"]
