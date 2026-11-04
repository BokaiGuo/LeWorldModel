import json

import numpy as np

from experiments import pc_geometry_native_suffix_20261103_codex as source
from experiments import pc_geometry_rgb_action_manifest_20261103_codex as manifest
from experiments import pc_geometry_rgb_fixed_bank_physical_20261103_codex as physical
from experiments import pc_geometry_rgb_fixed_bank_select_20261103_codex as decision
from experiments import pc_native_velocity_intervention_20261103_codex as velocity
from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base


def test_velocity_pairing_and_arithmetic():
    saved = json.loads(velocity.OUTPUT.read_text())
    original = json.loads(source.OUTPUT.read_text())
    assert saved["source_sha256"] == base.digest(source.OUTPUT)
    assert saved["worker_sha256"] == base.digest(velocity.WORKER)
    assert len(saved["rows"]) == 48
    errors = []
    for row, prior in zip(saved["rows"], original["rows"], strict=True):
        assert (row["root_index"], row["prefix_name"]) == (prior["root_index"], prior["prefix_name"])
        for name, condition in row["conditions"].items():
            assert len(condition["traces"]) == 6
            for trace, distance in zip(condition["traces"], condition["distances"], strict=True):
                assert len(trace) == 11
                np.testing.assert_allclose(np.linalg.norm(np.asarray(trace[-1][:2]) - base.GOAL), distance)
                np.testing.assert_allclose(trace[0][:2], row["prefix_state"][:2])
                if name == "zero_velocity":
                    np.testing.assert_array_equal(trace[0][2:], [0, 0])
            np.testing.assert_allclose(condition["passive_change"], condition["distances"][0] - row["prefix_distance"])
            np.testing.assert_allclose(condition["active_changes"],
                                       np.asarray(condition["distances"]) - condition["distances"][0])
        delta = np.asarray(row["conditions"]["recorded"]["distances"]) - prior["common_distances"]
        np.testing.assert_allclose(row["reconstruction_difference"], delta, atol=1e-15)
        errors.extend(abs(delta))
    np.testing.assert_allclose(max(errors), saved["max_reconstruction_distance_error"], atol=1e-15)
    assert saved["reconstruction_pass"]


def test_decision_and_post_seal_physical_contract():
    actions = json.loads(manifest.OUTPUT.read_text())
    selected = json.loads(decision.OUTPUT.read_text())
    evaluated = json.loads(physical.OUTPUT.read_text())
    assert selected["action_manifest_sha256"] == base.digest(manifest.OUTPUT)
    assert selected["arrays_sha256"] == base.digest(decision.ARRAYS)
    assert selected["model_before"] == selected["model_after"]
    assert evaluated["decision_sha256"] == base.digest(decision.OUTPUT)
    assert len(selected["rows"]) == 4 and len(evaluated["rows"]) == 48
    for root, row in enumerate(selected["rows"]):
        assert len(row["candidates"]) == 12
        scores = [c["terminal_score"] for c in row["candidates"]]
        assert row["selected_index"] == int(np.argmin(scores))
        index = row["selected_index"]
        assert row["selected_is_qualified"] == (
            row["candidates"][index]["prefix_score"] >= row["initial_score"]
            and scores[index] < min(scores[:2]))
        assert row["abstained_reconfiguration"] != row["selected_is_qualified"]
        for i, candidate in enumerate(row["candidates"]):
            action = actions["rows"][root*12+i]
            observed = evaluated["rows"][root*12+i]
            assert candidate["name"] == action["prefix_name"] == observed["name"]
            np.testing.assert_allclose(candidate["prefix_raw"], action["prefix_actions"], atol=1e-7)
            assert len(candidate["evaluations"]) == 100
            vals = [e["score"] for e in candidate["evaluations"]]
            assert candidate["best_index"] == int(np.argmin(vals))
            assert candidate["terminal_score"] == min(vals)
            assert observed["selected"] == (i == index)
            assert len(observed["trace"]) == 17
            assert observed["warmup_state_error"] == 0
            end = np.asarray(observed["trace"][-1]["state"][:2])
            np.testing.assert_allclose(np.linalg.norm(end - base.GOAL), observed["terminal_distance"])
