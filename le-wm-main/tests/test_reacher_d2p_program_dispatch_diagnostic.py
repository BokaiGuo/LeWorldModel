from copy import deepcopy

import h5py
import numpy as np
import pytest

from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import planned_source_roots, root_execution_gates


def test_source_plan_is_metadata_only_deterministic_and_collision_checked(tmp_path):
    dataset = tmp_path / "source.h5"
    with h5py.File(dataset, "w") as handle:
        handle["ep_len"] = np.full(10000, 50)
        handle["ep_offset"] = np.arange(10000) * 50
    first = planned_source_roots(dataset, [202626000], set())
    assert first == planned_source_roots(dataset, [202626000], set())
    assert first[0]["source_episode"] * 50 <= first[0]["source_start_row"] <= first[0]["source_episode"] * 50 + 30
    with pytest.raises(ValueError, match="collision"):
        planned_source_roots(dataset, [202626000], {first[0]["source_episode"]})
    with pytest.raises(ValueError, match="unique"):
        planned_source_roots(dataset, [1, 1], set())


def _methods():
    branch = {"first_receipt": {"selected_proposal_index": 1}, "action_witness_raw": [[1., 2.]],
        "physical": {"final_qpos_distance": .5},
        "execution_identity": {"initial": {"reset_rgb_sha256": "r", "goal_rgb_sha256": "g"}}}
    methods = {key: deepcopy(branch) for key in ("dispatch", "same_bank_greedy", "zero", "lookahead_cem")}
    methods["dispatch"]["first_receipt"] = {"dispatch": {"reconfiguration_witness_found": False,
        "same_bank_greedy_index": 1, "physical_labels_used": False}}
    return methods


def test_fallback_requires_exact_physical_replay():
    methods = _methods()
    assert all(root_execution_gates(methods).values())
    methods["dispatch"]["action_witness_raw"][0][0] = 0.
    assert not root_execution_gates(methods)["fallback_executes_exact_greedy_actions"]
    methods["dispatch"]["physical"]["final_qpos_distance"] += 1e-14
    assert not root_execution_gates(methods)["fallback_replays_exact_greedy_final_distance"]


def test_reconfiguration_need_not_equal_greedy_but_must_share_root():
    methods = _methods()
    methods["dispatch"]["first_receipt"]["dispatch"]["reconfiguration_witness_found"] = True
    methods["dispatch"]["action_witness_raw"] = [[9., 1.]]
    methods["dispatch"]["physical"]["final_qpos_distance"] = .1
    assert all(root_execution_gates(methods).values())
    methods["zero"]["execution_identity"]["initial"]["goal_rgb_sha256"] = "wrong"
    assert not root_execution_gates(methods)["same_root_goal_for_all_methods"]
