import copy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_boundary_feedback_prereg_20261102_exclusive as p


def inputs():
    bindings = {k: "a" * 64 for k in (
        "source", "task", "model", "environment", "bank", "selector", "planner",
        "goal_rgb", "initial_rgb", "action_history")}
    prefixes = [np.zeros((5, 2), dtype=np.float32) for _ in range(17)]
    return dict(bindings=bindings, prefixes=prefixes, selected_index=4,
                common_tail=np.zeros((10, 2), dtype=np.float32),
                d2p_plan=np.zeros((10, 2), dtype=np.float32))


def image():
    value = np.zeros((224, 224, 3), np.uint8)
    value[20:24, 20:24] = p.rgb.BLUE
    value[40:44, 40:44] = p.rgb.BLOCK
    return value


def test_eight_fixed_branches_duplicate_roles_and_budget():
    c = p.contract(**inputs())
    assert [(x["role"], x["tail_mode"]) for x in c["branches"]] == [
        (r, t) for r in p.ROLES for t in p.TAILS]
    assert c["role_indices"]["selected"] == c["role_indices"]["greedy"] == 4
    assert c["role_indices"]["task_greedy_direct"] == 4
    assert c["greedy_definition"].startswith("task_greedy_direct:")
    assert c["latent_greedy"] == "diagnostic only; not a physical arm"
    assert c["status"] == "draft_not_authenticated_not_authorized"
    assert "input_source_and_plan_hash_authentication" in c["freeze_blockers"]
    assert c["duplicate_roles_retained"]
    assert c["own_tail_cap_per_branch"]["predictor_samples"] == 3200
    assert c["cost"]["maximum_control_steps"] == 120
    assert c["P1_admission"] is c["execution_authorized"] is False


def test_greedy_cannot_be_replaced_by_latent_greedy_index():
    values = inputs()
    values["greedy_index"] = 2
    with pytest.raises(TypeError):
        p.contract(**values)


def test_rgb_crop_missing_and_exact_margin():
    good = image()
    assert p.screen_rgb(good)["markers"]["pusher"]["margin_pixels"] == 20
    good[20:24, 20:24] = 0
    good[8:12, 8:12] = p.rgb.BLUE
    assert p.screen_rgb(good)["markers"]["pusher"]["margin_pixels"] == 8
    good[8:12, 8:12] = 0
    good[7:11, 7:11] = p.rgb.BLUE
    with pytest.raises(ValueError, match="crop margin"):
        p.screen_rgb(good)
    good[:] = 0
    with pytest.raises(ValueError):
        p.screen_rgb(good)


def test_bounded_actions_and_common_tail_binding():
    plan = np.ones((10, 2), dtype=np.float32)
    tail = np.concatenate([plan[5:], np.zeros((5, 2), np.float32)])
    assert p.validate_common_tail(tail, plan) == p.action_sha(tail, 10)
    tail[-1, 0] = 1
    with pytest.raises(ValueError, match="D2P tail"):
        p.validate_common_tail(tail, plan)
    with pytest.raises(ValueError):
        p.action_sha(np.zeros((5, 2)), 5)


def test_source_and_action_contract_rejects_tamper():
    x = inputs()
    x["bindings"]["task"] = "bad"
    with pytest.raises(ValueError, match="SHA256"):
        p.contract(**x)
    x = inputs()
    x["prefixes"][0][0, 0] = 0.1
    with pytest.raises(ValueError, match="zero candidate"):
        p.contract(**x)


def test_exclusive_protocol_write(tmp_path):
    path = tmp_path / (p.NAME + "_PREP_TEST.json")
    value = p.contract(**inputs())
    assert len(p.write_new(path, value)) == 64
    with pytest.raises(ValueError, match="exclusive"):
        p.write_new(path, copy.deepcopy(value))
