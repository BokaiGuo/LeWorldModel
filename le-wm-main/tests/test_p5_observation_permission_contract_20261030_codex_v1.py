import copy

import pytest

from experiments.spectral_reachability import p5_observation_permission_contract_20261030_codex_v1 as m


@pytest.fixture(scope="module")
def data():
    return (m.read(m.PANEL, m.PANEL_SHA, 8 * 1024 * 1024),
            m.read(m.BRIDGE, m.BRIDGE_SHA, 8 * 1024 * 1024))


def test_actual_contract(data):
    c = m.build(*data)
    assert len(c["rows"]) == 164 and len(c["pairs"]) == 80
    assert c["arms"]["jepa_native_h1"]["observation_channels"] == [
        "rgb_history", "agent_position", "agent_velocity"]
    assert c["arms"]["lewm_existing"]["observation_channels"] == ["rgb_history"]
    assert c["rows"][0]["lewm_exact_history_and_runtime_input_receipt"] is None
    assert c["P5_admission"] is False


@pytest.mark.parametrize("mutation", ["jepa_rgb_only", "unknown_channel", "goal_hidden",
                                      "missing_arm", "extra_arm", "velocity_missing"])
def test_permission_relabeling_rejected(mutation):
    arms = copy.deepcopy(m.ARMS)
    if mutation == "jepa_rgb_only":
        arms["jepa_native_h1"]["observation_channels"] = ["rgb_history"]
    elif mutation == "unknown_channel":
        arms["lewm_existing"]["observation_channels"].append("state")
    elif mutation == "goal_hidden":
        arms["lewm_existing"]["goal_access"] = "none_in_predictor"
    elif mutation == "missing_arm":
        del arms["lewm_existing"]
    elif mutation == "extra_arm":
        arms["invented"] = {}
    else:
        arms["jepa_native_h1"]["observation_channels"].remove("agent_velocity")
    with pytest.raises(ValueError):
        m.validate_arm_declarations(arms)


@pytest.mark.parametrize("claim", ["rgb_only_jepa", "matched_observation_cross_model",
                                  "causal_effect_of_proprio", "cross_model_raw_latent_mse_ranking",
                                  "physical_control_benefit", "p5_scientific_admission", "unknown"])
def test_comparison_rejected(claim):
    with pytest.raises(ValueError):
        m.validate_comparison(m.ARMS, claim)


def test_descriptive_comparison_boundaries():
    result = m.validate_comparison(m.ARMS, "same_physical_panel_descriptive")
    assert result["permission_disclosure_required"] is True
    assert result["same_observation_permissions"] is False


@pytest.mark.parametrize("mutation", ["missing", "order", "rgb", "label", "hash", "pair"])
def test_panel_identity_errors(data, mutation):
    panel, bridge = data
    panel = copy.deepcopy(panel)
    if mutation == "missing":
        panel["rows"].pop()
    elif mutation == "order":
        panel["rows"][0], panel["rows"][1] = panel["rows"][1], panel["rows"][0]
    elif mutation == "rgb":
        panel["rows"][0]["history_rgb"][0]["file_sha256"] = "0" * 64
    elif mutation == "label":
        panel["rows"][0]["successor_rgb_identity_only"]["step"] = 5
    elif mutation == "hash":
        panel["rows"][0]["normalized_proprio_float32_sha256"] = "invalid"
    else:
        panel["pairs"][20]["minus_index"] = 1
    with pytest.raises(ValueError):
        m.build(panel, bridge)


def test_input_hash_and_duplicate(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b'{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="SHA mismatch"):
        m.read(path, "0" * 64, 100)
    with pytest.raises(ValueError, match="duplicate key"):
        m.read(path, m.sha(path.read_bytes()), 100)
