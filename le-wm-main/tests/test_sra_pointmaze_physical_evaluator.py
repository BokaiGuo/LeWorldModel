import inspect

from experiments.spectral_reachability import pointmaze_physical_evaluator as evaluator
from experiments.spectral_reachability import run_sra_p1_pointmaze_cuda as p1
from experiments.spectral_reachability import run_sra_p2_action_order_cuda as p2


def test_replay_uses_recorded_high_level_transition_contract():
    source = inspect.getsource(evaluator.replay_recorded_actions)
    assert "VELOCITY_LIMIT" in source
    assert "FRAME_SKIP" in source
    assert "mujoco.mj_step" in source


def test_sra_physical_labelers_share_the_contract_evaluator():
    assert "replay_position_endpoints" in inspect.getsource(p1.physical_tree_labels)
    assert "replay_position_endpoints" in inspect.getsource(p2.physical_endpoint)


def test_p1_can_score_the_exact_fixed_continuation_bank_without_outcome_labels():
    source = inspect.getsource(p1)
    assert 'choices=("analytic_dls", "discrete_reachable_set")' in source
    assert "discrete_reachable_set_cost(model, chart, goal_latent, continuation_blocks)" in source
    assert "physical_two_block_oracle_distance" not in inspect.getsource(p1.discrete_reachable_set_cost)


def test_position_contract_fails_closed_on_endpoint_script_drift():
    source = inspect.getsource(evaluator.position_backend_receipt)
    assert "endpoint_backend_script_sha256" in source
    assert "does not bind the current endpoint evaluator scripts" in source


def test_endpoint_evaluator_returns_a_request_bound_receipt():
    from pathlib import Path

    endpoint = Path(evaluator.__file__).with_name("evaluate_sra_pointmaze_candidate_endpoints.py")
    source = endpoint.read_text(encoding="utf-8")
    assert "request_sha256" in source
    assert "replay_backend_script_sha256" in source
    assert "endpoint response receipt does not match" in inspect.getsource(evaluator.replay_position_endpoints)


def test_legacy_xml_preserves_the_vendored_ground_contact_mask():
    from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _legacy_pointmaze_xml

    assert 'name="ground" type="plane" size="40 40 .25" pos="0 0 -.1" contype="1" conaffinity="0"' in _legacy_pointmaze_xml()
