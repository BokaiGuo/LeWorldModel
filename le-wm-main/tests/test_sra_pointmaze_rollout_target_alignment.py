import inspect

from experiments.spectral_reachability import audit_sra_pointmaze_rollout_target_alignment_cuda as audit


def test_target_alignment_audit_is_cuda_only_and_immutable():
    source = inspect.getsource(audit)
    assert "require_cuda()" in source
    assert "immutable audit output exists" in source
    assert "do not interpret P1/P2 effects" in source


def test_target_alignment_binds_packet_width_to_frameskip_and_targets():
    source = inspect.getsource(audit)
    assert "actions_per_packet != configured_frameskip" in source
    assert "five_raw_actions_per_model_packet" in source
    assert "one_packet_target_offset" in source
    assert "two_packet_target_offset" in source
    assert "start_frame + horizon * actions_per_packet" in source
    assert "--max-packets" in source
    assert "packet_horizons" in source


def test_target_alignment_checks_direct_and_composed_recorded_rollouts():
    source = inspect.getsource(audit)
    assert "predicted_chart_context" in source
    assert "two_packet_direct_vs_sequential_max_abs_error" in source
    assert "all_horizons_direct_vs_sequential_max_abs_error" in source
    assert "_neighbor_metrics" in source
    assert "prepare_archive_observation_for_model" in source
