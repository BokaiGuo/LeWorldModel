import pytest

from experiments.spectral_reachability.run_reacher_pc_common_tangent_cuda import verify_preflight


def test_preflight_binding_rejects_any_changed_identity():
    keys = ("schema", "roots", "q_report_sha256", "root_source_sha256", "bank_report_sha256",
            "weights_sha256", "action_stats_sha256", "model_parameter_hash_expected",
            "proposal_code_sha256", "preflight_code_sha256")
    valid = {key: key for key in keys}
    verify_preflight(valid, dict(valid))
    for key in keys:
        altered = dict(valid)
        altered[key] = "wrong"
        with pytest.raises(ValueError, match="preflight mismatch"):
            verify_preflight(valid, altered)
