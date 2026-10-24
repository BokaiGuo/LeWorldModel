from experiments.spectral_reachability.summarize_sra_p3_confirmation import summarize


def _report(roots, offset=0.0):
    rows = []
    for root in roots:
        rows.append({"root": root, "horizons": {"2": {"spectral_regret": offset, "immediate_regret": 1.0}}})
    return {
        "schema": "sra_p3_fixed_tree_horizon_expansion.v1",
        "status": "mechanism_ready",
        "inputs": {"roots": roots, "horizons": [2], "model_batch_size": 64, "checkpoint_sha256": "x", "physical_position_backend": {"contract": "x"}},
        "gates": {"cuda_only": True},
        "rows": rows,
    }


def test_p3_confirmation_requires_disjoint_panels_and_reports_paired_mean():
    result = summarize(_report([1, 2]), _report([3, 4], offset=0.25))
    assert result["status"] == "pass"
    assert result["horizons"]["2"]["pooled_root_mean"] == -0.875


def test_p3_confirmation_rejects_root_overlap():
    result = summarize(_report([1, 2]), _report([2, 3]))
    assert result["status"] == "invalid"
    assert not result["gates"]["root_disjoint_panels"]
