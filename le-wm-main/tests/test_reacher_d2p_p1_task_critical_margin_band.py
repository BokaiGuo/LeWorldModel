from experiments.spectral_reachability.select_reacher_d2p_p1_task_critical_margin_band import select


def _report(rows):
    return {"schema": "d2p_reacher_p1_reconfiguration.v1", "status": "calibration_only", "protocol": {"mechanism_observability": "qpos_distance"}, "rows": rows}


def _row(seed, margin, delta):
    return {"seed": seed, "methods": {"reconfiguration": {"first_receipt": {"non_greedy_witness_found": True, "non_greedy_model_margin": margin}, "physical": {"after_zero_prefix_qpos_distance": 1.0, "after_first_chunk_qpos_distance": 1.0 + delta}}}}


def test_task_critical_selector_keeps_largest_physically_safe_margin_interval():
    result = select(_report([_row(1, .1, .1), _row(2, .2, .2), _row(3, .3, -.1), _row(4, .4, .2)]), minimum_passing_roots=2)
    assert result["selected"]["calibration_admitted_root_seeds"] == [1, 2]
    assert result["protocol_binding"]["mechanism_observability"] == "qpos_distance"
