import copy

import pytest

from experiments.spectral_reachability import p1_uphill_physical_analysis_20261030 as analysis


def test_pairing_and_censoring():
    rows = []
    for i in range(8):
        for role in ("incumbent", "model_selected"):
            rows.append({"role": role, "root_slot": i // 4, "condition": "off",
                         "arm": str(i), "task_id": str(i), "goal": [0.] * 7,
                         "status": "horizon_completed", "own_returned": 15,
                         "checkpoints": [{"step": step, "state": [float(step)] * 7}
                                         for step in (0, 5, 15)],
                         "missing_checkpoints": [], "plan": {"raw15_sha256": role}})
    report = {"rows": rows}
    receipt = {"status": "saved_checks_passed_not_admitted"}
    result = analysis.analyze(report, receipt)
    assert len(result["pairs"]) == 8
    censored = copy.deepcopy(report)
    censored["rows"][1].update(own_returned=5, status="native_stopped",
                                checkpoints=censored["rows"][1]["checkpoints"][:2],
                                missing_checkpoints=[10, 15])
    pair = analysis.analyze(censored, receipt)["pairs"][0]
    assert pair["latent_rerank"]["endpoint"] is None
    assert pair["rerank_minus_original_endpoint"] is None
    censored["rows"][1]["own_returned"] = 15
    with pytest.raises(ValueError, match="endpoint censorship"):
        analysis.analyze(censored, receipt)
