import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_cross_action_actual_mid_20261030_main as audit


def test_all_saved_vectors_and_source_records_recompute():
    path = audit.base.ROOT/"results/codex_p1_cross_action_actual_mid_20261030_main_RUN01.json"
    value = audit.pair.read(path, "198ea24cbb55d58ce1ba6a883819587999ecb79a5771d37fe6b8f8a92af283ad")
    assert len(value["rows"]) == 24
    for i, row in enumerate(value["rows"]):
        assert row["identity"]["index"] == i
        for key in ("predicted_context_action", "real_context_action"):
            item = row[key]
            a = np.asarray(item["actual_latent"], dtype=np.float64)
            p = np.asarray(item["predicted_latent"], dtype=np.float64)
            assert a.shape == p.shape == (192,)
            assert abs(np.mean((a-p)**2) - item["prediction_error_mse"]) < 1e-6
            assert len(item["packet_sha256"]) == 64


def test_action_requires_actual_transition_match():
    folder = audit.compare.PRED_DIR/"row_00"
    run = audit.pair.read(audit.compare.PRED_DIR/"run_report.json", audit.compare.PRED_REPORT)
    report = audit.pair.read(folder/"physical/report.json", run["shards"][0]["report_sha256"])
    transitions = audit.pair.transitions(folder/"physical", run["shards"][0]["report_sha256"])
    packet = audit.action(folder, transitions, report)
    assert packet.shape == (5, 2)
    assert hashlib.sha256(packet.tobytes()).hexdigest()
    altered = [dict(t) for t in transitions]
    altered[5]["actual_bits"] = "00"*8
    with pytest.raises(ValueError, match="executed packet"):
        audit.action(folder, altered, report)
