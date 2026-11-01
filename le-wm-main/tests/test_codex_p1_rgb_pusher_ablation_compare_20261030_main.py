import pytest

from experiments.spectral_reachability import codex_p1_rgb_pusher_ablation_compare_20261030_main as m


def fixture():
    rows, shards = [], []
    metric = {k: 1.0 for k in m.FIELDS}
    metric["native_success"] = False
    for i in range(24):
        rows.append({"index": i, "logical_pair": i // 3, "root_slot": 0, "condition": "off",
                     "arm": "a", "role": "selected", "status": "probe_completed",
                     "rgb_decision": "model_packet", "accepted_suffix_steps": 10,
                     "endpoint_metrics": metric.copy()})
        shards.append({"index": i, "report_sha256": "a"})
    trace = [{"step": i+1, "actual_bits": "ab", "state": [], "snapshot": {},
              "terminated": False, "truncated": False} for i in range(5)]
    return {"rows": rows}, {"shards": shards}, trace


def test_pair_keeps_rejection_and_never_fills_censored_endpoint():
    old, report, trace = fixture()
    new, _, _ = fixture()
    new["rows"][0].update(status="rgb_rejected", rgb_decision="rgb_rejected",
                          accepted_suffix_steps=5, endpoint_metrics=None)
    result = m.compare(old, new, report, report, lambda *_: (trace, trace, True))
    assert result["summary"]["comparable"] == 23
    assert result["rows"][0]["new_minus_old"] is None
    assert result["rows"][0]["new_status"] == "rgb_rejected"


def test_pair_rejects_different_midpoint():
    old, report, trace = fixture()
    with pytest.raises(ValueError):
        m.compare(old, old, report, report, lambda *_: (trace, trace, False))
