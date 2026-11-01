import json

from experiments.spectral_reachability import p1_joint_anchor_table_20261030 as original
from experiments.spectral_reachability import p1_uphill_cp5_analysis_20261030 as analysis


def test_full_roster_and_censored_comparison():
    values = {key: json.loads((original.ROOT / path).read_bytes())
              for key, (path, _) in original.FILES.items()}
    rows = values["physical_consumed"]["rows"]
    rows[0]["completed_endpoint"] = None
    rows[0]["status"] = "native_stopped"
    events = []
    for i in range(24):
        events.append({"index": i, "result": {"ledger_receipt": {
            "encoding": {"attempted": {}},
            "planning": {"charged_upper_bounds": {}, "completed_events": 0,
                         "actual_usage_verified": False, "coverage_verified": False},
        }}})
    result = analysis.analyze(values["h10_prepare"], values["physical_prepare"],
                              values["physical_consumed"], events)
    assert len(result["rows"]) == 24
    assert result["rows"][0]["independent_suffix_20step"]["endpoint"] is None
    assert result["rows"][0]["independent_suffix_20step"]["endpoint_minus_zero_prefix"] is None
    assert result["rows"][0]["independent_suffix_20step"]["last_accepted_state_metrics"] is not None
    assert result["equal_budget_certified"] is False
