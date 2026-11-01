import json

from experiments.spectral_reachability import p1_joint_anchor_table_20261030 as original
from experiments.spectral_reachability import p1_uphill_fixedtail_analysis_20261030 as analysis


def test_three_role_pairing_and_missing_endpoint():
    data = {key: json.loads((original.ROOT / path).read_bytes())
            for key, (path, _) in original.FILES.items()}
    data["physical_consumed"]["rows"][1]["completed_endpoint"] = None
    data["physical_consumed"]["rows"][1]["status"] = "native_stopped"
    result = analysis.analyze(data["h10_prepare"], data["physical_prepare"], data["physical_consumed"])
    assert len(result["rows"]) == 24
    assert result["rows"][0]["fixed_original_suffix"]["same_suffix_endpoint_minus_zero_prefix"] is None
    assert result["rows"][1]["fixed_original_suffix"]["endpoint"] is None
    assert result["rows"][2]["fixed_original_suffix"]["same_suffix_endpoint_minus_zero_prefix"] is None
