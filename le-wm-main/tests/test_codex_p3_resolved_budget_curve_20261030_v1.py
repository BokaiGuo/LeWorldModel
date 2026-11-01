import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p3_resolved_budget_curve_20261030_v1 as curve


def inputs():
    checked = curve.audit.build(curve.RUN, curve.RUN_SHA)
    batch = json.loads((curve.RUN/"report.json").read_bytes())
    prep = json.loads(curve.PREP.read_bytes())
    contract = json.loads(curve.CONTRACT.read_bytes())
    return checked, batch, prep, contract


def test_actual_curve_preserves_failed_denominator():
    result = curve.build()
    p20, p30 = result["points"]
    assert not p20["full_roster_consumed"] and p30["full_roster_consumed"]
    missing = p20["arms"][0]
    assert missing["status"] == "missing_or_failed"
    assert missing["axis_attempted"] is None and missing["root_model_costs"] is None
    assert missing["failure"]["unobserved_work"] == "unknown"
    assert p20["arms"][1]["axis_attempted"]["jacobian"] == 2
    assert p30["arms"][0]["axis_attempted"]["jacobian"] == 0
    assert result["P3_admission"] is False


def test_under_cap_is_valid_without_padding_and_over_cap_rejected():
    checked, batch, prep, contract = inputs()
    saved = json.loads(Path(batch["rows"][1]["receipt"]).read_bytes())
    def loader(row):
        result = json.loads(Path(row["receipt"]).read_bytes())
        if row["point"] == 20 and row["arm"] == "signal":
            result = copy.deepcopy(saved)
            result["rows"][0]["attempted"]["svd"] = 1
        return result
    result = curve.assemble(checked, batch, prep, contract, loader)
    assert result["points"][0]["arms"][1]["axis_attempted"]["svd"] == 1
    def over(row):
        result = loader(row)
        result["rows"][0]["attempted"]["svd"] = 3
        return result
    with pytest.raises(ValueError, match="per-axis caps"):
        curve.assemble(checked, batch, prep, contract, over)
