from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p1_backtrack_action_inventory_20261030 as inventory


@pytest.mark.skipif(
    not Path("/home/gbk/code/LeWorldModel/le-wm-main/results/p1_bounded_backtrack_host_20261030_MAIN01.json").is_file(),
    reason="sealed host CUDA report not available",
)
def test_real_inventory_actions_and_fail_closed():
    one = inventory.verifier.producer.base.one
    result = one.saved.read_verified(one.source.ROOT / "results/p1_bounded_backtrack_host_20261030_MAIN01.json", inventory.SHA)
    report = one.saved.read_verified(one.source.REPORT, one.source.REPORT_SHA)
    prepared = one.saved.read_verified(report["prepared"], report["prepare_sha256"])
    value = inventory.build(result, report, prepared)
    assert len(value["rows"]) == 16
    assert not value["physical_authorized"] and not value["P1_admission"]
    for row in value["rows"]:
        assert np.asarray(row["raw15"], dtype="<f4").tobytes().hex() == row["raw15_le_float32_hex"]
    result["rows"][0]["trajectory"][0]["selected"] = 99
    with pytest.raises(ValueError):
        inventory.build(result, report, prepared)
