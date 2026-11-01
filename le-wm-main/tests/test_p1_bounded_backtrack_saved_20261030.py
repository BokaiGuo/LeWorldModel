"""Regression against the sealed real report, including a negative control."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_bounded_backtrack_consume_20261030 as consumer


@pytest.mark.skipif(
    not Path("/home/gbk/code/LeWorldModel/le-wm-main/results/p1_bounded_backtrack_host_20261030_MAIN01.json").is_file(),
    reason="sealed host CUDA report not available",
)
def test_real_saved_and_changed_choice_rejected():
    one = consumer.producer.base.one
    result = one.saved.read_verified(
        str(one.source.ROOT / "results/p1_bounded_backtrack_host_20261030_MAIN01.json"),
        "bd829508ac62df40a82dd219c220367ae3925ecda77075b05deda2a18a7423fb",
    )
    report = one.saved.read_verified(one.source.REPORT, one.source.REPORT_SHA)
    prepared = one.saved.read_verified(report["prepared"], report["prepare_sha256"])
    assert consumer.verify(result, report, prepared)["selected_counts"] == [815, 142, 198, 100, 25]
    bad = deepcopy(result)
    choice = bad["rows"][0]["trajectory"][0]["selected"]
    bad["rows"][0]["trajectory"][0]["selected"] = None if choice is not None else 0
    with pytest.raises(ValueError, match="selected trial mismatch"):
        consumer.verify(bad, report, prepared)
