from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_backtrack_cuda_replay_consume_20261030 as consumer


@pytest.mark.skipif(
    not Path("/home/gbk/code/LeWorldModel/le-wm-main/results/p1_backtrack_cuda_replay_20261030_MAIN01.json").is_file(),
    reason="sealed CUDA replay report not present",
)
def test_real_saved_replay_and_endpoint_tamper():
    replay = consumer.replay
    result = replay.one.saved.read_verified(
        replay.ROOT / "results/p1_backtrack_cuda_replay_20261030_MAIN01.json",
        "83ea957ecb38025e6b5489a0bf843b112906b0954b8b6c62238c10461b31cfa4")
    manifest = replay.one.saved.read_verified(
        replay.ROOT / "results/p1_backtrack_cuda_replay_20261030_prepare_MAIN01.json",
        "f245d30ae0b28591a9c7308817951b2a38f7c89b81f24d6f6262434f6044310e")
    action = replay.one.saved.read_verified(replay.INPUT, replay.INPUT_SHA)
    original = replay.one.saved.read_verified(
        replay.ROOT / "results/p1_bounded_backtrack_host_20261030_MAIN01.json", replay.inventory.SHA)
    report = replay.one.saved.read_verified(replay.one.source.REPORT, replay.one.source.REPORT_SHA)
    prepared = replay.one.saved.read_verified(report["prepared"], report["prepare_sha256"])
    assert consumer.verify(result, manifest, action, original, report, prepared)["all_exact"]
    result["rows"][0]["endpoints"][0][0] += 1
    with pytest.raises(ValueError, match="saved context mismatch"):
        consumer.verify(result, manifest, action, original, report, prepared)
