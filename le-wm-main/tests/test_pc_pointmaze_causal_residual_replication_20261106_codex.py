import json

import pytest

from experiments import pc_pointmaze_causal_residual_replication_20261106_codex as replication

pytestmark = pytest.mark.skipif(not replication.REPLAY.exists(), reason="requires generated replay manifest")


def test_fixed_panel_is_bound_to_replay():
    namespace = replication.configure()
    replay = json.loads(replication.REPLAY.read_text())
    assert replay["roots"] == list(range(1892, 1900))
    assert replay["strict_replay_atol_1e_4"]
    assert namespace["REPLAY"] == replication.REPLAY
    assert namespace["PROTOCOL"] == replication.PROTOCOL
    assert namespace["PRIVATE"].name.startswith(replication.NAME)


def test_replay_protocol_drift_is_rejected(monkeypatch, tmp_path):
    replay = json.loads(replication.REPLAY.read_text())
    replay["roots"][0] = 1884
    altered = tmp_path / "replay.json"
    altered.write_text(json.dumps(replay))
    monkeypatch.setattr(replication, "REPLAY", altered)
    with pytest.raises(ValueError, match="replay manifest mismatch"):
        replication.configure()
