import hashlib
import pytest

from experiments.spectral_reachability import codex_p1_cp5_durable_budget_20261030_v1 as budget
from experiments.spectral_reachability import codex_p1_cp5_unified_consumer_20261030_v1 as consumer


def test_encoder_consume_and_tamper(tmp_path):
    caps = dict.fromkeys(budget.AXES, 1)
    path = tmp_path / "events"
    meter = budget.Budget(path, caps, 10)
    meter.call("encoder", 1, lambda: None)
    meter.call("projector", 1, lambda: None)
    sha = meter.close()
    assert consumer.consume_encoder(path, sha, caps, 10)["attempted"]["projector"] == 1
    raw = path.read_bytes().replace(b'"seq":1', b'"seq":9')
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="chain"):
        consumer.consume_encoder(path, hashlib.sha256(raw).hexdigest(), caps, 10)
    path.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="truncation"):
        consumer.consume_encoder(path, hashlib.sha256(raw[:-1]).hexdigest(), caps, 10)


def test_failure_and_block_are_not_complete(tmp_path):
    caps = dict.fromkeys(budget.AXES, 1)
    path = tmp_path / "failure"
    meter = budget.Budget(path, caps, 10)
    with pytest.raises(RuntimeError):
        meter.call("encoder", 1, lambda: (_ for _ in ()).throw(RuntimeError("fault")))
    sha = meter.close()
    assert consumer.consume_encoder(path, sha, caps, 10)["status"] == "failed"
    path = tmp_path / "blocked"
    meter = budget.Budget(path, caps, 10)
    meter.call("encoder", 1, lambda: None)
    with pytest.raises(ValueError):
        meter.call("encoder", 1, lambda: None)
    sha = meter.close()
    assert consumer.consume_encoder(path, sha, caps, 10)["status"] == "failed"
