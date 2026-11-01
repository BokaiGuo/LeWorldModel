import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import codex_sra_io_transfer_meter_20261030 as m


def ledger(tmp_path, **overrides):
    caps = {axis: 1_000_000_000 for axis in m.AXES}
    caps.update(overrides)
    source = m.sha(Path(m.__file__).read_bytes())
    return m.Ledger(tmp_path / "meter.jsonl", binding_sha256="a" * 64, caps=caps, source_sha256=source), source


class TensorFixture:
    def __init__(self, dev, count=5, width=4):
        self.device, self.count, self.width = SimpleNamespace(type=dev), count, width
    def numel(self):
        return self.count
    def element_size(self):
        return self.width


def consume(l, source):
    digest = l.finish()
    return m.consume(l.path, digest, "a" * 64, source)


def test_explicit_reads_load_and_two_transfer_directions(tmp_path):
    l, source = ledger(tmp_path)
    path = tmp_path / "asset"
    path.write_bytes(b"fixture contents")
    def load():
        assert l.read_asset("read1", path, expected_sha256=m.sha(path.read_bytes())) == b"fixture contents"
        return object()
    assert l.load_model("load1", load, wall_limit_ns=500_000_000) is not None
    out = l.transfer("up", TensorFixture("cpu"), SimpleNamespace(type="cuda"),
                     copy_fn=lambda tensor, dest: TensorFixture(dest.type))
    assert out.device.type == "cuda"
    l.transfer("down", out, SimpleNamespace(type="cpu"), copy_fn=lambda tensor, dest: TensorFixture(dest.type))
    report = consume(l, source)
    snap = report["snapshot"]
    assert snap["attempted_reservations"]["file_read_bytes"] == len(b"fixture contents")
    assert snap["attempted_reservations"]["model_load_attempts"] == 1
    assert snap["attempted_reservations"]["host_to_device_payload_bytes"] == 20
    assert snap["attempted_reservations"]["device_to_host_payload_bytes"] == 20
    assert snap["total_file_io_bytes"] is None and snap["physical_bus_bytes"] is None
    assert snap["model_load_internal_io_bytes"] is None and not snap["equal_total_budget"]
    assert next(e for e in report["observed"] if e["id"] == "up")["details"]["fixture"] is True
    assert next(e for e in report["observed"] if e["id"] == "load1")["observed"]["internal_io_bytes"] is None
    assert report["callbacks_reexecuted"] is False


@pytest.mark.parametrize("value", [None, 2])
def test_missing_or_exhausted_cap_blocks_before_file_read(tmp_path, value):
    l, source = ledger(tmp_path, file_read_bytes=value)
    asset = tmp_path / "asset"
    asset.write_bytes(b"123")
    with pytest.raises(ValueError, match="cap"):
        l.read_asset("read", asset, expected_sha256=m.sha(b"123"))
    assert not l.pending
    assert consume(l, source)["snapshot"]["total_file_io_bytes"] is None


def test_failure_keeps_attempt_pending_no_refund_or_reuse(tmp_path):
    l, source = ledger(tmp_path)
    path = tmp_path / "asset"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="SHA mismatch"):
        l.read_asset("read", path, expected_sha256=m.sha(b"expected"))
    with pytest.raises(ValueError, match="operation id"):
        l.read_asset("read", path, expected_sha256=m.sha(b"bad"))
    report = consume(l, source)
    assert report["snapshot"]["pending"]["read"]["reserved"] == {"file_read_bytes": 3}
    assert report["snapshot"]["returned_reservations"]["file_read_bytes"] == 0
    assert report["snapshot"]["total_file_io_bytes"] is None


def test_transfer_failure_retained_and_never_fabricated(tmp_path):
    l, source = ledger(tmp_path)
    def fail(*args):
        raise OSError("copy failed")
    with pytest.raises(OSError):
        l.transfer("copy", TensorFixture("cpu"), SimpleNamespace(type="cuda"), copy_fn=fail)
    snap = consume(l, source)["snapshot"]
    assert snap["pending"]["copy"]["reserved"]["host_to_device_payload_bytes"] == 20
    assert snap["physical_bus_bytes"] is None


def test_load_overrun_is_pending_not_preemptive(tmp_path, monkeypatch):
    l, source = ledger(tmp_path)
    times = iter([0, 101])
    monkeypatch.setattr(m.time, "monotonic_ns", lambda: next(times))
    with pytest.raises(ValueError, match="wall reservation exceeded"):
        l.load_model("load", lambda: object(), wall_limit_ns=100)
    report = consume(l, source)
    assert report["snapshot"]["pending"]["load"]["reserved"]["load_wall_ns"] == 100
    assert report["snapshot"]["model_load_internal_transfer_bytes"] is None


def test_hash_chain_footer_and_anchor_refusal(tmp_path):
    l, source = ledger(tmp_path)
    l.load_model("load", lambda: object(), wall_limit_ns=500_000_000)
    digest = l.finish()
    with pytest.raises(ValueError, match="anchor"):
        m.consume(l.path, "0" * 64, "a" * 64, source)
    events = [json.loads(line) for line in l.path.read_bytes().splitlines()]
    events[1]["payload"]["reserved"]["model_load_attempts"] = 2
    tampered = b"".join(m.canonical(e) + b"\n" for e in events)
    l.path.write_bytes(tampered)
    with pytest.raises(ValueError, match="chain"):
        m.consume(l.path, m.sha(tampered), "a" * 64, source)


def test_unfinished_log_rejected_and_exclusive(tmp_path):
    l, source = ledger(tmp_path)
    with pytest.raises(FileExistsError):
        ledger(tmp_path)
    l.close_partial()
    with pytest.raises(ValueError, match="footer"):
        m.consume(l.path, m.sha(l.path.read_bytes()), "a" * 64, source)


def test_transfer_cap_and_wrong_destination_fail_closed(tmp_path):
    l, source = ledger(tmp_path, host_to_device_payload_bytes=None)
    called = []
    with pytest.raises(ValueError, match="cap"):
        l.transfer("blocked", TensorFixture("cpu"), SimpleNamespace(type="cuda"),
                   copy_fn=lambda *args: called.append(True))
    assert not called
    assert consume(l, source)["snapshot"]["total_device_transfer_bytes"] is None


def test_wrong_destination_stays_pending(tmp_path):
    l, source = ledger(tmp_path)
    with pytest.raises(ValueError, match="direction or size"):
        l.transfer("copy", TensorFixture("cpu"), SimpleNamespace(type="cuda"),
                   copy_fn=lambda *args: TensorFixture("cpu"))
    assert "copy" in consume(l, source)["snapshot"]["pending"]
