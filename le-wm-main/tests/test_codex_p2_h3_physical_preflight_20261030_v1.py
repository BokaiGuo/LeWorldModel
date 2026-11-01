from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import codex_p2_h3_physical_preflight_20261030_v1 as m


def test_actual_roots_and_missing_actions():
    p = m.inspect()
    assert len(p["roots"]) == 4 and len(p["slots"]) == 8
    assert all(x["raw_actions"] is None and not x["physical_attempted"] for x in p["slots"])
    assert not p["P2_admission"]


def test_source_tampering_rejected():
    protocol, cases = m.checks.run.verify_prepare(m.PREP, m.PREP_SHA)
    refs = m.physical.validate_prepared(m.REF, m.REF_SHA)
    damaged = deepcopy(refs)
    damaged["packets"][0]["goal"][0] += 1
    with pytest.raises(ValueError): m.source_join(cases, protocol, damaged)


class FakeTensor:
    class Device: type = "cuda"
    device = Device()
    def __init__(self, a): self.a = a
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


def test_capture_and_failure_closed():
    a = np.arange(10, dtype=np.float32).reshape(5, 2)/10
    events = []
    record = m.save_cuda_raw_packet(FakeTensor(a), index=0, sink=events.append)
    assert np.array_equal(np.frombuffer(bytes.fromhex(record["raw_hex"]), dtype="<f4").reshape(5,2), a)
    assert events == [record]
    with pytest.raises(ValueError): m.save_cuda_raw_packet(FakeTensor(a.astype(np.float64)), index=0, sink=events.append)
    with pytest.raises(RuntimeError): m.save_cuda_raw_packet(FakeTensor(a), index=0, sink=lambda _: (_ for _ in ()).throw(RuntimeError("sink")))


def test_cpu_rejected():
    x = FakeTensor(np.zeros((5, 2), np.float32))
    x.device = type("Device", (), {"type": "cpu"})()
    with pytest.raises(ValueError): m.save_cuda_raw_packet(x, index=0, sink=lambda _: None)
