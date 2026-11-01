import json
import numpy as np
import pytest
from experiments.spectral_reachability import codex_p2_symmetric_h3_run_20261030_v1 as m


class Fake:
    def __init__(self, meter): self.meter = meter
    def packet(self):
        return self.meter.call("packet", lambda: self.meter.call(
            "action_encoder", lambda: self.meter.call("predict", lambda: None)))
    def rollout(self, context, u):
        for _ in range(3): self.packet()
        z = np.pad(u, (0,162))
        return [dict(z=z.tolist(), i=i) for i in range(3)], z
    def jacobian(self, context, u):
        for _ in range(3): self.packet()
        return np.pad(np.eye(30), ((0,162),(0,0)))


def test_complete_equal_counts_and_legal_real_candidates():
    case = dict(context={}, projection=np.eye(192), goal=np.ones(192))
    for arm in ("joint", "suffix"):
        events = []
        meter = m.Meter(events.append, lambda: 0.)
        row = m.run_arm(Fake(meter), case, arm, meter)
        assert row["attempted"] == row["returned"] == m.LIMITS
        for e in events:
            if e["kind"] == "candidate":
                assert np.linalg.norm(e["witness"]["logits"]) <= m.CAP


def test_failure_and_deadline_are_charged():
    events = []
    meter = m.Meter(events.append, lambda: 0.)
    with pytest.raises(RuntimeError):
        meter.call("svd", lambda: (_ for _ in ()).throw(RuntimeError("broken")))
    assert meter.attempted["svd"] == 1 and meter.returned["svd"] == 0
    assert events[-1]["kind"] == "failure"
    times = iter([0., 450.])
    meter = m.Meter(events.append, lambda: next(times))
    with pytest.raises(RuntimeError): meter.call("packet", lambda: None)
    assert meter.attempted["packet"] == 0


def test_replay_context_tamper_fails():
    class Broken(Fake):
        def rollout(self, context, u):
            contexts, z = super().rollout(context, u)
            if self.meter.attempted["packet"] > 33: contexts[0]["i"] = 9
            return contexts, z
    meter = m.Meter(lambda e: None, lambda: 0.)
    with pytest.raises(ValueError, match="replay mismatch"):
        m.run_arm(Broken(meter), dict(context={}, projection=np.eye(192), goal=np.ones(192)), "joint", meter)


def test_journal_exclusive_chain(tmp_path):
    p = tmp_path/"journal"
    j = m.Journal(p)
    j.emit(dict(kind="a")); j.emit(dict(kind="b")); j.close()
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert rows[1]["previous"] == rows[0]["sha256"]
    with pytest.raises(FileExistsError): m.Journal(p)


def test_real_roots_prepare_and_source_binding(tmp_path):
    path = tmp_path/(m.NAME+"_prepare.json")
    sha = m.prepare(path)
    _, cases = m.verify_prepare(path, sha)
    assert len(cases) == 4
    with pytest.raises(ValueError): m.verify_prepare(path, "0"*64)
