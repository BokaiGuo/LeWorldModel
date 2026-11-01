"""V2 exact protocol reader and frozen arithmetic validation."""
import builtins
import pytest
from experiments.spectral_reachability import p3p4_goal_patch_outcomes_20261027_v2 as m


def test_large_protocol_exact():
    reader=m.Evidence(m.PROJECT);assert len(reader.raw(m.LARGE))>32_000_000
    with pytest.raises(ValueError):reader.raw(m.LARGE,'wrong')
    reader.recheck()


def test_real_saved_audit_no_runtime(monkeypatch):
    original=builtins.__import__
    def no_runtime(name,*a,**kw):
        assert name.split('.')[0] not in {'torch','gymnasium','pymunk','mujoco'},name
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',no_runtime)
    path=m.PROJECT/m.v1.RUN/'report.json'
    result=m.v1.compute(path.parent,m.io.sha(path.read_bytes()),m.Evidence(m.PROJECT))
    assert len(result['rows'])==8 and result['physical_env_step_attempts']<=90


def test_prepare_verify_tamper(tmp_path):
    out=tmp_path/'prepared';m.prepare(out);sha=m.io.sha((out/'prepare.json').read_bytes())
    assert m.validate_prepared(out,sha)['large_protocol_sha256']==m.LARGE_SHA
    with pytest.raises(ValueError):m.prepare(out)
    with pytest.raises(ValueError):m.validate_prepared(out,'wrong')
    (out/'protocol.json').write_text('{}')
    with pytest.raises(ValueError):m.validate_prepared(out,sha)
