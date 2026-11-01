from pathlib import Path
import runpy
import pytest
from experiments.spectral_reachability import p2_h15_host_lifecycle_20261029_v1 as m


@pytest.fixture(scope='module')
def fixtures():
    p=Path(__file__).parent
    native=runpy.run_path(str(p/'test_agent_p3_parent_native_cuda_20261029_v1.py'))['NumpyNative']
    att=runpy.run_path(str(p/'test_p2_h15_host_attestation_20261029_v1.py'))
    return native,att


def session(tmp_path, att):
    p=tmp_path/'source'; p.write_bytes(b'x')
    import hashlib
    return m.att.Session(m.bridge.build_bridge(),att['Model'](),att['Device'](),att['Flags'],
                         {str(p):hashlib.sha256(b'x').hexdigest()},m.att.model_hash(att['Model']()),fixture=True)


def test_full_dual_root_audited_and_tamper_rejected(tmp_path,fixtures):
    native,att=fixtures
    out=tmp_path/'run'; s=session(tmp_path,att)
    result=m.run_fixture(out,s,lambda t,k:native(t,k))
    receipt=m.consume(out,result['report_sha256'],permit_fixture=True)
    assert receipt['root_status']==['sealed','sealed']
    assert receipt['attestation']['model_before']==receipt['attestation']['model_after']
    assert all(x['attempted']['predict_samples']==132 for x in receipt['resources'])
    events=m.transport.read_journal(out/'lifecycle.jsonl')
    assert events[1]['kind']=='attestation_before'
    with pytest.raises(ValueError): m.consume(out,result['report_sha256'])
    with (out/'root0'/'events.jsonl').open('ab') as f:f.write(b'{}\n')
    with pytest.raises(ValueError): m.consume(out,result['report_sha256'],permit_fixture=True)


def test_failed_root_is_saved_and_stops_next(tmp_path,fixtures):
    native,att=fixtures
    out=tmp_path/'run'; result=m.run_fixture(out,session(tmp_path,att),lambda t,k:native(t,k,fail=True))
    receipt=m.consume(out,result['report_sha256'],permit_fixture=True)
    assert receipt['root_status']==['failed','not_started']
    assert not (out/'root1').exists()


def test_host_cannot_be_dispatched():
    with pytest.raises(ValueError,match='not authorized'): m.run_host(None,None)


def test_precision_change_fails_closed_and_is_saved(tmp_path,fixtures):
    native,att=fixtures
    class MutableFlags(att['Flags']):
        changed=False
        @classmethod
        def is_autocast_enabled(cls,device): return cls.changed
    s=session(tmp_path,att); s.torch=MutableFlags
    def make(t,sink):
        obj=native(t,sink)
        original=obj.signal
        def signal(req,ledger):
            value=original(req,ledger); MutableFlags.changed=True; return value
        obj.signal=signal
        return obj
    out=tmp_path/'run'; result=m.run_fixture(out,s,make)
    receipt=m.consume(out,result['report_sha256'],permit_fixture=True)
    assert receipt['root_status']==['failed','not_started']
    assert receipt['attestation'] is None and receipt['attestation_error'] is not None


def test_internal_engine_does_not_allow_host_kind(tmp_path,fixtures):
    _,att=fixtures
    with pytest.raises(ValueError):
        m._run(tmp_path/'run',session(tmp_path,att),lambda *_:None,'HOST_CUDA')
