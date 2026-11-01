"""Only RAM/metadata fixtures; never invoke torch or model/physics execution."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_equal144_baseline_cuda_20261028_v1 as m


@pytest.fixture
def task():
    p,h=m.native.PREPARES["kinodynamic"]
    slot=m.native.saved.load(p,h)["slots"][0]
    return m.old.load_export(slot["task_export"],slot["task_sha256"])["gn_task"]


def literal(value):
    a=np.asarray(value,np.float32)
    return dict(shape=list(a.shape),dtype="float32",values=a.astype(float).tolist())


class MockBackend:
    """Synthetic deterministic packets; fake neural receipts ONLY inside RAM tests."""
    def packet(self,task,before,raw,meter):
        z=np.asarray(before["latent_history"],np.float32)[None]
        outgoing=np.concatenate((np.asarray(before["action_history"],np.float32),m.core.normalized_tokens(task,raw)))[None]
        ae=meter.begin("action_encoder",[literal(outgoing)]); meter.end(ae,literal(outgoing))
        api=meter.begin("predict_API",[literal(z),literal(outgoing)])
        pred=meter.begin("predictor",[literal(z),literal(outgoing)])
        prediction=z.copy(); prediction[0,-1]+=np.float32(np.sum(np.asarray(raw,np.float32))*.01)
        meter.end(pred,literal(prediction))
        proj=meter.begin("pred_proj",[literal(prediction.reshape(3,192))]); meter.end(proj,literal(prediction.reshape(3,192)))
        meter.end(api,literal(prediction))
        return m.beam.rolled(task,before,raw,prediction[0,-1])

    def advance(self,task,before,raw,meter):
        paths=[]; c=deepcopy(before); meter.packet=0
        for i in range(len(raw)//5):
            c=self.packet(task,c,raw[5*i:5*i+5],meter); paths.append(c)
            meter.emit("native_packet",packet=i,context=c); meter.packet+=1
        return paths

    def rollout(self,task,raw,meter,mode):
        return self.advance(task,task["root_context"],raw,meter)

    def jacobian(self,task,incumbent,meter):
        c=deepcopy(task["root_context"]); meter.packet=0
        for i in range(2):
            c=self.packet(task,c,incumbent["commands"][5*i:5*i+5],meter)
            meter.emit("jacobian_forward_packet",packet=i,context=c); meter.packet+=1
        z=c["latent_history"][-1]; meter.emit("jacobian_forward_complete",terminal=z,logits=incumbent["logits"])
        rows=[]
        for i in range(192):
            row=meter.call("vjp_directions",lambda:np.zeros(20)); rows.append(row)
            meter.emit("jacobian_row",row=i,raw_control_derivative=row.tolist())
        j=np.stack(rows); meter.emit("joint_logit_jacobian",terminal=z,jacobian=j.tolist())
        return np.asarray(z),j


def test_protocol_no_padding_no_equal_FLOPs():
    p=m.protocol()
    assert p["spectral"]["candidates"]==68 and p["spectral"]["total_predict"]==144
    assert sum(p["spectral"][k] for k in ("nominal","AD_forward","candidate_predict","winner_whole","winner_segmented"))==144
    assert p["total_predict_API"]==576 and not p["equal_FLOPs"] and not p["equal_AD"]
    assert p["wall_seconds"]==900 and p["padding"]==p["retries"]==0


def test_bank68_first40_preserves_old_kernel(task):
    j=np.arange(192*20,dtype=float).reshape(192,20)/10000
    z=np.asarray(task["root_context"]["latent_history"][-1])
    bank=m.spectral_bank(task,j,z)
    old=m.old.bank(task,j,z)
    assert len(bank["logits"])==68 and len(bank["right_vectors"])==20
    assert bank["logits"][:40]==old["logits"]
    assert bank["commands"][:40]==old["commands"]
    assert np.max(np.linalg.norm(np.asarray(bank["logits"]),axis=2))<=task["packet_logit_cap"]
    assert np.asarray(bank["commands"]).shape==(68,10,2)


def test_illegal_J_shape_rejected(task):
    with pytest.raises(ValueError):m.spectral_bank(task,np.zeros((192,10)),np.zeros(192))


@pytest.mark.parametrize("spectral",[False,True])
def test_full_mock_search_real_budget_and_saved_checks(task,spectral):
    events=[]; meter=m.Meter(events.append,spectral=spectral); backend=MockBackend()
    result=m.spectral_search(task,0,backend,meter) if spectral else m.beam_search(task,0,backend,meter)
    row=dict(arm=m.ARMS[1 if spectral else 0],resources=meter.counts(),math=meter.math_counts(),predictor_samples=meter.predictor_samples())
    assert meter.counts()==m.sampling.budget()
    m.checked_operations(task,events,row)
    m.checked_packets(task,events,result,row["arm"])
    m.checked_schedule(events,result,row["arm"])
    if spectral:m.checked_spectral(task,result)
    else:
        emitted=m.checked_beam(task,result,0)
        assert [e["event"] for e in events if e["kind"]=="beam_event"]==emitted


def test_api_attempt_cap_before145():
    meter=m.Meter(lambda e:None)
    meter.group="MOCK"; meter.packet=0
    for i in range(144):
        t=meter.begin("action_encoder",[]); meter.end(t,{"MOCK":True})
    with pytest.raises(ValueError,match="cap"):meter.begin("action_encoder",[])


def test_math_caps_and_beam_forbidden_math():
    meter=m.Meter(lambda e:None,spectral=True)
    assert meter.math_call("svd",lambda:1)==1
    meter.math_call("svd",lambda:2)
    with pytest.raises(ValueError,match="cap"):meter.math_call("svd",lambda:3)
    with pytest.raises(ValueError,match="only spectral"):m.Meter(lambda e:None).math_call("svd",lambda:1)


def test_failed_math_durable_no_retry():
    events=[]; meter=m.Meter(events.append,spectral=True)
    def bad():raise RuntimeError("MOCK failure")
    with pytest.raises(RuntimeError):meter.math_call("svd",bad)
    assert [e["kind"] for e in events]==["math_attempted","math_failed"]
    assert meter.math_counts()["svd"]==dict(attempted=1,completed=0)
    with pytest.raises(ValueError):meter.math_call("svd",lambda:1)


def test_failed_API_no_retry():
    events=[]; meter=m.Meter(events.append)
    def bad():raise RuntimeError("MOCK api failure")
    with pytest.raises(RuntimeError):meter.call("predict_API",[],bad)
    assert events[-1]["kind"]=="dispatch_failed"
    assert meter.counts()["predict_API"]==dict(attempted=1,completed=0)
    with pytest.raises(ValueError):meter.begin("predict_API",[])


def test_missing_all68_refused(task):
    meter=m.Meter(lambda e:None,spectral=True); result=m.spectral_search(task,0,MockBackend(),meter)
    result["candidates"].pop()
    with pytest.raises(ValueError):m.checked_spectral(task,result)


def loader_fixture():
    policy="/MOCK_POLICY"
    p=dict(policy=policy,expected_model_hash="tensor",bindings_sha256={policy+"/config.json":"c",policy+"/weights.pt":"w"})
    receipt=dict(schema=m.NAME+".loader_receipt",prepare_sha256="prepare",adapter_source_sha256=m.sha(m.__file__),
        legacy_loader_source_sha256=m.native.SOURCES["experiments/checkpoint_compat.py"],device="cuda:0",frozen_eval=True,
        missing=[],unexpected=[],tensor_hash_before="tensor",files=[dict(path=path,sha256=h,hash_buffer_sha256=h,
        deserialize_buffer_sha256=h,same_buffer=True,bytes=123) for path,h in p["bindings_sha256"].items()])
    return p,receipt


def test_loader_samebuffer_positive_schema_only():
    p,r=loader_fixture(); m.validate_loader(r,p,"prepare")


@pytest.mark.parametrize("bad",["reopen","other_bytes","missing_file","cpu","old_contract","old_source","tensor"])
def test_loader_mutations(bad):
    p,r=loader_fixture()
    if bad=="reopen":r["files"][0]["same_buffer"]=False
    elif bad=="other_bytes":r["files"][0]["deserialize_buffer_sha256"]="other"
    elif bad=="missing_file":r["files"].pop()
    elif bad=="cpu":r["device"]="cpu"
    elif bad=="old_contract":r["prepare_sha256"]="old"
    elif bad=="old_source":r["adapter_source_sha256"]="old"
    else:r["tensor_hash_before"]="other"
    with pytest.raises(ValueError):m.validate_loader(r,p,"prepare")


def test_actual_metadata_material_only_no_model_hashing():
    p=m.material()
    assert len(p["slots"])==2 and len(p["group_denominator"])==4
    assert len(m.canonical(p))<=256*1024
    assert p["inherited_encoding_once"]["encoder"]["completed"]==8
    assert p["inherited_other_costs"] is None


def test_saved_missing_return_refused(task):
    meter=m.Meter(lambda e:None)
    with pytest.raises(ValueError):m.checked_operations(task,[],dict(arm=m.ARMS[0],resources=m.sampling.budget(),math=meter.math_counts(),predictor_samples=dict(attempted=144,completed=144)))


def test_byte_hash_is_streaming_RAM(monkeypatch):
    from io import BytesIO
    monkeypatch.setattr(m.Path,"open",lambda *a,**k:BytesIO(b"MOCK_BYTE_INPUT"))
    assert m.sha("/MOCK")==hashlib.sha256(b"MOCK_BYTE_INPUT").hexdigest()
