from copy import deepcopy
import numpy as np
import pytest

from experiments.spectral_reachability import codex_p2_program_library_pair_20261030_v1 as p
from experiments.spectral_reachability import codex_p2_program_library_saved_20261030_v1 as saved
from experiments.spectral_reachability import codex_p2_program_library_cuda_adapter_20261030_v1 as cuda


def inputs():
    library,bank=p.candidates(np.eye(20))
    context={"latent_history":[[0.0]*192],"action_history":[]}
    def transition(c, packet):
        c["latent_history"]=[(np.asarray(c["latent_history"][-1])+
            float(packet.sum())/100).tolist()]
        c["action_history"].append(packet.tolist())
        return c
    return library,bank,context,transition


def test_fixed_program_cartesian_library_and_budget():
    library,bank,context,transition=inputs()
    assert [len(x) for x in library]==[4,4]
    assert len(bank["joint"])==len(bank["library_chain"])==16
    assert len({tuple(x["program_ids"]) for x in bank["library_chain"]})==16
    assert len({x["program_ids"][0] for x in bank["library_chain"]})==4
    for arm in p.ARMS:
        events=[]
        result=p.execute_arm(context,[0.0]*192,arm,bank[arm],transition,events.append)
        assert result["packet_returns"]==34 and len([x for x in events if x["kind"]=="packet_return"])==34
        assert len([x for x in events if x["kind"]=="candidate_return"])==17
        for row in result["candidates"]:
            assert row["edges"][0]["output_context_sha256"]==row["edges"][1]["input_context_sha256"]
            assert "".join(e["raw_float32_hex"] for e in row["edges"])==row["candidate"]["raw_float32_hex"]


def test_failure_is_recorded_as_intent_without_retry():
    _,bank,context,_=inputs();events=[];calls=[]
    def fail(c,a):
        calls.append(1)
        raise RuntimeError("failed real callback")
    with pytest.raises(RuntimeError):
        p.execute_arm(context,[0.0]*192,"library_chain",bank["library_chain"],fail,events.append)
    assert len(calls)==1 and [e["kind"] for e in events]==["packet_intent","packet_failure"]


def test_wrong_bank_and_nondeterministic_replay_fail_closed():
    _,bank,context,transition=inputs()
    wrong=deepcopy(bank["joint"]);wrong.reverse()
    with pytest.raises(ValueError):
        p.execute_arm(context,[0.0]*192,"joint",wrong,transition,lambda _:None)
    n=[0]
    def drifting(c,a):
        n[0]+=1;c=transition(c,a);c["latent_history"][-1][0]+=n[0]*1e-9;return c
    with pytest.raises(ValueError,match="replay"):
        p.execute_arm(context,[0.0]*192,"joint",bank["joint"],drifting,lambda _:None)


def test_basis_validation_and_literal_bits():
    with pytest.raises(ValueError):
        p.candidates(np.zeros((20,20)))
    _,v=p.canonical_basis(np.concatenate([np.eye(20),np.zeros((172,20))]))
    _,bank=p.candidates(v)
    assert all(len(bytes.fromhex(x["raw_float32_hex"]))==80 for a in p.ARMS for x in bank[a])


@pytest.mark.parametrize("mutation", ["action","context","program","cost","winner"])
def test_saved_chain_tamper_rejected(mutation):
    _,bank,context,transition=inputs()
    result=p.execute_arm(context,[0.0]*192,"library_chain",bank["library_chain"],transition,lambda _:None)
    assert saved.validate(result,np.eye(20),context,[0.0]*192)["candidates"]==16
    bad=deepcopy(result)
    if mutation=="action":
        bad["candidates"][0]["edges"][0]["raw_float32_hex"]="00"*40
    elif mutation=="context":
        bad["candidates"][0]["edges"][0]["output_context"]["latent_history"][-1][0]+=1
    elif mutation=="program":
        bad["candidates"][0]["edges"][1]["program_id"]="bad"
    elif mutation=="cost":
        bad["candidates"][0]["cost"]+=1
    else:
        bad["winner_index"]=99
    with pytest.raises(ValueError):
        saved.validate(bad,np.eye(20),context,[0.0]*192)


def test_no_cpu_or_unmetered_neural_adapter():
    class Device:
        type="cpu"
    with pytest.raises(ValueError,match="CUDA"):
        cuda.CudaAdapter(None,Device(),None)
