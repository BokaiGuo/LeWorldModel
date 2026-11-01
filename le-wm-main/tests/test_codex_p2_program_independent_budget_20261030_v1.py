from contextlib import contextmanager
import numpy as np
import pytest
from experiments.spectral_reachability import codex_p2_program_independent_budget_20261030_v1 as m


@contextmanager
def mock_scope(_):
    yield


class FakeAdapter:
    def __init__(self,budget):
        self.budget=budget

    def root_jacobian(self,context):
        return self.budget.call("jacobian",lambda:np.concatenate([np.eye(20),np.zeros((172,20))]))

    def step(self,context,packet):
        def work():
            c=dict(context)
            c["latent_history"]=[(np.asarray(context["latent_history"][-1])+float(packet.sum())/100).tolist()]
            return c
        return self.budget.call("packet",work)


def test_separate_compile_search_per_arm_and_no_padding():
    case=dict(base=0,context={"latent_history":[[0.0]*192],"action_history":[]},goal=[0.0]*192)
    events=[];results=[]
    for arm in m.ARMS:
        results.append(m.run_branch(case,arm,FakeAdapter,events.append,model_scope=mock_scope))
    for r in results:
        c=r["resource_counts"]
        assert c["attempted"]["jacobian"]==c["attempted"]["svd"]==1
        assert c["attempted"]["packet"]==34 and c["attempted"]["score"]==17
        assert c["attempted"]["predict"]==0  # fixture: no fabricated calls
        assert 1<=r["unique_action_sequences"]<=16
    assert len([e for e in events if e["event"]["kind"]=="branch_start"])==2
    assert [r["arm"] for r in results]==list(m.ARMS)


def test_per_arm_cap_and_failure_no_retry():
    events=[];b=m.Budget(events.append)
    for _ in range(m.CAPS["score"]):b.call("score",lambda:1)
    called=[]
    with pytest.raises(ValueError):b.call("score",lambda:called.append(1))
    assert not called and b.attempted["score"]==b.returned["score"]==17


def test_model_instrumentation_required():
    case=dict(base=0,context={},goal=[])
    with pytest.raises(ValueError,match="instrumentation"):
        m.run_branch(case,"joint",FakeAdapter,lambda _:None)
