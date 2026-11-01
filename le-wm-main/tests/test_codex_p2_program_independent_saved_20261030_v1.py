from contextlib import contextmanager
import numpy as np
import pytest
from experiments.spectral_reachability import codex_p2_program_independent_budget_20261030_v1 as b
from experiments.spectral_reachability import codex_p2_program_independent_saved_20261030_v1 as s


@contextmanager
def scope(_):
    yield


class Sim:
    def __init__(self,m):self.m=m
    def nested(self):
        return self.m.call("packet",lambda:self.m.call("action_encoder",lambda:self.m.call("predict",
                lambda:self.m.call("predictor",lambda:self.m.call("pred_proj",lambda:None)))))
    def root_jacobian(self,c):
        def work():
            self.nested();self.nested()
            return np.concatenate([np.eye(20),np.zeros((172,20))])
        return self.m.call("jacobian",work)
    def step(self,c,a):
        # Test the census with fake nested callbacks, not real model inference.
        def work():
            return self.m.call("action_encoder",lambda:self.m.call("predict",
                lambda:self.m.call("predictor",lambda:self.m.call("pred_proj",lambda:c))))
        return self.m.call("packet",work)


def test_saved_journal_structure_and_tamper():
    case={"base":0,"context":{"latent_history":[[0.0]*192],"action_history":[]},"goal":[0.0]*192}
    events=[];row=b.run_branch(case,"joint",Sim,events.append,model_scope=scope)
    # Fake callbacks deliberately mimic the census but cannot authenticate CUDA.
    assert row["resource_counts"]["attempted"]["packet"]==36
    result=s.validate_branch(case,row,events)
    assert result["status"]=="branch_structure_and_resource_checked_not_cuda_authentication"
    events[-1]["event"]["result_sha256"]="bad"
    with pytest.raises(ValueError):
        s.validate_branch(case,row,events)
