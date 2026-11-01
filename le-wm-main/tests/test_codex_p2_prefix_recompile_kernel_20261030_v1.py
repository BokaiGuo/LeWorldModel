from contextlib import contextmanager

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p2_prefix_recompile_kernel_20261030_v1 as m


@contextmanager
def scope(_budget):
    yield


class FakeAdapter:
    def __init__(self,budget):
        self.budget=budget
        self.prefixes=[]

    def root_jacobian(self,context):
        def work():
            for _ in range(2):
                self.budget.call('packet',lambda:None)
            return np.vstack((np.diag(np.arange(20,0,-1)),np.zeros((172,20))))
        return self.budget.call('jacobian',work)

    def packet_jacobian(self,context):
        def work():
            self.budget.call('packet',lambda:None)
            self.prefixes.append(m.old.digest(context))
            return np.vstack((np.diag(np.arange(10,0,-1)),np.zeros((182,10))))
        return self.budget.call('jacobian',work)

    def step(self,context,packet):
        def work():
            value=float(np.asarray(packet).reshape(-1)@np.arange(1,11))/100
            c=dict(context)
            z=np.asarray(context['latent_history'][-1]).copy()
            z[0]+=value
            c['latent_history']=[z.tolist()]
            return c
        return self.budget.call('packet',work)


def case():
    return dict(base=0,context={'latent_history':[[0.0]*192],'action_history':[]},goal=[0.1]*192)


def test_both_arms_same_limits_and_independent_prefixes():
    results=[];adapters=[];events=[]
    def factory(b):
        a=FakeAdapter(b);adapters.append(a);return a
    for arm in m.ARMS:
        results.append(m.run_branch(case(),arm,factory,events.append,scope))
    joint,chain=results
    assert len(joint['candidates'])==len(chain['candidates'])==16
    assert joint['resource_counts']['attempted']['jacobian']==1
    assert joint['resource_counts']['attempted']['packet']==36
    assert chain['resource_counts']['attempted']['jacobian']==5
    assert chain['resource_counts']['attempted']['svd']==5
    assert chain['resource_counts']['attempted']['packet']==28
    assert chain['resource_counts']['attempted']['score']==17
    assert len(adapters[1].prefixes)==4 and len(set(adapters[1].prefixes))==4
    assert [x['context_sha256'] for x in chain['prefix_contexts']]==adapters[1].prefixes
    assert all(r['resource_counts']['attempted']==r['resource_counts']['returned'] for r in results)
    assert all(r['P2_admission'] is False for r in results)


def test_failure_no_substitution_or_retry():
    events=[]
    class Broken(FakeAdapter):
        def packet_jacobian(self,context):
            return self.budget.call('jacobian',lambda:(_ for _ in ()).throw(RuntimeError('failure')))
    with pytest.raises(RuntimeError,match='failure'):
        m.run_branch(case(),'prefix_chain',Broken,events.append,scope)
    assert sum(e['event']['kind']=='resource_failure' for e in events)==1
    assert not any(e['event']['kind']=='branch_complete' for e in events)
    assert sum(e['event']['kind']=='prefix_return' for e in events)==1


def test_caps_block_before_dispatch():
    calls=[];meter=m.Budget(lambda _:None)
    for _ in range(m.CAPS['score']):
        meter.call('score',lambda:None)
    with pytest.raises(ValueError,match='budget predispatch'):
        meter.call('score',lambda:calls.append(1))
    assert calls==[]


def test_packet_jacobian_shape_rejected():
    with pytest.raises(ValueError):
        m.local_basis(np.zeros((192,20)))


def test_replay_mismatch_is_failure():
    events=[]
    class Drift(FakeAdapter):
        def __init__(self,b):
            super().__init__(b);self.calls=0

        def step(self,context,packet):
            self.calls+=1
            result=super().step(context,packet)
            if self.calls==22:
                result['latent_history'][0][0]+=1
            return result
    with pytest.raises(ValueError,match='whole winner replay exact'):
        m.run_branch(case(),'prefix_chain',Drift,events.append,scope)
    assert not any(e['event']['kind']=='branch_complete' for e in events)


def test_cpu_device_rejected_before_torch_or_model_use():
    from experiments.spectral_reachability.codex_p2_prefix_recompile_cuda_20261030_v1 import PrefixCudaAdapter
    class Device:
        type='cpu'
    with pytest.raises(ValueError,match='CUDA'):
        PrefixCudaAdapter(None,Device(),m.Budget(lambda _:None))
