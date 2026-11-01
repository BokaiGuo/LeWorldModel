"""E3 K1/K3 GD at actual boundary 10/15; caller must authenticate context.

No model loading, physical execution, receipt or production authorization here.
"""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_e3_gd_host_cuda_20261029_v1 as host

n = host.numerical
need = host.need


def validate_body(body):
    r = n.resolver
    need(type(body) is dict and set(body) == set(r.PLANNER_FIELDS), 'exact planner whitelist')
    need(body['schema'] == r.NAME+'.task' and body['model_sha256'] == r.MODEL_SHA
         and body['root_sha256'] in r.KNOWN_ROOTS, 'task identity')
    k = body['horizon_packets']; n.packet_budget(k)
    need(body['domain'] == r.domain_spec(body['domain']['name']), 'original domain')
    r.cp5.context(body['root_context'])
    r.array(body['goal'], (192,), f32=True)
    need(np.array_equal(r.array(body['projection'], (192,192), f32=False), np.eye(192)), 'identity Q')
    r.array(body['action_mean'], (2,), f32=True)
    need((r.array(body['action_std'], (2,), f32=True)>0).all(), 'positive normalizer')
    need(type(body['feedback_boundary']) is int and body['feedback_boundary'] in (10,15)
         and type(body['task_deadline_control']) is int and body['task_deadline_control']==20, 'actual later boundary')
    r.cp5.hash_value(body['identity_sha256'])
    return deepcopy(body)


def solve(body, backend, event):
    b=validate_body(body); k=b['horizon_packets']; domain=b['domain']['name']
    goal=np.asarray(b['goal'],float); candidates=[]; gradients=[]
    def evaluate(x,tag):
        a=n.resolver.mask_chart(x.tolist(),k,domain); replay=tag.startswith('replay')
        if replay:event(dict(kind='replay_start',tag=tag))
        c=backend.forward(b['root_context'],a,tag)
        event(dict(kind='metric_start',tag=tag))
        cost=float(np.mean((np.asarray(c['latent_history'][-1],float)-goal)**2))
        event(dict(kind='metric_end',tag=tag))
        value=dict(logits=x.tolist(),actions=a.tolist(),context=c,cost=cost)
        if replay:event(dict(kind='replay_end',tag=tag))
        event(dict(kind='candidate',tag=tag,value=value));return value
    current=evaluate(np.zeros((k,5,2)),'iterate0');candidates.append(current)
    for i in range(32):
        c,g,loss=backend.gradient(b['root_context'],np.asarray(current['actions'],np.float32),goal,'AD'+str(i))
        host.same(c,current['context'],'fresh full AD primal exact incumbent')
        need(isinstance(g,np.ndarray) and g.dtype==np.float32 and g.shape==(5*k,2) and np.isfinite(g).all(),'full raw gradient')
        host.scalar_bound(loss,c['latent_history'][-1],goal)
        derivative=n.resolver.mask_gradient((g.astype(float)*(1-np.asarray(current['actions'],float)**2)).reshape(k,5,2).tolist(),k,domain)
        gradients.append(dict(raw_gradient=g.tolist(),loss=float(loss),objective_gradient=derivative.tolist()))
        x=n.project(np.asarray(current['logits'])-derivative,k,domain)
        current=evaluate(x,'iterate'+str(i+1));candidates.append(current)
    winner=min(range(33),key=lambda i:(candidates[i]['cost'],i));event(dict(kind='selection',winner=winner))
    for i in range(3):host.same(evaluate(np.asarray(candidates[winner]['logits']),'replay'+str(i)),candidates[winner],'independent replay exact')
    return dict(candidates=candidates,gradients=gradients,winner=winner,commands=candidates[winner]['actions'],
                context=candidates[winner]['context'],first5=candidates[winner]['actions'][:5])
