from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_p1_fixed_prefix_refinement_20261029_v1 as m
import test_agent_nonlinear_suffix_observed_recompile_20261028_v1 as fixture

inputs=fixture.inputs
complete=fixture.complete


@pytest.fixture
def group(complete):
    request,_,_=fixture.make_request(complete);task=request['task'];backend=fixture.Arithmetic(task)
    x=[0.]*10;y=[0.]*10;prefix=m.F.chart(x);suffix=m.F.chart(y)
    mid=backend.after(task['root_context'],prefix);end=backend.after(mid,suffix)
    return dict(index=0,slot=0,source_arm='stage_descent',source_iteration=1,task=task,prefix_logits=x,suffix_logits=y,
                expected_raw_bits=m.F.bits([*prefix,*suffix]),expected_mid=mid,expected_end=end,
                initial_Q=m.F.cost(task['goal'],task['root_context']),baseline_Q=m.F.cost(task['goal'],end),source_report_sha256='a'*64)


def test_full_group_same_prefix_and_exact_cost(group):
    b=fixture.Arithmetic(group['task']);events=[];r=m.run_group(group,b,events.append)
    assert len(b.calls)==19 and sum(c['kind']=='ADforward' for c in b.calls)==8
    assert len(r['iterates'])==9 and len(r['gradients'])==8
    assert all(c['context']==group['expected_mid'] for c in b.calls[1:])
    assert all(v['witness']==(v['uphill_from_initial'] and v['better_terminal_than_original']) for v in r['iterates'])
    assert not r['physical_release'] and not r['P1_admission']


@pytest.mark.parametrize('bad',['prefix','baseline','context','outcome','domain'])
def test_fail_closed(group,bad):
    q=deepcopy(group)
    if bad=='prefix':q['expected_raw_bits']='00'
    elif bad=='baseline':q['initial_Q']+=1
    elif bad=='context':q['expected_mid']['latent_history'][-1][0]+=1
    elif bad=='outcome':q['reward']=1
    else:q['prefix_logits']=[10.]*10
    with pytest.raises(ValueError):m.run_group(q,fixture.Arithmetic(q['task']),lambda e:None)


def test_real_saved_64_group_identity_only():
    root=m.F.ROOT/'results';read=m.F.signal.read
    p=read(root/'agent_p1_two_packet_host_cuda_20261029_v1_prepare_MAIN01.json',128*1024**2)
    r=read(root/'agent_p1_two_packet_host_cuda_20261029_v1_cuda_MAIN01/report.json',128*1024**2)
    u=read(root/'agent_p1_two_packet_host_cuda_20261029_v1_consume_MAIN01.json',128*1024**2)
    rows=m.requests(p,r,u,'73e0a684f3641c343d02d2537344479e47854ae4dbdc55cf181cad14f7042a3a')
    assert len(rows)==64 and [x['index'] for x in rows]==list(range(64))
    assert [(x['source_arm'],x['source_iteration']) for x in rows[:16]]==[(a,i) for a in m.parent.ARMS for i in range(1,9)]
    assert m.contract()['new_total']==dict(predict=1216,scalar_reverse=512)


def test_no_group_filter_or_partial_fallback(group):
    groups=[]
    for i in range(64):
        q=deepcopy(group);q.update(index=i,slot=i//16,source_arm=m.parent.ARMS[(i%16)//8],source_iteration=i%8+1);groups.append(q)
    def fail(q):raise RuntimeError('injected')
    p=m.panel(groups,fail,lambda e:None)
    assert p['status']=='partial_failclosed' and len(p['rows'])==64 and p['candidate_denominator']==576
    assert p['rows'][0]['status']=='failed_partial' and all(x['status']=='not_started' for x in p['rows'][1:])
