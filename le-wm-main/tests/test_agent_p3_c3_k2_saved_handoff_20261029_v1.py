"""Pure K2 contract fixtures: no NN, physical execution, or production receipt."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_p3_c3_k2_saved_handoff_20261029_v1 as m


def sample(plan=False):
    i = m.host.k2.roster()[0]; caps=m.host.k2.caps(); zero={k:0 for k in caps}
    events=[]; rr=dict(caps=caps,attempted=zero,completed=zero,events=events)
    resources=dict(caps=caps,attempted=zero,completed=zero,events_count=0,
                   events_semantic_sha256=m.host.codec.semantic(events))
    status='tree_complete_not_admitted' if plan else 'resource_cap_before_node_queue'
    row=dict(**i,status=status); arm=dict(**i,status=status,plan_available=plan,resources=resources)
    path=[{'latent':'fixture'}]; obj=dict(actions=[[0.,0.] for _ in range(5)],path=path)
    tree=dict(schema=m.host.k2.NAME+'.tree',backend_kind='HOST_CUDA',mode=i['mode'],K=2,status=status,
        resource_receipt=rr,jobs=[dict(slot=n,status='not_requested',request=None) for n in range(144)],
        queues=[dict(status='resource_cap_before_node_queue',requests=[{'x':1}],minimum={},verification_reserve={})],
        model_plan=obj if plan else None,replays=[dict(mode=z,status='returned',path=path) for z in ('whole','segmented')])
    return i,row,tree,arm


def test_no_witness_preserves_null_and_full_denominator():
    s=m.summarize_tree(*sample())
    assert s['job_denominator']==144 and s['unassigned']==144 and s['refused_unassigned']==1
    assert s['candidate_first5_raw'] is None and s['physical_outcome'] is None
    assert not s['physical_dispatch_ready'] and len(s['axes'])==27


def test_plan_is_not_physical_authority():
    s=m.summarize_tree(*sample(True))
    assert len(s['candidate_first5_raw']) == 5 and not s['physical_dispatch_ready']
    assert len(s['model_plan_semantic_sha256']) == 64 and len(s['physical_missing']) == 4


@pytest.mark.parametrize('damage',['root','cap','ledger','replay','action','no_plan'])
def test_tampering_rejected(damage):
    i,row,t,a=sample(True);t=deepcopy(t);a=deepcopy(a)
    if damage=='root': a['root']=1
    elif damage=='cap': t['resource_receipt']['caps']['predict_API']+=1
    elif damage=='ledger': a['resources']['events_semantic_sha256']='0'*64
    elif damage=='replay': t['replays'][1]['path']=[]
    elif damage=='action': t['model_plan']['actions'][0][0]=1.1
    else: t['model_plan']=None
    with pytest.raises(ValueError):m.summarize_tree(i,row,t,a)


def test_missing_real_receipt_fail_closed(tmp_path):
    with pytest.raises(ValueError):m.build(tmp_path/'missing','0'*64,tmp_path/'out')


def test_tempdisk_pinned_bytes_reject_mutation(tmp_path):
    p=tmp_path/'value.json';p.write_bytes(m.canon({'row':None}))
    expected=m.host.codec.file_sha(p)
    assert m.pinned(p,expected)=={'row':None}
    p.write_bytes(m.canon({'row':0}))
    with pytest.raises(ValueError,match='pinned'):m.pinned(p,expected)
