from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import pytest
from experiments.spectral_reachability import agent_p3_c3_budget_schedule_saved_20261029_v1 as m


def fixture():
    caps={a:20 for a in m.AXES};attempted={a:6 for a in m.AXES};completed={a:5 for a in m.AXES}
    base=dict(slot=0,root=0,K=1,mode='fixedH1',status='resource_cap_before_node_queue',plan_available=False,
              resources=dict(caps=caps,attempted=attempted,completed=completed),R_groups=0,eligible_counts=[],
              native_calls=1,AD_primal_predict_API=1,Jacobian_elements=192,wall_ns=12,inclusive_operation_wall_ns={'predict':4})
    requests=[dict(id=i) for i in range(8)]
    decision=dict(candidates=4,continue_expanding=True,horizon=1,extension_type='progress',program_type='progress')
    queues=[dict(status='queued',requests=requests[:4],decision=decision,minimum=caps,verification_reserve=caps),
            dict(status='resource_cap_before_node_queue',requests=requests[4:],decision=decision,minimum=caps,verification_reserve=caps)]
    jobs=[dict(slot=i,status='returned_eligible' if i<4 else 'not_requested',request=requests[i] if i<4 else None) for i in range(144)]
    tree=dict(schema=m.prior.original.NAME+'.tree',backend_kind='HOST_CUDA',status=base['status'],mode=base['mode'],K=base['K'],
              resource_receipt=base['resources'],jobs=jobs,queues=queues)
    return tree,base


def test_refused_is_separate_from_unassigned():
    tree,base=fixture();row=m.summarize_tree(0,tree,base)
    assert row['assigned_jobs']==4 and row['unassigned_job_identities']==140
    assert row['generated_refused_requests']==4 and row['unused_not_assigned_to_refused_requests']==136
    assert row['axis'][m.AXES[0]]==dict(cap=20,attempted=6,completed=5,unspent_cap=14,attempted_not_completed=1)
    assert len(row['jobs'])==144 and row['FLOPs'] is None


@pytest.mark.parametrize('fault',['lost_slot','wrong_request','wrong_axis','bad_status','wrong_count','wrong_tree'])
def test_tamper_rejected(fault):
    tree,base=deepcopy(fixture())
    if fault=='lost_slot':tree['jobs'].pop()
    elif fault=='wrong_request':tree['jobs'][0]['request']={'id':99}
    elif fault=='wrong_axis':tree['resource_receipt']['attempted'][m.AXES[0]]=21
    elif fault=='bad_status':tree['jobs'][0]['status']='unknown'
    elif fault=='wrong_count':tree['queues'][0]['requests'].pop()
    else:tree['mode']='different'
    with pytest.raises((ValueError,KeyError)):m.summarize_tree(0,tree,base)


def test_saved_build_inherits_receipt_and_reader(tmp_path):
    tree,template=fixture();bases=[]; rows=[]
    for i in range(12):
        b=deepcopy(template);b.update(slot=i,root=i//6,K=1 if i%6<3 else 17,mode=('fixedH1','staticH3_progress','adaptive')[i%3]);bases.append(b)
        rows.append(dict(slot=i,root=b['root'],K=b['K'],mode=b['mode'],status=b['status'],result={'slot':i}))
    receipt=dict(manifest_sha256='a'*64,completion_sha256='b'*64,report_sha256='c'*64)
    class Reader:
        def __init__(self,*a):self.report=dict(schema=m.HOST+'.report',status='twelve_slots_settled_not_admitted',rows=rows)
        def result(self,i,ref):
            t=deepcopy(tree);t.update(mode=bases[i]['mode'],K=bases[i]['K']);return t
        def close(self):pass
    with patch.object(m.prior,'build',return_value={'rows':bases}),patch.object(m.prior,'read_pinned',return_value=receipt),\
         patch.object(m.host,'Reader',Reader),patch.object(m.host.codec,'file_sha',return_value='c'*64):
        result=m.build(str(tmp_path/'receipt.json'),'d'*64,str(tmp_path))
    assert result['denominator']==12 and result['request_identity_denominator']==1728 and len(result['rows'])==12
