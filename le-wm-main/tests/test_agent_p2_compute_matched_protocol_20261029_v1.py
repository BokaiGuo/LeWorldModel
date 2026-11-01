"""Model-free preregistration tests; no production or physical outcomes."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_compute_matched_protocol_20261029_v1 as m


def test_full_vector_refuses_predict_or_jacobian_only_equivalence():
    a=m.budget('composition');b=m.budget('joint_two_pass')
    assert a['jacobian']==b['jacobian']==85
    assert a['mathematical_VJP']==b['mathematical_VJP']==16320
    assert a['predict']==936 and b['predict']==1902
    assert a['jacobian_elements']==192*(30+42*20+42*10)
    assert b['jacobian_elements']==192*(30+84*30)
    assert not m.compare(a,b)['matched']
    assert not m.compare(a,a)['matched']  # unknown FLOPs/backward internals


@pytest.mark.parametrize('key', ['predict','jacobian','mathematical_VJP','jacobian_elements','AD_rollout_steps'])
def test_resource_mutation_cannot_be_hidden(key):
    a=m.budget('composition');b=deepcopy(a);b[key]+=1
    assert key in m.compare(a,b)['different']


def test_full_schedule_fixed_before_outcomes():
    s=m.schedule()
    assert len(s)==2*42*2
    assert [(x['root'],x['seed'],x['pass']) for x in s]==[(r,i,p) for r in (0,1) for i in range(42) for p in (0,1)]
    assert all(x['dimensions']==30 and x['horizon']==3 and x['alphas']==[1.,.5,.25,.125] for x in s)


def test_prepare_is_explicitly_blocked_not_equal_compute():
    p=m.protocol({'report_sha256':'a','receipt_sha256':'b'})
    assert p['status']=='blocked_no_vector_compute_match'
    assert p['run_authorized'] is p['equal_FLOPs'] is p['P2_admission'] is False
    assert p['comparison']['different'] and p['comparison']['unknown']


class Backend:
    def __init__(self):self.f=0;self.j=0
    def rollout(self,c,commands,tag):
        self.f+=len(commands)
        return [dict(latent_history=[[0.]*192]*3,action_history=[[0.]*10]*2) for _ in commands]
    def jacobian(self,c,u,tag,logit):
        self.j+=1;self.last_ad_contexts=self.rollout(c,m.old.chart(u),tag+'/forward')
        return np.zeros((192,30))


def test_executable_local_rule_keeps_incumbent_on_ties_and_pays_all_calls():
    b=Backend();c=dict(latent_history=[[0.]*192]*3,action_history=[[0.]*10]*2)
    u=np.zeros(30);x=m.local_joint(c,u,[0.]*192,b,'RAM')
    assert x['selected_index']==0 and x['logits']==u.tolist()
    assert b.f==21 and b.j==1
    assert len(x['evaluations'])==5


def test_illegal_incumbent_rejected_not_clipped():
    with pytest.raises(ValueError):m.local_joint({},np.ones(30)*10,[0.]*192,Backend(),'RAM')


def test_runtime_vector_missing_and_extra_fields_fail_closed():
    with pytest.raises(ValueError):m.validate_ledger({'predict':936},m.budget('composition'))
    a=m.budget('composition');a['unknown_padding']=0
    with pytest.raises(ValueError):m.validate_ledger(a,m.budget('composition'))


def test_all42_two_passes_pay_derived_full_denominator():
    b=Backend();c=dict(latent_history=[[0.]*192]*3,action_history=[[0.]*10]*2)
    result=m.joint_candidates(dict(packet=dict(root_context=c,goal=[0.]*192)),
        dict(params=[np.zeros(30).tolist() for _ in range(42)]),b)
    assert len(result['candidates'])==42 and result['winner']['index']==0
    assert b.f+6==1902 and b.j+1==85  # caller must separately pay shared6/rootJ


def test_prepare_binds_real_schema_authority_and_rebuilds_strict(tmp_path,monkeypatch):
    import json
    output=tmp_path/'planner';output.mkdir();receipt=tmp_path/'strict.json'
    (output/'report.json').write_text(json.dumps(dict(prepared='/saved/p.json',prepare_sha256='prepareSHA',artifact_sha256={'root0.json.gz':'rootSHA'})))
    receipt.write_text(json.dumps(dict(plans=[0,1,2,3])))
    calls=[]
    monkeypatch.setattr(m.old,'consume',lambda out,h:(calls.append((out,h)) or dict(plans=[0,1,2,3])))
    monkeypatch.setattr(m.old,'verify',lambda *a:dict(slots=[dict(root_slot=i,condition='off',task_id=str(i),task_sha256='t') for i in (0,1)],bindings_sha256={'/saved/model':'modelSHA'}))
    original_sha=m.old.sha
    def sha(path):
        p=str(path)
        if p==str(m.old.__file__):return m.SOURCE_SHA
        if p==str(output/'report.json'):return m.REPORT_SHA
        if p==str(receipt):return m.RECEIPT_SHA
        return 'ownFileSHA'
    monkeypatch.setattr(m.old,'sha',sha)
    p=m.build(output,receipt)
    assert calls==[(output,m.REPORT_SHA)]
    assert p['bindings_sha256']['/saved/model']=='modelSHA'
    assert p['bindings_sha256'][str(output/'root0.json.gz')]=='rootSHA'
    assert p['authority']['receipt_sha256']==m.RECEIPT_SHA
    receipt.write_text(json.dumps(dict(plans=[0,1,2])))
    with pytest.raises(ValueError):m.build(output,receipt)
    monkeypatch.setattr(m.old,'sha',original_sha)


def test_mismatched_report_rejected_before_strict_rebuild(tmp_path,monkeypatch):
    monkeypatch.setattr(m.old,'sha',lambda p:m.SOURCE_SHA if str(p)==str(m.old.__file__) else 'bad')
    monkeypatch.setattr(m.old,'consume',lambda *a:pytest.fail('unbound strict call'))
    with pytest.raises(ValueError):m.build(tmp_path,tmp_path/'receipt')
