"""RAM-only joins and metadata writer fixtures; no model/physics/real prepare."""
from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_execution_assets_20261028_v1 as m


def fixture():
    norm=m.physical.helpers.NORMALIZATION;cases=[];compiled={};rows=[];plans=[]
    for i in range(4):
        r=str(i+1)*64
        initial=dict(m.physical.planner.META,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
        c=dict(input_context=dict(emb=[initial['latent_history']],actions=[np.zeros((3,10)).tolist()]),
            goal_vectors=[np.zeros(192).tolist()],scoring=dict(metric=dict(projection=np.eye(192).tolist(),sha256='f'*64)))
        case=dict(root_index=i,root_sha256=r,root=dict(context=initial),goal_vector_sha256=m.digest(c['goal_vectors'][0]),
            projection_sha256=m.digest(c['scoring']['metric']['projection']),metric_sha256='f'*64)
        results=[]
        for arm in m.ARMS:
            raw=np.full((10,2),.1,dtype=np.float32).tolist()
            tokens=[m.saved.native._normalized_token(raw[:5]),m.saved.native._normalized_token(raw[5:])]
            mid=dict(initial,latent_history=np.zeros((3,192)).tolist(),action_history=[initial['action_history'][-1],tokens[0]])
            end=dict(initial,latent_history=np.zeros((3,192)).tolist(),action_history=tokens)
            edges=[dict(actions=raw,input_context=initial,output_context=end)] if arm==m.ARMS[0] else [
                dict(actions=raw[:5],input_context=initial,output_context=mid),dict(actions=raw[5:],input_context=mid,output_context=end)]
            w=dict(edges=edges,receipt=dict(actions=raw,terminal_context=end,receipt_sha256='e'*64))
            best=dict(candidate_id=arm+'_00',cost=0.,actions=raw,witness=w)
            results.append(dict(arm=arm,attempts=[best],winner=dict(candidate_id=best['candidate_id'],witness=w,independent_full_replay_exact=True)))
            plans.append(dict(root_slot=i,root_sha256=r,arm=arm,commands=raw,selection_digest=m.digest(best),
                candidate_id=best['candidate_id'],native_receipt_sha256='e'*64,report_sha256=m.REPORT_SHA,physical_execution_authorized=False))
        rows.append(dict(root_slot=i,root_sha256=r,results=results));compiled[r]=c;cases.append(case)
    return dict(selected_plans=plans,report_sha256=m.REPORT_SHA),cases,compiled,rows,norm


def test_all8_plans_own_chain_and_old_branch_abi():
    plans=m.make_plans(*fixture())
    assert len(plans)==8
    assert [(p['root_slot'],p['arm']) for p in plans]==[(i,a) for i in range(4) for a in (m.ARMS if i%2==0 else m.ARMS[::-1])]
    for p in plans:
        assert p['schema']==m.NAME+'.plan' and p['execution_plan']['schema']==m.physical.NAME+'.plan'
        assert p['handoff_sha256']==m.digest({k:v for k,v in p.items() if k!='handoff_sha256'})
        assert not p['P1_admission'] and not p['P2_admission']
        m.physical.validate_plan(p['execution_plan'])


@pytest.mark.parametrize('field',['subset','duplicate','root','arm','commands','digest','receipt','report','authority','candidate','goal','context','shape','float','edge'])
def test_reject_crosswire(field):
    lib,cases,compiled,rows,norm=deepcopy(fixture());p=lib['selected_plans'][0]
    if field=='subset':lib['selected_plans'].pop()
    elif field=='duplicate':lib['selected_plans'][-1]=deepcopy(p)
    elif field=='root':p['root_sha256']='f'*64
    elif field=='arm':p['arm']='new'
    elif field=='commands':p['commands'][0][0]=.5
    elif field=='digest':p['selection_digest']='0'*64
    elif field=='receipt':p['native_receipt_sha256']='0'*64
    elif field=='report':p['report_sha256']='0'*64
    elif field=='authority':p['physical_execution_authorized']=True
    elif field=='candidate':p['candidate_id']='invented'
    elif field=='goal':compiled[cases[0]['root_sha256']]['goal_vectors'][0][0]=1.
    elif field=='context':cases[0]['root']['context']['action_history'][0][0]=1.
    elif field in ('shape','float'):
        best=rows[0]['results'][0]['attempts'][0]
        raw=np.zeros((9,2)).tolist() if field=='shape' else np.full((10,2),.123456789).tolist()
        best['actions']=p['commands']=best['witness']['receipt']['actions']=raw
        p['selection_digest']=m.digest(best)
    else:
        best=rows[0]['results'][0]['attempts'][0];best['witness']['edges']*=2;p['selection_digest']=m.digest(best)
    with pytest.raises((ValueError,KeyError)):m.make_plans(lib,cases,compiled,rows,norm)


def test_denominator_not_dependent_on_success():
    p=dict(plans=m.make_plans(*fixture()));rows=m.initial_rows(p)
    assert len(rows)==8 and all(x['status']=='not_started' and x['physical'] is None for x in rows)
    rows[0]['status']='failed_partial'
    assert len(rows)==8 and sum(x['status']=='not_started' for x in rows)==7


def test_contract_no_neural_and_no_science_release():
    c=m.contract();assert c['selected_step_cap']==80 and c['warmup_step_cap']==120
    assert c['new_predict']==c['new_encoder']==c['new_projector']==c['AD']==0
    assert not any(c[k] for k in ('P1_admission','P2_admission','formal_execution_authorized','equal_full_resource_cost','independent_cohort'))
    assert c['internal_physics_substeps'] is c['reset_internal_work'] is None


def test_verify_rebuild_failclosed(monkeypatch):
    p=dict(schema=m.NAME+'.prepare',plans=[1])
    monkeypatch.setattr(m,'sha',lambda p:'a'*64);monkeypatch.setattr(m,'read',lambda *a:deepcopy(p))
    monkeypatch.setattr(m,'build',lambda:deepcopy(p))
    assert m.verify('virtual','a'*64)==p
    with pytest.raises(ValueError):m.verify('virtual','b'*64)
    monkeypatch.setattr(m,'build',lambda:dict(p,plans=[2]))
    with pytest.raises(ValueError):m.verify('virtual','a'*64)


def test_prepare_size_denial_no_write(monkeypatch):
    monkeypatch.setattr(m,'build',lambda:dict(large='x'*(m.PREPARE_CAP+1)))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=10**12))
    called=[];monkeypatch.setattr(m,'write',lambda *a:called.append(a))
    with pytest.raises(ValueError):m.prepare('virtual_new.json')
    assert not called


def test_prepare_floor_denial_no_write(monkeypatch):
    monkeypatch.setattr(m,'build',lambda:dict(small=True))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=m.FLOOR))
    called=[];monkeypatch.setattr(m,'write',lambda *a:called.append(a))
    with pytest.raises(ValueError):m.prepare('virtual_new.json')
    assert not called


def test_pins():
    assert m.sha(m.saved.__file__)==m.SAVED_SHA and m.sha(m.physical.__file__)==m.PHYSICAL_SHA
