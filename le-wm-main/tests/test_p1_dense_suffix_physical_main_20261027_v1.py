from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import p1_dense_suffix_physical_main_20261027_v1 as m


def test_fixed_bank_all_legal_persistent_not_sparse():
    bank=m.suffixes(); assert len(bank)==17 and bank[0]['logits']==[0.]*10
    assert [p['index'] for p in bank]==list(range(17))
    for p in bank[1:]:
        x=np.asarray(p['logits']); assert np.array_equal(x,x.astype(np.float32))
        assert np.linalg.norm(x)<=p['radius']<=1.5<.75*np.sqrt(5)
        assert np.allclose(np.linalg.norm(x),p['radius'],atol=1e-6)
        assert np.array_equal(x.reshape(5,2),np.tile(x[:2],(5,1)))
    for start in (1,9):
        for i in range(4):
            np.testing.assert_array_equal(bank[start+i]['logits'],-np.asarray(bank[start+i+4]['logits']))


def test_actual_model_only_roles():
    a=m.collect(); assert len(a['plans'])==102 and len(a['roles'])==6
    for i in range(0,102,17):
        plans=a['plans'][i:i+17]; role=a['roles'][i//17]
        assert [p['probe'] for p in plans]==m.suffixes()
        for p in plans:
            assert p['commands'][:5]==role['prefix']
            np.testing.assert_array_equal(p['commands'][5:],np.tanh(np.asarray(p['probe']['logits'],np.float32)).reshape(5,2))
            assert p['selection_sha256']==m.source.digest({k:v for k,v in p.items() if k!='selection_sha256'})
    assert a['roles'][4]['prefix']==a['roles'][5]['prefix']  # known role duplication retained


def fixture_rows():
    plans=[];rows=[]
    checkpoint=dict(state=[0.]*7,rgb_sha256='f'*64)
    for root in ('r0','r1'):
        for role in m.sparse.ROLES:
            for p in m.suffixes():
                plan=dict(root_sha256=root,role=role,arm=role+str(p['index']),selection_sha256=role+str(p['index']),probe=p)
                row=dict(root_sha256=root,arm=plan['arm'],selection_sha256=plan['selection_sha256'],status='completed',full_horizon=True,
                    checkpoints=[deepcopy(checkpoint) for _ in range(3)],body_dynamics={'5':{'recorded':'same'}},
                    final_state=[0.,0.,p['radius'],0.,0.,0.,0.],official_evaluation=dict(success=False),
                    pose_diagnostics=dict(independent=dict(wrapped_angle_error=0.)))
                plans.append(plan);rows.append(row)
    return plans,rows,dict(r0=[0.]*7,r1=[0.]*7)


def test_analysis_keeps_all_and_no_action_selection():
    p,r,g=fixture_rows(); a=m.analyze(p,r,g)
    assert len(a['groups'])==6 and not a['P1_admission'] and not a['controller_selected']
    assert all(x['planned']==x['completed']==17 for x in a['groups'])
    assert a['groups'][0]['responses'][-1]['response']['object_xy_delta']==[1.5,0.]


def test_partial_kept_no_denominator_replacement():
    p,r,g=fixture_rows(); r[3].update(status='environment_terminated',full_horizon=False)
    a=m.analyze(p,r,g); group=a['groups'][0]
    assert group['completed']==16 and len(group['responses'])==group['planned']==17
    assert group['responses'][3]['response'] is None


@pytest.mark.parametrize('defect',['state','rgb','dynamics','identity','bank','census'])
def test_analysis_refuses_mismatch(defect):
    p,r,g=fixture_rows()
    if defect=='state': r[1]['checkpoints'][1]['state'][0]=1.
    elif defect=='rgb': r[1]['checkpoints'][1]['rgb_sha256']='e'*64
    elif defect=='dynamics': r[1]['body_dynamics']['5']['recorded']='different'
    elif defect=='identity': r[1]['selection_sha256']='wrong'
    elif defect=='bank': p[1]['probe']['radius']=100.
    else: r.pop()
    with pytest.raises(ValueError):m.analyze(p,r,g)


def test_authorization_precedes_verify(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a: (_ for _ in ()).throw(AssertionError('must not read')))
    with pytest.raises(ValueError,match='authorization'):m.run('none','none','none',lambda:None)


def test_prepare_material_no_writes():
    p=m.material(); assert p['schema']==m.NAME+'.prepare'
    assert len((m.source.canonical(p)+'\n').encode())<1024**2
    assert p['contract']['step_cap']==1020 and len(p['inputs']['plans'])==102


@pytest.mark.parametrize('failure',['factory','branch_closed','run_finished'])
def test_run_faults_keep_all_planned_and_attempt_final_report(monkeypatch,failure):
    plans,_,_=fixture_rows(); saved=[]
    monkeypatch.setattr(m,'verify',lambda *a:dict(inputs=dict(plans=plans)))
    monkeypatch.setattr(m.base,'sha',lambda *a:'f'*64)
    class Store:
        path=Path('/memory_only'); active_body=None
        def __init__(self,*a): self.events=[]
        def check(self,*a): return None
        def emit(self,kind,**values):
            self.events.append(kind)
            if kind==failure: raise OSError('injected_'+failure)
        def json(self,name,value): saved.append((name,deepcopy(value)))
    monkeypatch.setattr(m.sparse,'Store',Store)
    def factory(): raise RuntimeError('factory_failure')
    result=m.run('memory','f'*64,'memory',factory,authorize_physics=True)
    assert result['status']=='partial_or_failed' and len(result['rows'])==102
    assert saved[-1][0]=='report.json' and len(saved[-1][1]['rows'])==102
    assert result['rows'][0]['first_error']['message']=='factory_failure'
    if failure=='branch_closed':
        assert result['rows'][1]['status']=='not_started'
        assert result['errors'][0]['stage']=='execution'
    elif failure=='run_finished': assert any(e['stage']=='final_journal' for e in result['errors'])
